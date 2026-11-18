"""Start llama-server with one set of options, time generation on the test prompts, stop it."""
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CTX = 4096
BASE_ARGS = ["-c", str(DEFAULT_CTX), "-np", "1"]
SPILL_ABORT_MIB = 1024
GPU_BACKENDS = r"CUDA|ROCm|HIP|Vulkan|Metal|SYCL|MUSA|CANN|OpenCL"


class NoGPUBackend(RuntimeError):
    """The machine has a GPU but llama-server loaded no GPU backend, so every trial would run on the CPU."""


NO_GPU_HELP = ("llama-server started without a GPU backend, so everything would run on the CPU and every number "
               "would be wrong. On Windows this usually means the CUDA runtime DLLs (cudart64_*.dll, cublas64_*.dll, "
               "cublasLt64_*.dll) are neither next to llama-server.exe nor on PATH: get them from llama.cpp's "
               "cudart-llama-bin-win-cuda-*.zip release file, or add the folder that holds them to PATH. "
               "Log: {log}")


@dataclass
class Features:
    """What this llama-server build supports, read from --help."""
    version: str = "unknown"
    fit: bool = False
    fit_target: bool = False
    spec_types: list = field(default_factory=list)
    n_cpu_moe: bool = False
    cache_ram: bool = False
    no_webui: bool = False
    load_mode: bool = False
    verbosity: bool = False
    p_min: bool = False         # --spec-draft-p-min: the helper only offers guesses it is this sure of


def features(server_cmd):
    def run(flag):
        try:
            p = subprocess.run(server_cmd + [flag], capture_output=True, text=True, timeout=120,
                               encoding="utf-8", errors="replace")
            return (p.stdout or "") + (p.stderr or "")
        except (OSError, subprocess.SubprocessError):
            return ""
    h = run("--help")
    if not h:
        raise FileNotFoundError(f"could not run {' '.join(server_cmd)} --help")
    m = re.search(r"--spec-type\s+([\w,-]+)", h)
    v = re.search(r"version:\s*(.+)", run("--version"))
    return Features(version=v.group(1).strip() if v else "unknown",
                    fit=bool(re.search(r"--fit\b", h)), fit_target=bool(re.search(r"--fit-target\b", h)),
                    spec_types=m.group(1).split(",") if m else [], n_cpu_moe="--n-cpu-moe" in h,
                    cache_ram="--cache-ram" in h, no_webui="--no-webui" in h,
                    load_mode="--load-mode" in h, verbosity="--log-verbosity" in h or "--verbosity" in h,
                    p_min=bool(re.search(r"--spec-draft-p-min\b", h)))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(port, path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError):
        return False


def parse_log(text):
    """GPU layers and the memory llama.cpp placed on each GPU (MiB), from the server log."""
    m = re.findall(r"offloaded (\d+)/(\d+) layers", text)
    dev, seen = {}, set()
    for d, kind, mib in re.findall(r"(CUDA\d+|ROCm\d+|Vulkan\d+|Metal|SYCL\d+)\s+(model|KV|compute|RS)\s+buffer size"
                                   r"\s*=\s*([\d.]+) MiB", text):
        if kind == "compute":   # re-reserved (logged again) when a helper model attaches: count each size once
            if (d, mib) in seen:
                continue
            seen.add((d, mib))
        dev[d] = dev.get(d, 0.0) + float(mib)
    gpu_loaded = bool(re.search(rf"loaded ({GPU_BACKENDS}) backend", text) or dev)
    cpu_loaded = "load_backend:" in text or bool(m)    # the log shows backends at all (needs -lv 4 on new builds)
    return {"gpu_layers": f"{m[0][0]}/{m[0][1]}" if m else None,
            "gpu_mib": {k: round(v) for k, v in dev.items()},
            "gpu_backend": gpu_loaded if cpu_loaded else None}


def spill_mib(gpu_mib, gpus):
    """How far llama.cpp's buffers exceed each GPU's free memory. On Windows the NVIDIA driver does not fail
    when VRAM runs out, it silently moves the excess into system RAM (very slow), so a load that 'worked' can
    still be overflowing. Returns the worst overflow in MiB (<= 0 means it fits)."""
    worst = None
    for name, mib in gpu_mib.items():
        i = int(re.sub(r"\D", "", name) or 0)
        g = next((g for g in gpus if g.index == i), None)
        if g is not None:
            over = mib - g.free_mib
            worst = over if worst is None else max(worst, over)
    return worst


def _parallel(port, prompts, n, max_tokens):
    """n requests at once; total generated tokens / wall time."""
    import threading
    got = [0] * n

    def one(k):
        c = _post(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompts[k % len(prompts)][1]}], "max_tokens": max_tokens,
            "temperature": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}},
            timeout=3600)
        got[k] = (c.get("timings") or {}).get("predicted_n") or (c.get("usage") or {}).get("completion_tokens", 0)

    t0 = time.time()
    th = [threading.Thread(target=one, args=(k,)) for k in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = max(time.time() - t0, 1e-3)   # Windows clock ticks are ~15 ms
    return {"concurrent": n, "tokens": sum(got), "wall_s": round(wall, 1), "agg_tps": round(sum(got) / wall, 2)}


def run_trial(server_cmd, model, args, env_extra, prompts, max_tokens, log_path, gpus, load_timeout=1800,
              concurrency=None):
    """Returns a dict with speed, helper acceptance, placement and the generated texts, or an 'error' key."""
    port = free_port()
    cmd = server_cmd + ["-m", str(model), "--port", str(port)] + args
    env = dict(os.environ, **env_extra)
    rec = {"cmd": cmd, "env": env_extra}
    t0 = time.time()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while not _healthy(port):
                if proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {proc.returncode} while loading")
                if time.time() - t0 > load_timeout:
                    raise RuntimeError("model took too long to load")
                time.sleep(1)
            rec["load_s"] = round(time.time() - t0, 1)
            log.flush()
            rec.update(parse_log(Path(log_path).read_text(encoding="utf-8", errors="replace")))
            if gpus and rec["gpu_backend"] is False:
                raise NoGPUBackend(NO_GPU_HELP.format(log=log_path))
            # The log total overstates use a little (compute buffers are re-sized when a helper attaches), and
            # the driver's own counters cannot see the spill at all, so only a clear overflow stops the trial;
            # a small one is flagged and measured, and the measured speed decides.
            over = spill_mib(rec["gpu_mib"], gpus)
            rec["over_mib"] = over
            if over is not None and over > SPILL_ABORT_MIB:
                raise RuntimeError(f"needs {over} MiB more GPU memory than is free (it would spill into system RAM)")
            _post(port, "/completion", {"prompt": "Hello", "n_predict": 4, "temperature": 0})   # warm-up
            if concurrency:
                rec["levels"] = [_parallel(port, prompts, n, max_tokens) for n in concurrency]
                return rec
            runs = []
            for kind, text in prompts:
                c = _post(port, "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                    "temperature": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}})
                t = c.get("timings", {})
                runs.append({"kind": kind, "tokens": t.get("predicted_n") or 0,
                             "tps": t.get("predicted_per_second") or 0,
                             "draft_n": t.get("draft_n") or 0, "draft_ok": t.get("draft_n_accepted") or 0,
                             "text": c["choices"][0]["message"].get("content") or ""})
            n = sum(r["tokens"] for r in runs)
            secs = sum(r["tokens"] / r["tps"] for r in runs if r["tps"])
            rec["tps"] = round(n / secs, 2) if secs else 0.0
            dn = sum(r["draft_n"] for r in runs)
            rec["accept"] = round(sum(r["draft_ok"] for r in runs) / dn, 3) if dn else None
            rec["runs"] = runs
        except NoGPUBackend:        # wrong for every trial: stop the whole tune
            raise
        except Exception as e:      # one failed trial must not stop the tune
            rec["error"] = f"{e}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
            rec["total_s"] = round(time.time() - t0, 1)
    return rec
