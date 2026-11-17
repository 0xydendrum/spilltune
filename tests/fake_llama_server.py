"""A stand-in for llama-server used by the tests: same flags, same log lines, same HTTP endpoints, and a
made-up speed that depends on the options, so the tuner's search can be checked end to end without a GPU.

Speed model (tokens/sec): 3.4 at the defaults, +0.3 per 256 MiB less --fit-target, and with a helper a curve
that peaks at 16 guesses with streaming for DFlash on a dense model. Like real helpers, a helper slows prose
prompts (explanation, story) below no helper, unless --spec-draft-p-min is set, which works best at 0.85.
FAKE_NO_PMIN=1 hides --spec-draft-p-min from --help, like an older llama.cpp build.
"""
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HELP = """-ngl,  --gpu-layers N
-fit,  --fit [on|off]   default: 'on'
-fitt, --fit-target MiB0,...
-ncmoe, --n-cpu-moe N
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,ngram-simple
--spec-draft-n-max N
--spec-draft-p-min, --draft-p-min P     minimum speculative decoding probability (greedy) (default: 0.00)
-cram, --cache-ram N
--ui,  --webui, --no-ui, --no-webui
-lm,   --load-mode MODE
-lv,   --verbosity, --log-verbosity N
"""


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def speed(args, prose=False):
    margin = int(opt(args, "--fit-target", 1024))
    tps = 3.4 + 0.3 * (1024 - margin) / 256
    if opt(args, "--load-mode") == "none":
        tps *= 1.5
    st = opt(args, "--spec-type")
    if st:
        n = int(opt(args, "--spec-draft-n-max", 3))
        stream = os.environ.get("GGML_OP_OFFLOAD_MIN_BATCH") is not None
        peak = 16 if st == "draft-dflash" else 4
        gain = 3.0 * math.exp(-((math.log2(n) - math.log2(peak)) ** 2) / 2)
        if st == "draft-dflash" and not stream:
            gain *= 0.6
        p = float(opt(args, "--spec-draft-p-min", 0))
        if prose:
            gain = gain * 0.6 * (1 - abs(p - 0.85)) if p else -0.25
        elif p:
            gain *= 0.97
        tps *= 1 + gain
    return round(tps, 2)


def main():
    args = sys.argv[1:]
    if "--help" in args:
        h = HELP
        if os.environ.get("FAKE_NO_PMIN"):
            h = "\n".join(l for l in h.splitlines() if "p-min" not in l)
        print(h)
        return
    if "--version" in args:
        print("version: 9.9.9-fake (build 1, commit fake)")
        return
    if os.environ.get("FAKE_FAIL_IF") and os.environ["FAKE_FAIL_IF"] in " ".join(args):
        print("simulated load failure", flush=True)
        sys.exit(3)
    port = int(opt(args, "--port"))
    margin = int(opt(args, "--fit-target", 1024))
    vram_mib = int(os.environ.get("FAKE_GPU_MIB", "0"))
    if opt(args, "-lv") != "4":     # like llama-server: load details only at trace verbosity
        print("srv    load_model: loading model", flush=True)
    else:
        print_load(margin, vram_mib)
    has_helper = "--spec-type" in args
    serve(port, (speed(args), speed(args, prose=True)), has_helper)


def print_load(margin, vram_mib):
    no_gpu = bool(os.environ.get("FAKE_NO_GPU"))     # like a CUDA build whose runtime DLLs are missing
    if not no_gpu:
        print("load_backend: loaded CUDA backend from fake-ggml-cuda.dll", flush=True)
    print("load_backend: loaded CPU backend from fake-ggml-cpu.dll", flush=True)
    print("load_tensors: offloaded 31/61 layers to GPU", flush=True)
    if no_gpu:
        return
    print(f"load_tensors:        CUDA0 model buffer size =  {vram_mib or 8000 - margin}.00 MiB", flush=True)
    print("sched_reserve:      CUDA0 compute buffer size =   188.02 MiB", flush=True)
    print("sched_reserve:      CUDA0 compute buffer size =   188.02 MiB", flush=True)


def serve(port, tps, has_helper):

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"status": "ok"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            n = body.get("max_tokens") or body.get("n_predict") or 8
            msg = json.dumps(body.get("messages", ""))
            prose = "Explain" in msg or "story" in msg
            t = {"predicted_n": n, "predicted_per_second": tps[1] if prose else tps[0]}
            if has_helper:
                t.update(draft_n=n, draft_n_accepted=n // 2)
            text = "same answer " + str(len(json.dumps(body.get("messages", ""))))
            self._send({"choices": [{"message": {"content": text}}], "timings": t,
                        "usage": {"completion_tokens": n}})

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
