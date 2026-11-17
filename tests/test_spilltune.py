import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spilltune import cli, gguf, hw, report, server, tune  # noqa: E402

FAKE = str(Path(__file__).with_name("fake_llama_server.py"))


def write_gguf(path, arch, n_layers, layer_mib=1, experts=0, extra_meta=None, other_mib=1):
    """Minimal GGUF v3 file with real tensor offsets (data is zeros)."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, s(arch)), (f"{arch}.block_count", 4, struct.pack("<I", n_layers))]
    if experts:
        kv.append((f"{arch}.expert_count", 4, struct.pack("<I", experts)))
    for k, v in (extra_meta or {}).items():
        kv.append((k, 4, struct.pack("<I", v)))
    kv.append(("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 100) + b"".join(s(f"t{i}") for i in range(100))))
    tensors = [("token_embd.weight", other_mib)]
    for i in range(n_layers):
        tensors.append((f"blk.{i}.attn_q.weight", layer_mib / 4))
        if experts:
            tensors.append((f"blk.{i}.ffn_up_exps.weight", layer_mib * 3 / 4))
        else:
            tensors.append((f"blk.{i}.ffn_up.weight", layer_mib * 3 / 4))
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        head += s(k) + struct.pack("<I", t) + v
    off, info = 0, b""
    sizes = []
    for name, mib in tensors:
        n = int(mib * 2 ** 20)
        info += s(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<IQ", 24, off)
        sizes.append(n)
        off += (n + 31) // 32 * 32
    head += info
    pad = (32 - len(head) % 32) % 32
    with open(path, "wb") as f:
        f.write(head + b"\0" * pad)
        f.truncate(len(head) + pad + off)


class GGUFTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_dense(self):
        p = os.path.join(self.d, "m.gguf")
        write_gguf(p, "llama", 4, layer_mib=2, extra_meta={"llama.embedding_length": 64})
        i = gguf.read(p)
        self.assertEqual((i.arch, i.n_layers, i.is_moe), ("llama", 4, False))
        self.assertAlmostEqual(i.layer_bytes[0] / 2 ** 20, 2, places=2)
        self.assertAlmostEqual(i.other_bytes / 2 ** 20, 1, places=2)
        self.assertEqual(i.meta["llama.embedding_length"], 64)
        self.assertNotIn("tokenizer.ggml.tokens", i.meta)   # large array skipped

    def test_moe(self):
        p = os.path.join(self.d, "moe.gguf")
        write_gguf(p, "qwen3moe", 3, layer_mib=4, experts=8)
        i = gguf.read(p)
        self.assertTrue(i.is_moe)
        self.assertAlmostEqual(sum(i.expert_bytes) / 2 ** 20, 9, places=2)

    def test_draft_type_and_mismatch(self):
        m, d = os.path.join(self.d, "m.gguf"), os.path.join(self.d, "x-dflash.gguf")
        write_gguf(m, "gemma4", 2, extra_meta={"gemma4.embedding_length": 64})
        write_gguf(d, "dflash", 2, extra_meta={"dflash.embedding_length": 32})
        mi, di = gguf.read(m), gguf.read(d)
        self.assertEqual(gguf.draft_type(di), "draft-dflash")
        self.assertIn("different model", gguf.helper_mismatch(mi, di))
        write_gguf(d, "dflash", 2, extra_meta={"dflash.embedding_length": 64})
        self.assertIsNone(gguf.helper_mismatch(mi, gguf.read(d)))

    def test_not_gguf(self):
        p = os.path.join(self.d, "x.gguf")
        Path(p).write_bytes(b"nope" * 10)
        with self.assertRaises(gguf.GGUFError):
            gguf.read(p)


class LogTests(unittest.TestCase):
    LOG = """load_tensors: offloaded 27/61 layers to GPU
load_tensors:        CUDA0 model buffer size =  8048.78 MiB
llama_kv_cache:      CUDA0 KV buffer size =   160.00 MiB
sched_reserve:      CUDA0 compute buffer size =   188.02 MiB
sched_reserve:  CUDA_Host compute buffer size =    47.53 MiB
sched_reserve:      CUDA0 compute buffer size =   188.02 MiB
"""

    def test_parse_counts_repeated_compute_once(self):
        r = server.parse_log(self.LOG)
        self.assertEqual(r["gpu_layers"], "27/61")
        self.assertEqual(r["gpu_mib"], {"CUDA0": round(8048.78 + 160 + 188.02)})

    def test_spill(self):
        g = [hw.GPU(0, "x", 12288, 700)]
        self.assertLess(server.spill_mib({"CUDA0": 9000}, g), 0)
        self.assertEqual(server.spill_mib({"CUDA0": 19153}, g), 19153 - 11588)
        self.assertIsNone(server.spill_mib({"CUDA0": 1}, []))


def model(moe=False, path="m.gguf"):
    return gguf.ModelInfo(path=path, arch="x", file_bytes=17 * 2 ** 30, n_layers=60, expert_count=8 if moe else 0,
                          layer_bytes=[280 * 2 ** 20] * 60, expert_bytes=[200 * 2 ** 20] * 60 if moe else [0] * 60,
                          other_bytes=700 * 2 ** 20)


FEATS = server.Features(version="t", fit=True, fit_target=True,
                        spec_types=["draft-mtp", "draft-dflash", "draft-eagle3", "draft-simple"],
                        cache_ram=True, no_webui=True)


def fake_runner(fn, calls):
    def run(t):
        calls.append(t)
        env = dict(t.env)
        tps = fn(t.args, env)
        if tps is None:
            return {"error": "simulated failure"}
        return {"tps": tps, "accept": 0.5 if "--spec-type" in t.args else None,
                "runs": [{"kind": "math", "text": "a"}]}
    return run


def speed_curve(peak, stream_helps=True):
    import math

    def fn(args, env):
        m = int(server_opt(args, "--fit-target", 1024))
        tps = 3.4 + 0.3 * (1024 - m) / 256
        if "--spec-type" in args:
            n = int(server_opt(args, "--spec-draft-n-max"))
            g = 3.0 * math.exp(-((math.log2(n) - math.log2(peak)) ** 2) / 2)
            if stream_helps and "GGML_OP_OFFLOAD_MIN_BATCH" not in env:
                g *= 0.6
            tps *= 1 + g
        return round(tps, 3)
    return fn


def server_opt(args, k, default=None):
    return args[args.index(k) + 1] if k in args else default


class TunerTests(unittest.TestCase):
    def draft(self, name="h-dflash.gguf"):
        return gguf.ModelInfo(path=name, arch="dflash", file_bytes=850 * 2 ** 20, n_layers=5)

    def test_finds_peak_dense_dflash(self):
        calls = []
        t = tune.Tuner(model(), FEATS, fake_runner(speed_curve(16), calls), self.draft(), "draft-dflash",
                       say=lambda s: None)
        res = t.tune()
        b = res["best"]
        self.assertEqual(server_opt(b["args"], "--spec-draft-n-max"), "16")
        self.assertIn("GGML_OP_OFFLOAD_MIN_BATCH", b["env"])
        self.assertLessEqual(len(calls), 14)
        # helper trials reserve the helper's VRAM on top of the best placement margin (256)
        self.assertEqual(server_opt(calls[3].args, "--fit-target"), str(256 + 850 + 1100))

    def test_moe_starts_lower_and_climbs_down(self):
        calls = []
        t = tune.Tuner(model(moe=True), FEATS, fake_runner(speed_curve(4, stream_helps=False), calls),
                       self.draft(), "draft-dflash", say=lambda s: None)
        b = t.tune()["best"]
        self.assertEqual(server_opt(b["args"], "--spec-draft-n-max"), "4")
        self.assertEqual(server_opt(calls[3].args, "--spec-draft-n-max"), "8")

    def test_placement_stops_when_not_faster(self):
        calls = []
        flat = lambda a, e: 5.0
        t = tune.Tuner(model(), FEATS, fake_runner(flat, calls), say=lambda s: None)
        res = t.tune()
        self.assertEqual(len(calls), 2)          # defaults, then 512 (no gain) -> stop
        self.assertEqual(res["best"]["label"], "baseline: llama-server defaults")

    def test_failed_helper_retries_with_more_room(self):
        calls = []

        def fn(args, env):
            if "--spec-type" in args and int(server_opt(args, "--fit-target")) < 3000:
                return None
            return speed_curve(16)(args, env)
        t = tune.Tuner(model(), FEATS, fake_runner(fn, calls), self.draft(), "draft-dflash", say=lambda s: None)
        res = t.tune()
        self.assertTrue(any("more spare VRAM" in r["label"] and not r.get("error") for r in res["results"]))

    def test_resume_uses_saved_trials(self):
        d = tempfile.mkdtemp()
        cache = os.path.join(d, "trials.jsonl")
        calls = []
        tune.Tuner(model(), FEATS, fake_runner(speed_curve(16), calls), self.draft(), "draft-dflash",
                   cache_path=cache, say=lambda s: None).tune()
        n = len(calls)
        calls2 = []
        res = tune.Tuner(model(), FEATS, fake_runner(speed_curve(16), calls2), self.draft(), "draft-dflash",
                         cache_path=cache, say=lambda s: None).tune()
        self.assertGreater(n, 3)
        self.assertEqual(calls2, [])
        self.assertTrue(all(r.get("cached") for r in res["results"]))

    def test_max_trials(self):
        calls = []
        tune.Tuner(model(), FEATS, fake_runner(speed_curve(16), calls), self.draft(), "draft-dflash",
                   say=lambda s: None, max_trials=3).tune()
        self.assertEqual(len(calls), 3)

    def test_no_fit_target_uses_defaults_only(self):
        calls = []
        f = server.Features(version="t", fit=False, fit_target=False)
        tune.Tuner(model(), f, fake_runner(lambda a, e: 4.0, calls), say=lambda s: None).tune()
        self.assertEqual(len(calls), 1)

    def test_load_mode_kept_only_when_faster(self):
        f = server.Features(version="t", fit=True, fit_target=True, load_mode=True, verbosity=True)
        for gain, expect in ((1.5, True), (1.0, False)):
            calls = []

            def fn(args, env, gain=gain):
                return 5.0 * (gain if "--load-mode" in args else 1.0)
            res = tune.Tuner(model(), f, fake_runner(fn, calls), say=lambda s: None).tune()
            self.assertEqual("--load-mode" in res["best"]["args"], expect)
            self.assertIn("-lv", calls[0].args)

    def test_manual_placement(self):
        a = tune.manual_placement(model(), 1024, vram_free_mib=11500)
        self.assertEqual(a[0], "-ngl")
        self.assertTrue(20 < int(a[1]) < 35)
        a = tune.manual_placement(model(moe=True), 1024, vram_free_mib=11500)
        self.assertEqual(a[:3], ["-ngl", "99", "--n-cpu-moe"])
        self.assertGreater(int(a[3]), 0)


class PMinAndKindTests(unittest.TestCase):
    def draft(self):
        return gguf.ModelInfo(path="h-mtp.gguf", arch="mtp", file_bytes=500 * 2 ** 20, n_layers=1)

    @staticmethod
    def curve(best_p):
        base = speed_curve(4, stream_helps=False)

        def fn(args, env):
            tps = base(args, env)
            if "--spec-draft-p-min" in args:
                tps *= 1.3 - abs(float(server_opt(args, "--spec-draft-p-min")) - best_p)
            return tps
        return fn

    def tuner(self, fn, calls, p_min=True):
        f = server.Features(**dict(FEATS.__dict__, p_min=p_min))
        return tune.Tuner(model(), f, fake_runner(fn, calls), self.draft(), "draft-mtp", say=lambda s: None)

    def test_p_min_tried_and_tuned(self):
        calls = []
        res = self.tuner(self.curve(0.85), calls).tune()
        self.assertIn("confidence cut-off 0.85", res["best"]["label"])
        ps = [server_opt(c.args, "--spec-draft-p-min") for c in calls if "--spec-draft-p-min" in c.args]
        self.assertEqual(ps[0], "0.7")                 # tried first at the starting guess count
        self.assertIn("0.95", ps)                      # 0.85 won, so it climbed one more and stopped
        self.assertNotIn("0.5", ps)

    def test_p_min_dropped_when_slower(self):
        calls = []
        slow = lambda a, e: speed_curve(4, False)(a, e) * (0.8 if "--spec-draft-p-min" in a else 1)
        res = self.tuner(slow, calls).tune()
        self.assertNotIn("--spec-draft-p-min", res["best"]["args"])
        # tried at the starting count, then once more one step up (a cut-off can pay only with more guesses)
        self.assertEqual(sum("--spec-draft-p-min" in c.args for c in calls), 2)

    def test_p_min_retried_with_more_guesses(self):
        # Real run (MTP on a 31B, 2026-11-17): the cut-off changed nothing at 4 guesses but won at 6 and up.
        plain = {2: 7.0, 3: 7.9, 4: 8.1, 6: 8.1, 8: 7.0, 12: 6.0, 16: 5.0, 24: 4.0, 32: 3.0}
        cut = {2: 7.0, 3: 7.9, 4: 8.1, 6: 8.8, 8: 9.1, 12: 8.9, 16: 8.5, 24: 8.0, 32: 7.5}

        def fn(a, e):
            if "--spec-type" not in a:
                return 3.6
            n = int(server_opt(a, "--spec-draft-n-max"))
            if "--spec-draft-p-min" not in a:
                return plain[n]
            return cut[n] * (1.0 if server_opt(a, "--spec-draft-p-min") == "0.7" else 0.97)
        calls = []
        res = self.tuner(fn, calls).tune()
        self.assertIn("mtp helper, 8 guesses, confidence cut-off 0.7", res["best"]["label"])
        tried = [(server_opt(c.args, "--spec-draft-n-max"), server_opt(c.args, "--spec-draft-p-min"))
                 for c in calls if "--spec-type" in c.args]
        self.assertEqual(tried.count(("4", "0.7")), 1)       # never measured twice
        self.assertIn(("6", "0.7"), tried)

    def test_no_p_min_on_old_builds(self):
        calls = []
        self.tuner(self.curve(0.85), calls, p_min=False).tune()
        self.assertFalse(any("--spec-draft-p-min" in c.args for c in calls))

    def test_per_kind_advice(self):
        def rec(label, helper, **k):
            return {"label": label, "key": label, "args": ["--spec-type", "x"] if helper else [],
                    "runs": [{"kind": n, "tps": v} for n, v in k.items()]}
        plain = rec("plain", False, math=3.6, code=3.6, explain=3.6, writing=3.6)
        helped = rec("helped", True, math=7.2, code=6.7, explain=2.7, writing=2.2)
        best, advice = tune.per_kind([plain, helped])
        self.assertEqual(best["math"]["label"], "helped")
        self.assertEqual(best["writing"]["label"], "plain")
        self.assertEqual(advice, "helper on for math and code, off for explanations and writing")
        fixed = rec("fixed", True, math=8.1, code=8.0, explain=5.2, writing=4.4)
        self.assertIsNone(tune.per_kind([plain, helped, fixed])[1])
        t = {"results": [plain, helped], "per_kind": best, "advice": advice}
        self.assertEqual([r["label"] for r in report.split_setups(t)], ["helped", "plain"])


class ReportTests(unittest.TestCase):
    def test_command_shells(self):
        rec = {"args": ["-c", "4096", "--no-webui", "--cache-ram", "0", "-lv", "4", "--fit-target", "256"],
               "env": {"GGML_OP_OFFLOAD_MIN_BATCH": "8"}}
        ps = report.command([r"C:\llama cpp\llama-server.exe"], r"D:\m.gguf", rec, "powershell")
        self.assertTrue(ps.startswith("$env:GGML_OP_OFFLOAD_MIN_BATCH='8'; & 'C:\\llama cpp\\llama-server.exe'"))
        self.assertNotIn("--cache-ram", ps)
        self.assertNotIn("--no-webui", ps)
        self.assertNotIn("-lv", ps)
        sh = report.command(["/opt/llama-server"], "/m/x y.gguf", rec, "bash")
        self.assertTrue(sh.startswith("GGML_OP_OFFLOAD_MIN_BATCH=8 /opt/llama-server -m '/m/x y.gguf'"))

    def test_same_text(self):
        a = {"runs": [{"kind": "a", "text": "x"}, {"kind": "b", "text": "y"}]}
        b = {"runs": [{"kind": "a", "text": "x"}, {"kind": "b", "text": "z"}]}
        self.assertEqual(tune.same_text(a, b), "1/2")


class EndToEnd(unittest.TestCase):
    """The real CLI and server runner against the fake llama-server, on a fake 24 GB GPU (so the result does not
    depend on the machine running the tests)."""

    def run_cli(self, args):
        from unittest import mock
        buf = io.StringIO()
        with mock.patch.object(hw, "gpus", lambda: [hw.GPU(0, "Fake GPU", 24576, 500)]),                 contextlib.redirect_stdout(buf):
            code = cli.main(args)
        return code, buf.getvalue()

    def test_full_tune(self):
        d = tempfile.mkdtemp()
        m, h = os.path.join(d, "model.gguf"), os.path.join(d, "helper-dflash.gguf")
        write_gguf(m, "gemma4", 4, extra_meta={"gemma4.embedding_length": 64})
        write_gguf(h, "dflash", 2, extra_meta={"dflash.embedding_length": 64})
        srv = f'"{sys.executable}" "{FAKE}"'
        out = os.path.join(d, "out")
        code, text = self.run_cli([m, "--draft", h, "--server", srv, "--quick", "--parallel", "4", "--out", out])
        self.assertEqual(code, 0, text)
        self.assertIn("BEST for one chat", text)
        self.assertIn("dflash helper, 16 guesses, streaming", text)
        self.assertIn("--load-mode none", text)        # the fake is 1.5x faster with it
        self.assertIn("layers on GPU", text)            # load lines were logged (needs -lv 4)
        html = Path(out, "report.html").read_text(encoding="utf-8")
        self.assertIn("Fastest setup for one chat", html)
        self.assertIn("Many requests at once", html)
        res = json.loads(Path(out, "results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["tune"]["best"]["same_as_baseline"], "2/2")
        # second run resumes from the saved trials without loading anything
        code, text2 = self.run_cli([m, "--draft", h, "--server", srv, "--quick", "--parallel", "4", "--out", out])
        self.assertEqual(code, 0)
        self.assertNotIn(" ...\n", text2)

    def full_prompts(self, no_pmin):
        from unittest import mock
        d = tempfile.mkdtemp()
        m, h = os.path.join(d, "model.gguf"), os.path.join(d, "helper-dflash.gguf")
        write_gguf(m, "gemma4", 4, extra_meta={"gemma4.embedding_length": 64})
        write_gguf(h, "dflash", 2, extra_meta={"dflash.embedding_length": 64})
        srv = f'"{sys.executable}" "{FAKE}"'
        out = os.path.join(d, "out")
        env = {"FAKE_NO_PMIN": "1"} if no_pmin else {}
        with mock.patch.dict(os.environ, env):
            code, text = self.run_cli([m, "--draft", h, "--server", srv, "--out", out])
        self.assertEqual(code, 0, text)
        return text, Path(out, "report.html").read_text(encoding="utf-8")

    def test_old_build_gets_per_kind_advice(self):
        text, html = self.full_prompts(no_pmin=True)
        self.assertNotIn("confidence cut-off", text)
        self.assertIn("Advice: helper on for math and code, off for explanations and writing.", text)
        self.assertIn("without the helper: ", text)
        self.assertIn("Best setup per kind of prompt", html)
        self.assertIn("Without the helper", html)

    def test_p_min_fixes_prose_end_to_end(self):
        text, html = self.full_prompts(no_pmin=False)
        best = text[text.index("BEST for one chat"):].splitlines()[0]
        self.assertIn("confidence cut-off 0.85", best)
        self.assertNotIn("Advice:", text)              # with the cut-off the helper wins every kind
        self.assertIn("Best per kind of prompt:", text)
        self.assertIn("--spec-draft-p-min 0.85", text)

    def test_dry_run_and_errors(self):
        d = tempfile.mkdtemp()
        m = os.path.join(d, "model.gguf")
        write_gguf(m, "llama", 2)
        srv = f'"{sys.executable}" "{FAKE}"'
        code, text = self.run_cli([m, "--server", srv, "--dry-run"])
        self.assertEqual(code, 0, text)
        self.assertIn("Baseline command", text)
        code, _ = self.run_cli([os.path.join(d, "missing.gguf"), "--server", srv])
        self.assertEqual(code, 2)

    def test_load_failure_is_reported_not_fatal(self):
        d = tempfile.mkdtemp()
        m = os.path.join(d, "model.gguf")
        write_gguf(m, "llama", 2)
        os.environ["FAKE_FAIL_IF"] = "--fit-target 512"
        try:
            code, text = self.run_cli([m, "--server", f'"{sys.executable}" "{FAKE}"', "--quick",
                                       "--out", os.path.join(d, "o")])
        finally:
            del os.environ["FAKE_FAIL_IF"]
        self.assertEqual(code, 0, text)
        self.assertIn("failed: llama-server exited with code 3", text)

    def test_missing_gpu_backend_stops_the_tune(self):
        d = tempfile.mkdtemp()
        m = os.path.join(d, "model.gguf")
        write_gguf(m, "llama", 2)
        os.environ["FAKE_NO_GPU"] = "1"
        try:
            code, text = self.run_cli([m, "--server", f'"{sys.executable}" "{FAKE}"', "--quick",
                                       "--out", os.path.join(d, "o")])
        finally:
            del os.environ["FAKE_NO_GPU"]
        self.assertEqual(code, 3, text)
        self.assertIn("without a GPU backend", text)
        self.assertNotIn("BEST for one chat", text)

    def test_parse_log_gpu_backend(self):
        cpu_only = "load_backend: loaded CPU backend from x.dll\nload_tensors: offloaded 31/31 layers to GPU\n"
        self.assertIs(server.parse_log(cpu_only)["gpu_backend"], False)
        self.assertIs(server.parse_log("load_backend: loaded CUDA backend from y.dll\n" + cpu_only)["gpu_backend"], True)
        self.assertIsNone(server.parse_log("srv load_model: loading model\n")["gpu_backend"])   # log too quiet to tell


if __name__ == "__main__":
    unittest.main()
