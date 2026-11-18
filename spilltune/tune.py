"""The search. Staged and greedy, so a tune takes ~8-14 model loads instead of hundreds:

1. placement  baseline (llama-server defaults), then a smaller --fit-target margin while it keeps getting faster
2. helper     (only with a draft/helper file) climb the guess-count ladder from a sensible start, testing
              weight streaming and a confidence cut-off once, then tune the cut-off and one tighter margin
3. parallel   (opt-in) total throughput with many requests at once

`runner(trial) -> record` does the measuring, so the search can be tested without a GPU.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import server

LADDER = [2, 3, 4, 6, 8, 12, 16, 24, 32]
MARGINS = [1024, 512, 256]          # MiB of VRAM llama.cpp's --fit leaves free
STREAM_ENV = "GGML_OP_OFFLOAD_MIN_BATCH"   # batches at least this big stream RAM weights to the GPU
STREAM_MIN = 8
MIN_GAIN = 1.01                     # a step must be 1% faster to count
P_START = 0.7                       # --spec-draft-p-min tried first: helper only offers guesses it is 70% sure of
P_LADDER = [0.5, 0.7, 0.85, 0.95]
KINDS_PROSE = ("explain", "writing")


@dataclass
class Trial:
    label: str
    stage: str
    args: list
    env: dict = field(default_factory=dict)
    concurrency: list = field(default_factory=list)   # parallel stage only


def start_n(spec_type, moe):
    if spec_type == "draft-dflash":
        return 8 if moe else 16     # on an MoE each extra guess can wake more experts in RAM
    if spec_type in ("draft-mtp", "draft-eagle3"):
        return 4
    return 8


def helper_reserve_mib(draft_info, spec_type):
    """VRAM the helper needs on top of its weights (KV + compute; DFlash's compute buffer is large)."""
    extra = 1100 if spec_type == "draft-dflash" else 300
    return int(draft_info.file_bytes / 2 ** 20) + extra


class Tuner:
    def __init__(self, model, feats, runner, draft=None, spec_type=None, cache_path=None, say=print,
                 key_extra=None, parallel=0, max_trials=40, ctx=server.DEFAULT_CTX):
        self.model, self.feats, self.runner = model, feats, runner
        self.ctx = ctx
        self.draft, self.spec_type = draft, spec_type
        self.say, self.parallel, self.max_trials = say, parallel, max_trials
        self.key_extra = key_extra or {}
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r
                except (ValueError, KeyError):
                    continue
        self.results = []
        self.load = []          # ["--load-mode", "none"] once that proves faster
        self.baseline = None
        self.n_run = 0

    # ---- plumbing -----------------------------------------------------------------------------------------
    def base_args(self):
        a = ["-c", str(self.ctx), "-np", "1"]
        if self.feats.no_webui:
            a.append("--no-webui")
        if self.feats.cache_ram:
            a += ["--cache-ram", "0"]
        if self.feats.verbosity:
            a += ["-lv", "4"]      # the load lines (layers, GPU buffers) that overflow detection reads
        return a + self.load

    def placement(self, margin):
        if margin is None:
            return []                                   # llama-server's own defaults
        if self.feats.fit_target:
            return ["--fit", "on", "--fit-target", str(int(margin))]
        if self.feats.fit:
            return ["--fit", "on"]
        return manual_placement(self.model, margin, ctx=self.ctx)

    def helper(self, n, p=None):
        a = ["-md", str(self.draft.path), "-ngld", "99", "--spec-type", self.spec_type,
             "--spec-draft-n-max", str(n)]
        return a + (["--spec-draft-p-min", str(p)] if p else [])

    def key(self, t):
        blob = json.dumps([self.model.name, self.model.file_bytes, self.draft.name if self.draft else None,
                           t.args, t.env, t.concurrency, self.key_extra], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def run(self, t):
        k = self.key(t)
        if k in self.cache:
            r = dict(self.cache[k], cached=True)
        else:
            if self.n_run >= self.max_trials:
                return None
            self.n_run += 1
            self.say(f"  trial {len(self.results) + 1:2d}  {t.label} ...")
            r = self.runner(t)
            r.update(key=k, label=t.label, stage=t.stage, args=t.args, env=t.env)
            if self.baseline and not r.get("error") and r.get("runs"):
                r["same_as_baseline"] = same_text(self.baseline, r)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
            self.cache[k] = r
        self.results.append(r)
        self.say("            " + summary(r) + ("  (saved result)" if r.get("cached") else ""))
        return r

    @staticmethod
    def tps(r):
        return 0.0 if (r is None or r.get("error")) else (r.get("tps") or 0.0)

    # ---- stages -------------------------------------------------------------------------------------------
    def tune(self):
        b = self.base_args()
        self.say("Stage 1/3: where the model's layers go (GPU vs RAM)")
        base = self.run(Trial("baseline: llama-server defaults", "placement", b))
        if base and not base.get("error"):
            self.baseline = base
        best_m, best = None, base
        if self.feats.fit_target:
            for m in MARGINS[1:]:
                r = self.run(Trial(f"--fit with {m} MiB spare VRAM", "placement", b + self.placement(m)))
                if self.tps(r) > self.tps(best) * MIN_GAIN:
                    best_m, best = m, r
                else:
                    break
        if self.baseline is None and best and not best.get("error"):
            self.baseline = best
        if self.feats.load_mode:
            # Default loading memory-maps the file, and Windows may drop mapped weights from RAM and re-read
            # them from disk. Copying the model into RAM instead was ~3x faster for an MoE on one test PC.
            r = self.run(Trial("copy the model into RAM (--load-mode none)", "placement",
                               b + ["--load-mode", "none"] + self.placement(best_m)))
            if self.tps(r) > self.tps(best) * MIN_GAIN:
                best = r
                self.load = ["--load-mode", "none"]
                b = self.base_args()
        margin = best_m if best_m is not None else (MARGINS[0] if self.feats.fit_target else None)

        if self.draft:
            self.tune_helper(b, margin)
        if self.parallel:
            self.tune_parallel(b, margin)
        return self.report()

    def tune_helper(self, b, margin):
        st = self.spec_type
        self.say(f"Stage 2/3: guess-ahead helper ({st.replace('draft-', '')}, {self.draft.name})")
        if self.feats.spec_types and st not in self.feats.spec_types:
            self.say(f"  this llama-server build does not support --spec-type {st}; skipping")
            return
        res = helper_reserve_mib(self.draft, st)
        m0 = (margin if margin is not None else MARGINS[0]) + res
        tried = {}

        def go(n, stream, m=m0, p=None):
            if (n, stream, m, p) in tried:
                return tried[(n, stream, m, p)]
            env = {STREAM_ENV: str(STREAM_MIN)} if stream else {}
            label = f"{st.replace('draft-', '')} helper, {n} guesses" + (", streaming" if stream else "") + \
                    (f", confidence cut-off {p}" if p else "") + \
                    ("" if m == m0 else f", {m0 - m} MiB less VRAM held back")
            r = self.run(Trial(label, "helper", b + self.placement(m) + self.helper(n, p), env))
            if r is not None and r.get("error") and m == m0 and not tried.get("retried"):
                tried["retried"] = True        # too big: leave another GiB free and retry once
                r = self.run(Trial(label + ", more spare VRAM", "helper",
                                   b + self.placement(m + 1024) + self.helper(n, p), env))
            tried[(n, stream, m, p)] = r
            return r

        n0 = start_n(st, self.model.is_moe)
        i0 = LADDER.index(n0)
        r = go(n0, False)
        stream = False
        if n0 + 1 >= STREAM_MIN:
            rs = go(n0, True)
            if self.tps(rs) > self.tps(r) * MIN_GAIN:
                r, stream = rs, True
        p = None
        if self.feats.p_min:
            # A helper that offers every guess wastes the big model's time on prose, where most guesses are
            # wrong (one test: 1380 guesses drafted for 94 accepted). A cut-off keeps only confident guesses.
            rp = go(n0, stream, p=P_START)
            if self.tps(rp) > self.tps(r) * MIN_GAIN:
                r, p = rp, P_START
        best_n, best = n0, r
        for step in (1, -1):                       # climb up; if the first step up loses, climb down
            i, moved = i0, False
            while 0 <= i + step < len(LADDER):
                n = LADDER[i + step]
                s = stream and n + 1 >= STREAM_MIN
                rn = go(n, s, p=p)
                if self.tps(rn) > self.tps(best) * MIN_GAIN:
                    best_n, best, i, moved = n, rn, i + step, True
                else:
                    break
            if moved:
                break
        if self.feats.p_min and p is None:
            # With few guesses a cut-off changes little (few guesses are wasted), so it can lose at the start
            # and still win once more guesses are allowed. One test (MTP on a 31B): 4 guesses ran 8.1 tokens/sec
            # with or without it, while 6 guesses with it ran 8.8. So try it again at the best count and one
            # step up, and if it wins, keep climbing with it.
            i = LADDER.index(best_n)
            for n in [best_n] + LADDER[i + 1:i + 2]:
                rq = go(n, stream and n + 1 >= STREAM_MIN, p=P_START)
                if self.tps(rq) > self.tps(best) * MIN_GAIN:
                    best_n, best, p = n, rq, P_START
            if p is not None:
                i = LADDER.index(best_n)
                while i + 1 < len(LADDER):
                    n = LADDER[i + 1]
                    rn = go(n, stream and n + 1 >= STREAM_MIN, p=p)
                    if self.tps(rn) > self.tps(best) * MIN_GAIN:
                        best_n, best, i = n, rn, i + 1
                    else:
                        break
        s = stream and best_n + 1 >= STREAM_MIN
        if p is not None:                          # tune the cut-off: one step each way, keep climbing if it wins
            j = P_LADDER.index(p)
            for step in (1, -1):
                k, moved = j, False
                while 0 <= k + step < len(P_LADDER):
                    rq = go(best_n, s, p=P_LADDER[k + step])
                    if self.tps(rq) > self.tps(best) * MIN_GAIN:
                        best, p, k, moved = rq, P_LADDER[k + step], k + step, True
                    else:
                        break
                if moved:
                    break
        if best and not best.get("error") and self.feats.fit_target:
            go(best_n, s, m0 - 384, p)             # a little tighter: more layers on the GPU

    def tune_parallel(self, b, margin):
        n = self.parallel
        self.say(f"Stage 3/3: {n} requests at once (total throughput)")
        levels = sorted({1, max(1, n // 4), max(1, n // 2), n})
        m = (margin if margin is not None else MARGINS[0]) + 512
        args = [a for a in b]
        i = args.index("-np"); del args[i:i + 2]
        i = args.index("-c"); del args[i:i + 2]
        args += ["-np", str(n), "-c", str(self.ctx * n), "-ctk", "q8_0", "-ctv", "q8_0"] + self.placement(m)
        self.run(Trial(f"{n} slots, 8-bit memory cache", "parallel", args, {}, levels))

    # ---- results ------------------------------------------------------------------------------------------
    def report(self):
        single = [r for r in self.results if r.get("stage") != "parallel" and not r.get("error")]
        best = max(single, key=lambda r: r.get("tps") or 0, default=None)
        par = next((r for r in self.results if r.get("stage") == "parallel" and not r.get("error")), None)
        kinds, advice = per_kind(single)
        return {"results": self.results, "baseline": self.baseline, "best": best, "parallel": par,
                "per_kind": kinds, "advice": advice}


KV_MIB_PER_4K = 512


def manual_placement(model, margin, vram_free_mib=None, ctx=server.DEFAULT_CTX):
    """Fallback for llama.cpp builds without --fit: estimate -ngl / --n-cpu-moe from the GGUF sizes."""
    from . import hw
    free = vram_free_mib if vram_free_mib is not None else sum(g.free_mib for g in hw.gpus())
    budget = (free - margin - 1200) * 2 ** 20           # ~1.2 GiB for cache + compute at 4K context
    if ctx > server.DEFAULT_CTX:
        # Extra KV cache for a longer context. Not estimated from GGUF metadata: hybrid (Qwen3.5/3.8) and
        # sliding-window (Gemma 4) models cache only some layers, so a metadata formula was 4-47x too high.
        # Measured full-context cache at 4K on four models was 80-320 MiB; reserve 512 MiB per extra 4K.
        budget -= int(KV_MIB_PER_4K * (ctx / server.DEFAULT_CTX - 1)) * 2 ** 20
    if model.is_moe:
        gpu = sum(model.layer_bytes) + model.other_bytes
        n = 0
        while n < model.n_layers and gpu > budget:
            gpu -= model.expert_bytes[n]
            n += 1
        return ["-ngl", "99", "--n-cpu-moe", str(n)]
    used, n = model.other_bytes // 2, 0
    for b in reversed(model.layer_bytes):
        if used + b > budget:
            break
        used += b
        n += 1
    return ["-ngl", str(n)]


def kind_tps(r):
    """{kind: tokens/sec} for one trial (empty when the trial kept no per-prompt speeds)."""
    return {x["kind"]: x["tps"] for x in r.get("runs", []) if x.get("tps")}


def uses_helper(r):
    return "--spec-type" in (r.get("args") or [])


def per_kind(trials):
    """Best trial per prompt kind, plus advice when a helper wins on some kinds and loses on others.
    Helpers are accepted far more often on predictable text (math steps, code) than on prose, so the fastest
    setup on average can be the wrong choice for a writing assistant or a coding agent."""
    best = {}
    for r in trials:
        for k, v in kind_tps(r).items():
            if k not in best or v > best[k]["tps"]:
                best[k] = {"tps": v, "label": r.get("label"), "key": r.get("key"), "helper": uses_helper(r)}
    on = [k for k, b in best.items() if b["helper"]]
    off = [k for k, b in best.items() if not b["helper"]]
    advice = None
    if on and off:
        names = {"math": "math", "code": "code", "explain": "explanations", "writing": "writing"}
        j = lambda ks: " and ".join(names.get(k, k) for k in ks)
        advice = f"helper on for {j(on)}, off for {j(off)}"
    return best, advice


def same_text(base, r):
    a = {x["kind"]: x["text"] for x in base.get("runs", [])}
    b = {x["kind"]: x["text"] for x in r.get("runs", [])}
    ks = [k for k in b if k in a]
    return f"{sum(a[k] == b[k] for k in ks)}/{len(ks)}"


def summary(r):
    if r is None:
        return "skipped (trial limit reached)"
    if r.get("error"):
        return f"failed: {r['error']}"
    if r.get("levels"):
        return "total " + ", ".join(f"{lv['concurrent']}x: {lv['agg_tps']:.1f}" for lv in r["levels"]) + " tokens/sec"
    s = f"{r.get('tps', 0):.1f} tokens/sec"
    if r.get("accept") is not None:
        s += f", {100 * r['accept']:.0f}% of guesses accepted"
    if r.get("gpu_layers"):
        s += f", {r['gpu_layers']} layers on GPU"
    if r.get("over_mib") is not None and r["over_mib"] > 0:
        s += " (tight on VRAM)"
    return s
