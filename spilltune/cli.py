"""spilltune: find the fastest llama-server settings for a model that does not fit in your GPU."""
import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from . import __version__, gguf, hw, prompts, report, server, tune


def find_server(arg):
    for c in (arg, os.environ.get("LLAMA_SERVER"), shutil.which("llama-server"), shutil.which("llama-server.exe")):
        if c:
            return c
    return None


def build_parser():
    p = argparse.ArgumentParser(prog="spilltune", description=__doc__,
                                epilog="Example: spilltune gemma-31b-Q4_K_M.gguf --draft gemma-31b-dflash.gguf")
    p.add_argument("model", help="the GGUF model file")
    p.add_argument("--draft", help="optional guess-ahead helper GGUF (MTP, DFlash, EAGLE3, or a small model of "
                                   "the same family)")
    p.add_argument("--spec-type", default="auto", help="helper type (default: detected from the helper file): "
                                                       "draft-mtp, draft-dflash, draft-eagle3, draft-simple")
    p.add_argument("--server", help="path to llama-server (default: $LLAMA_SERVER or llama-server on PATH)")
    p.add_argument("--parallel", type=int, default=0, metavar="N",
                   help="also measure total speed with N requests at once (e.g. 16)")
    p.add_argument("--prompts", type=int, default=4, help="how many of the 4 built-in test prompts to use")
    p.add_argument("--max-tokens", type=int, default=192, help="tokens generated per prompt (default 192)")
    p.add_argument("--quick", action="store_true", help="2 prompts x 128 tokens: faster, noisier")
    p.add_argument("--ctx", type=int, default=4096, metavar="TOKENS",
                   help="context length to tune for, per slot (default 4096). Use what you will run with: "
                        "a longer context needs more VRAM, which changes the best placement")
    p.add_argument("--max-trials", type=int, default=40, help="stop after this many model loads")
    p.add_argument("--out", help="output folder (default: ./spilltune-out/<model name>)")
    p.add_argument("--fresh", action="store_true", help="ignore saved trial results and measure everything again")
    p.add_argument("--dry-run", action="store_true", help="show the hardware, model facts and plan, run nothing")
    p.add_argument("--version", action="version", version=f"spilltune {__version__}")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.quick:
        a.prompts, a.max_tokens = 2, 128
    if a.ctx < 512:
        build_parser().error("--ctx must be at least 512 tokens")
    say = lambda s="": print(s, flush=True)

    srv = find_server(a.server)
    if not srv:
        say("error: llama-server not found. Pass --server PATH or set LLAMA_SERVER.")
        return 2
    server_cmd = ([x.strip('"') for x in shlex.split(srv, posix=False)]
                  if (" " in srv and not Path(srv).exists()) else [srv])
    try:
        model = gguf.read(a.model)
        draft = gguf.read(a.draft) if a.draft else None
    except (OSError, gguf.GGUFError) as e:
        say(f"error: {e}")
        return 2
    spec_type = (gguf.draft_type(draft) if a.spec_type == "auto" else a.spec_type) if draft else None
    try:
        feats = server.features(server_cmd)
    except FileNotFoundError as e:
        say(f"error: {e}")
        return 2
    h = hw.probe()

    G = 2 ** 30
    need = model.file_bytes + (draft.file_bytes if draft else 0)
    say(f"spilltune {__version__}")
    say(f"Hardware: {h.describe()}")
    say(f"Model:    {model.name}: {model.file_bytes / G:.1f} GB, {model.n_layers} layers"
        + (f", MoE with {model.expert_count} experts ({sum(model.expert_bytes) / G:.1f} GB of expert weights)"
           if model.is_moe else ", dense"))
    say(f"Context:  {a.ctx} tokens per slot"
        + (" (llama-server sizes the cache; --fit plans around it)" if feats.fit_target else ""))
    if draft:
        say(f"Helper:   {draft.name}: {draft.file_bytes / G:.2f} GB, --spec-type {spec_type}")
    say(f"Server:   {' '.join(server_cmd)} ({feats.version})")
    vram = h.vram_total_mib * 2 ** 20
    if not h.gpus:
        say("warning: no NVIDIA GPU detected (nvidia-smi not found). Overflow detection is off; speeds still count.")
    elif need < vram - 1.5 * G:
        say("note: the model fits in VRAM, so there is no spill to tune; the helper and parallel stages still apply.")
    else:
        say(f"          {max(0, need - vram + G) / G:.1f} GB of it has to live in system RAM.")
    if h.ram_total_mib and need > h.ram_total_mib * 2 ** 20 + vram - 2 * G:
        say("error: the model is larger than your RAM + VRAM; it would page from disk. Use a smaller quant.")
        return 1
    busy = [g for g in h.gpus if g.used_mib > 1536]
    if busy:
        say(f"warning: {busy[0].used_mib / 1024:.1f} GB of VRAM is already in use by other programs. Close them "
            "for best results (the tune plans around whatever is free).")
    if draft and gguf.helper_mismatch(model, draft):
        say(f"error: {gguf.helper_mismatch(model, draft)}. Use the helper built for {model.name}.")
        return 2
    if draft and feats.spec_types and spec_type not in feats.spec_types:
        say(f"warning: this llama-server does not list --spec-type {spec_type}; the helper stage will be skipped.")

    if a.dry_run:
        t = tune.Tuner(model, feats, None, draft, spec_type, ctx=a.ctx)
        say("\nPlan (adaptive, so the exact trials depend on what gets faster):")
        say("  1. placement: llama-server defaults, then --fit with 512 / 256 MiB spare VRAM while it keeps helping"
            if feats.fit_target else "  1. placement: llama-server defaults only (this build has no --fit-target)")
        if draft:
            n0 = tune.start_n(spec_type, model.is_moe)
            say(f"  2. helper: {spec_type} starting at {n0} guesses, climbing {tune.LADDER}, streaming on/off, "
                + (f"confidence cut-off (--spec-draft-p-min {tune.P_START}, then {tune.P_LADDER}), " if feats.p_min else "")
                + f"reserving {tune.helper_reserve_mib(draft, spec_type)} MiB VRAM for it")
        if a.parallel:
            say(f"  3. parallel: {a.parallel} slots, 8-bit KV cache, levels {sorted({1, max(1, a.parallel // 4), max(1, a.parallel // 2), a.parallel})}")
        say(f"  Each trial: {a.prompts} prompts x {a.max_tokens} tokens. Usually 3-14 model loads.")
        say("\nBaseline command:\n  " + report.command(server_cmd, model.path, {"args": t.base_args(), "env": {}}))
        return 0

    out = Path(a.out or Path("spilltune-out") / Path(model.name).stem)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "trials.jsonl"
    if a.fresh and cache.exists():
        cache.unlink()
    ps = prompts.pick(a.prompts)
    counter = {"n": 0}

    def runner(t):
        counter["n"] += 1
        log = out / "logs" / f"trial-{counter['n']:02d}.log"
        return server.run_trial(server_cmd, model.path, t.args, t.env, ps, a.max_tokens, log, hw.gpus(),
                                concurrency=t.concurrency or None)

    say(f"\nResults are saved as each trial finishes (re-run to resume): {cache}\n")
    tuner = tune.Tuner(model, feats, runner, draft, spec_type, cache_path=cache, say=say, parallel=a.parallel,
                       max_trials=a.max_trials, ctx=a.ctx,
                       key_extra={"prompts": prompts.VERSION, "n": a.prompts, "max_tokens": a.max_tokens,
                                  "server": feats.version})
    try:
        res = tuner.tune()
    except KeyboardInterrupt:
        say("\nstopped; finished trials are saved, re-run the same command to resume.")
        return 130
    except server.NoGPUBackend as e:
        say(""); say(f"STOPPED: {e}")
        return 3
    info = {"hw": h.describe(), "model": model.name, "model_path": model.path,
            "draft": draft.name if draft else None, "server": feats.version, "server_cmd": server_cmd,
            "tune": res, "when": report.now()}
    (out / "report.html").write_text(report.html_report(info), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")

    best, base = res["best"], res["baseline"]
    say("")
    if not best:
        say("No setup worked; see the errors above and the logs in " + str(out / "logs"))
        return 1
    bt = (base or {}).get("tps") or 0
    say(f"BEST for one chat: {best['tps']:.1f} tokens/sec" + (f" ({best['tps'] / bt:.2f}x the defaults' {bt:.1f})" if bt else "")
        + f"  [{best['label']}]")
    say("  " + report.command(server_cmd, model.path, best))
    pk = res.get("per_kind") or {}
    if len(pk) > 1:
        say("Best per kind of prompt:")
        for k, b in pk.items():
            say(f"  {report.KIND_NAMES.get(k, k):<12} {b['tps']:5.1f} tokens/sec  [{b['label']}]")
    sp = report.split_setups(res)
    if sp:
        say(f"Advice: {res['advice']}.")
        say("  with the helper:    " + report.command(server_cmd, model.path, sp[0]))
        say("  without the helper: " + report.command(server_cmd, model.path, sp[1]))
    if res["parallel"]:
        lv = res["parallel"]["levels"][-1]
        say(f"Many requests: {lv['agg_tps']:.1f} tokens/sec in total with {lv['concurrent']} at once")
        say("  " + report.command(server_cmd, model.path, res["parallel"]))
    elif any(r.get("stage") == "parallel" for r in res["results"]):
        say("Many requests: the test failed (see above)")
    say(f"Report: {out / 'report.html'}")
    return 0


def entry():
    sys.exit(main())
