"""Self-contained HTML report + the ready-to-run command."""
import html
import shlex
import subprocess
import sys
import time

E = html.escape


def command(server_cmd, model, rec, shell=None):
    """The llama-server command line for a measured trial, for the user's shell."""
    shell = shell or ("powershell" if sys.platform == "win32" else "bash")
    args = [a for a in server_cmd] + ["-m", str(model)] + [a for a in rec["args"]
                                                         if a not in ("--no-webui",)]
    args = _drop_pair(_drop_pair(args, "--cache-ram"), "-lv")
    env = rec.get("env") or {}
    if shell == "powershell":
        q = [f"& {_ps(args[0])}"] + [_ps(a) for a in args[1:]]
        pre = "".join(f"$env:{k}='{v}'; " for k, v in env.items())
        return pre + " ".join(q)
    if shell == "cmd":
        pre = "".join(f"set {k}={v}&& " for k, v in env.items())
        return pre + subprocess.list2cmdline(args)
    return "".join(f"{k}={v} " for k, v in env.items()) + " ".join(shlex.quote(a) for a in args)


def _drop_pair(args, flag):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == flag:
            skip = True
            continue
        out.append(a)
    return out


def _ps(a):
    return a if all(c.isalnum() or c in "-_.:/\\=," for c in a) else "'" + a.replace("'", "''") + "'"


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--text:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--good:#1f7a4d;--bad:#b3261e;--accent:#2d5bd7;--code:#f0f0ec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--card:#1f1f1d;--text:#ececea;--muted:#a3a39d;--line:#34342f;--good:#5fc48f;--bad:#f2867d;--accent:#8aa8ff;--code:#2a2a27}}
:root[data-theme="dark"]{--bg:#161615;--card:#1f1f1d;--text:#ececea;--muted:#a3a39d;--line:#34342f;--good:#5fc48f;--bad:#f2867d;--accent:#8aa8ff;--code:#2a2a27}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:0 0 10px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}
.big{font-size:40px;font-weight:700;line-height:1.1}.good{color:var(--good)}.bad{color:var(--bad)}
.stats{display:flex;gap:28px;flex-wrap:wrap}.stats div{min-width:140px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.best td{background:color-mix(in srgb,var(--good) 12%,transparent)}
.wrap{overflow-x:auto}pre{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto;white-space:pre-wrap;word-break:break-all;font-size:13px;margin:6px 0}
"""


KIND_NAMES = {"math": "Math", "code": "Code", "explain": "Explanation", "writing": "Writing"}


def split_setups(t):
    """When a helper wins on some prompt kinds and loses on others: (best helper trial, best no-helper trial),
    each the one that wins the most kinds (ties: faster on average). Otherwise None."""
    if not t.get("advice"):
        return None
    by_key = {r.get("key"): r for r in t["results"]}
    pick = {}
    for helper in (True, False):
        wins = {}
        for b in t["per_kind"].values():
            if b["helper"] == helper and b["key"] in by_key:
                wins[b["key"]] = wins.get(b["key"], 0) + 1
        if not wins:
            return None
        pick[helper] = by_key[max(wins, key=lambda k: (wins[k], by_key[k].get("tps") or 0))]
    return pick[True], pick[False]


def html_report(info):
    """info: dict from cli with keys hw, model, draft, server, server_cmd, tune (Tuner.report()), when."""
    t = info["tune"]
    base, best, par = t["baseline"], t["best"], t["parallel"]
    bt = (base or {}).get("tps") or 0
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>spilltune report</title><style>{CSS}</style></head><body><main>",
           f"<h1>spilltune: {E(info['model'])}</h1>",
           f"<p class='muted'>{E(info['hw'])}<br>llama-server {E(info['server'])} · {E(info['when'])}"
           + (f"<br>helper: {E(info['draft'])}" if info.get("draft") else "") + "</p>"]
    if best:
        x = (best["tps"] / bt) if bt else None
        out.append("<div class='card'><h2>Fastest setup for one chat</h2><div class='stats'>"
                   f"<div><div class='big good'>{best['tps']:.1f}</div><div class='muted'>tokens/sec</div></div>"
                   f"<div><div class='big'>{bt:.1f}</div><div class='muted'>llama-server defaults</div></div>"
                   + (f"<div><div class='big'>{x:.2f}×</div><div class='muted'>faster</div></div>" if x else "")
                   + f"</div><p>{E(best['label'])}</p>")
        for sh, name in (("powershell", "PowerShell"), ("bash", "bash / Linux")):
            out.append(f"<div class='muted'>{name}</div><pre>{E(command(info['server_cmd'], info['model_path'], best, sh))}</pre>")
        if best.get("same_as_baseline"):
            out.append(f"<p class='muted'>Answers identical to the defaults on {best['same_as_baseline']} test prompts. "
                       "Helpers never change what the model would have said in principle; a different GPU/CPU split can "
                       "change rounding, so a word or two may differ.</p>")
        out.append("</div>")
    else:
        out.append("<div class='card'><h2>No setup worked</h2><p>Every trial failed. See the errors below and the "
                   "server logs next to this report.</p></div>")
    pk = t.get("per_kind") or {}
    if len(pk) > 1:
        out.append("<div class='card'><h2>Best setup per kind of prompt</h2><p>Guess-ahead helpers are right far "
                   "more often on predictable text (math steps, code) than on open writing, so the best setup can "
                   "depend on what you use the model for.</p><table><tr><th>Prompt</th><th class='num'>tokens/sec</th>"
                   "<th>Setup</th></tr>")
        for k, b in pk.items():
            out.append(f"<tr><td>{E(KIND_NAMES.get(k, k))}</td><td class='num'>{b['tps']:.1f}</td>"
                       f"<td>{E(b['label'])}</td></tr>")
        out.append("</table>")
        sp = split_setups(t)
        if sp:
            out.append(f"<p><b>Advice: {E(t['advice'])}.</b> Run one server per use, or pick the one that matches "
                       "most of your prompts.</p>")
            for rec, name in ((sp[0], "With the helper"), (sp[1], "Without the helper")):
                out.append(f"<div class='muted'>{name} ({E(rec['label'])})</div>"
                           f"<pre>{E(command(info['server_cmd'], info['model_path'], rec))}</pre>")
        out.append("</div>")
    perr = next((r for r in t["results"] if r.get("stage") == "parallel" and r.get("error")), None)
    if perr and not par:
        out.append(f"<div class='card'><h2>Many requests at once</h2><p class='bad'>Failed: {E(perr['error'])}</p></div>")
    if par:
        out.append("<div class='card'><h2>Many requests at once</h2><p>Total tokens/sec when several chats or "
                   "agents use the same server. Each request is slower than a single chat, but the total grows, "
                   "because the model weights read from RAM are shared across the batch.</p><table><tr><th>Requests at once</th>"
                   "<th class='num'>Total tokens/sec</th></tr>")
        for lv in par["levels"]:
            out.append(f"<tr><td>{lv['concurrent']}</td><td class='num'>{lv['agg_tps']:.1f}</td></tr>")
        out.append("</table>")
        for sh, name in (("powershell", "PowerShell"), ("bash", "bash / Linux")):
            out.append(f"<div class='muted'>{name}</div><pre>{E(command(info['server_cmd'], info['model_path'], par, sh))}</pre>")
        out.append("</div>")
    out.append("<div class='card'><h2>Every trial</h2><div class='wrap'><table><tr><th>Setup</th><th class='num'>tokens/sec</th>"
               "<th class='num'>vs defaults</th><th class='num'>Math</th><th class='num'>Code</th>"
               "<th class='num'>Explain</th><th class='num'>Writing</th><th class='num'>guesses accepted</th>"
               "<th class='num'>GPU layers</th><th class='num'>same answers</th></tr>")
    for r in t["results"]:
        if r.get("stage") == "parallel":
            continue
        cls = " class='best'" if r is best else ""
        if r.get("error"):
            out.append(f"<tr><td>{E(r['label'])}</td><td colspan='9' class='bad'>{E(r['error'])}</td></tr>")
            continue
        v = f"{r['tps'] / bt:.2f}×" if bt else "–"
        acc = f"{100 * r['accept']:.0f}%" if r.get("accept") is not None else "–"
        kt = {x["kind"]: x.get("tps") for x in r.get("runs", [])}
        kc = "".join(f"<td class='num'>{kt[k]:.1f}</td>" if kt.get(k) else "<td class='num'>–</td>"
                     for k in ("math", "code", "explain", "writing"))
        out.append(f"<tr{cls}><td>{E(r['label'])}</td><td class='num'>{r['tps']:.1f}</td><td class='num'>{v}</td>"
                   f"{kc}<td class='num'>{acc}</td><td class='num'>{E(r.get('gpu_layers') or '–')}</td>"
                   f"<td class='num'>{E(r.get('same_as_baseline') or '–')}</td></tr>")
    out.append("</table></div><p class='muted'>Speed = generated tokens ÷ generation time over the built-in test prompts "
               "(math, code, explanation, story), so helpers are not flattered by math-only text. Prompt reading time "
               "and model loading are not counted.</p></div>")
    out.append("</main></body></html>")
    return "\n".join(out)


def now():
    return time.strftime("%Y-%m-%d %I:%M %p")
