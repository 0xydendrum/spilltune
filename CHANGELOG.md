# Changelog

## 0.3.0 (2026-11-18)

- **`--ctx TOKENS`** tunes for the context length you will run (default 4096). It goes to every trial (`-c`), and the
  parallel stage uses `--ctx` x slots. Saved trials at a different context are not reused. Checked on an RTX 3060
  with Qwen3.5-35B-A3B at 32K: llama-server got `n_ctx = 32768` and a 640 MiB KV cache (8x the 80 MiB at 4K).
- Published on PyPI: `pip install spilltune`.

## 0.2.0 (2026-11-17)

- **Confidence cut-off for helpers.** On llama.cpp builds with `--spec-draft-p-min`, the helper stage tries a
  cut-off of 0.7 at the starting guess count, keeps it while climbing if it helps, then tunes it (0.5 / 0.85 / 0.95).
  On a Gemma 4 31B on a 12 GB GPU this took DFlash from 2.2-2.7 to 4.4-5.2 tokens/sec on prose, and math and code
  got faster too. If the cut-off doesn't help at the starting guess count, it is tried again at the best count and
  one step up (on MTP for a 31B it made no difference at 4 guesses but won at 6).
- **Best setup per kind of prompt.** spilltune now reports the fastest setup for math, code, explanations and
  writing separately. When a helper wins on some and loses on others, it prints advice and a command for each.
- The HTML report shows per-kind speeds for every trial and a "Best setup per kind of prompt" card.
- The plan printed by `--dry-run` mentions the cut-off step.
- Tests: 29 (was 22). The fake llama-server now models prompt kinds and the cut-off.

## 0.1.0 (2026-11-16)

- First version: placement (`--fit-target`), helper guess count and weight streaming, `--load-mode none`,
  parallel slots, overflow detection on Windows, resumable trials, HTML report.
