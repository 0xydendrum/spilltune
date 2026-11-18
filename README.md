# spilltune

**Find the fastest llama-server settings for a model that doesn't fit in your GPU.**

When a model is bigger than your VRAM, part of it lives in system RAM, and generation slows to whatever speed
RAM can feed the CPU. llama.cpp has several options that help here: where the layers go, guess-ahead helpers
(speculative decoding with MTP, DFlash or EAGLE3 helper files), streaming RAM weights to the GPU, MoE expert
offload, and serving many requests at once. Which combination wins depends on your GPU, your RAM and the model,
and finding it by hand takes an afternoon.

spilltune does that afternoon for you. Point it at a GGUF file, and it:

1. reads your VRAM and RAM and the model's layout,
2. starts `llama-server` with one setup at a time and times real generation on four built-in prompts
   (math, code, an explanation and a story, so helpers aren't flattered by easy text),
3. climbs toward the fastest setup instead of trying every combination (4 to 15 model loads, 2 to 47 minutes
   in the runs below),
4. checks the answers match the default setup's,
5. prints the fastest command, the best setup for each kind of prompt (with separate commands when a helper
   wins on code and math but loses on prose), and writes an HTML report.

```
$ spilltune gemma-4-31B-it-Q4_K_M.gguf --draft gemma4-31b-it-dflash-Q4_K_M.gguf
Hardware: NVIDIA GeForce RTX 3060 12 GB (0.4 GB in use); RAM 32 GB (25 GB free)
Model:    gemma-4-31B-it-Q4_K_M.gguf: 17.1 GB, 60 layers, dense
Helper:   gemma4-31b-it-dflash-Q4_K_M.gguf: 0.85 GB, --spec-type draft-dflash
          6.9 GB of it has to live in system RAM.

Stage 1/3: where the model's layers go (GPU vs RAM)
  trial  1  baseline: llama-server defaults ...
            3.4 tokens/sec, 31/61 layers on GPU
  ...
  trial 15  dflash helper, 6 guesses, confidence cut-off 0.7, 384 MiB less VRAM held back ...
            6.5 tokens/sec, 66% of guesses accepted, 21/61 layers on GPU

BEST for one chat: 6.5 tokens/sec (1.92x the defaults' 3.4)  [dflash helper, 6 guesses, confidence cut-off 0.7, 384 MiB less VRAM held back]
  & llama-server.exe -m gemma-4-31B-it-Q4_K_M.gguf -c 4096 -np 1 --fit on --fit-target 2098 -md gemma4-31b-it-dflash-Q4_K_M.gguf -ngld 99 --spec-type draft-dflash --spec-draft-n-max 6 --spec-draft-p-min 0.7
Report: spilltune-out\gemma-4-31B-it-Q4_K_M\report.html
```

## Install

Needs Python 3.9+ and a llama.cpp build with `llama-server`. No other packages.

```
pip install spilltune
```

or `pip install git+https://github.com/0xydendrum/spilltune` for the latest commit, or clone the repo and run `python -m spilltune ...`.

## Use

```
spilltune MODEL.gguf [--draft HELPER.gguf] [--parallel 16] [--server PATH] [--quick] [--dry-run]
```

| Option | What it does |
|---|---|
| `--draft FILE` | A guess-ahead helper made for this model (MTP, DFlash, EAGLE3, or a small model of the same family). The type is detected from the file; override with `--spec-type`. |
| `--parallel N` | Also measure total speed with N requests at once (for agents, batch jobs or several users). |
| `--server PATH` | Path to `llama-server` (default: `$LLAMA_SERVER`, then `llama-server` on your PATH). |
| `--quick` | 2 prompts x 128 tokens per trial instead of 4 x 192. Faster, noisier. |
| `--dry-run` | Show the hardware, model facts and plan without loading anything. |
| `--out DIR` | Where the report, results and server logs go (default `./spilltune-out/<model>`). |
| `--fresh` | Re-measure everything instead of reusing saved trials. |

Each finished trial is saved immediately. If a tune is interrupted, run the same command again and it picks up
where it stopped.

Close other programs that use the GPU (games, browsers with hardware acceleration, other AI tools) before
tuning. spilltune plans around whatever VRAM is free, so a busy GPU gives slower, different answers.

## What it tries

**Stage 1: placement.** First the llama-server defaults (`--fit on`, which keeps 1024 MiB of VRAM spare), then
512 and 256 MiB spare while each step keeps getting faster. More layers on the GPU means less to read from RAM.

**Stage 2: guess-ahead helper** (only with `--draft`). A small helper guesses several tokens ahead, and the big
model checks all of them in one pass. When the big model is waiting on RAM, checking 16 guesses costs about the
same as producing 1 token, so every accepted guess is nearly free. spilltune:
- holds back VRAM for the helper (its weights plus its working memory) so the main model doesn't overflow,
- starts at a sensible number of guesses for the helper type (16 for DFlash on dense models, 8 on MoE models,
  4 for MTP and EAGLE3) and climbs up or down while it gets faster,
- tests weight streaming (`GGML_OP_OFFLOAD_MIN_BATCH`) once, which matters when many guesses are checked at once,
- tests a confidence cut-off (`--spec-draft-p-min 0.7`), and if it helps keeps it while climbing, then tries
  0.5 / 0.85 / 0.95. With few guesses the cut-off changes little, so if it doesn't help at first, it gets one more
  try with more guesses. Without a cut-off a helper offers every guess, and on prose most are wrong: on one test
  DFlash drafted 1380 tokens to get 94 accepted, which made writing slower than no helper at all. With the
  cut-off it skips guesses it isn't sure of, and prose got about twice as fast (needs a llama.cpp build with
  `--spec-draft-p-min`; older builds skip this step),
- finally tries holding back a little less VRAM.

**Stage 3: many requests at once** (only with `--parallel`). One trial with N slots and an 8-bit memory cache,
measured at 1, N/4, N/2 and N requests.

## Overflow detection (Windows)

On Windows, the NVIDIA driver does not fail when VRAM runs out: by default it quietly moves the excess into
system RAM ("CUDA Sysmem Fallback"), which can make a setup several times slower while looking like it loaded
fine. The driver's own counters can't tell this apart from normal operation. spilltune adds up the GPU buffers
llama.cpp reports in its log and compares them with the free VRAM; a setup that clearly overflows is stopped and
marked as failed instead of being timed.

## Per-kind results

A helper is right far more often on math steps and code than on open writing, so spilltune keeps the speed of
each prompt kind for every trial and prints the winner for each:

```
Best per kind of prompt:
  Math          13.0 tokens/sec  [mtp helper, 6 guesses, confidence cut-off 0.5]
  Code          12.7 tokens/sec  [mtp helper, 6 guesses, confidence cut-off 0.85]
  Explanation    7.7 tokens/sec  [mtp helper, 6 guesses, confidence cut-off 0.7, 384 MiB less VRAM held back]
  Writing        6.5 tokens/sec  [mtp helper, 6 guesses, confidence cut-off 0.7]
```

When a helper wins on some kinds and loses on others (common on builds without `--spec-draft-p-min`), it says so
and prints both commands:

```
Advice: helper on for math and code, off for explanations and writing.
  with the helper:    & llama-server.exe -m ... --spec-type draft-dflash --spec-draft-n-max 16
  without the helper: & llama-server.exe -m ... --fit on --fit-target 512
```

## Results from one PC

RTX 3060 12 GB, i5-14600KF, 32 GB DDR4-3200, Windows 11, llama.cpp build 11798, spilltune's four prompts x 192
tokens. Speeds are tokens/sec: total tokens over total time across the four prompts.

**What spilltune picked, run unattended** (one run per row, 2026-11-17):

| Model (file size) | Helper | Defaults | spilltune's pick | Speed-up | Loads | Time |
|---|---|---|---|---|---|---|
| Gemma 4 31B Q4_K_M (17.1 GB) | MTP | 3.4 | 8.9 (MTP, 6 guesses, cut-off 0.7) | 2.61x | 13 | 32 min |
| Gemma 4 31B Q4_K_M (17.1 GB) | DFlash | 3.4 | 6.5 (DFlash, 6 guesses, cut-off 0.7) | 1.92x | 15 | 47 min |
| Gemma 4 31B Q4_K_M (17.1 GB) | EAGLE3 | 3.4 | 5.9 (EAGLE3, 3 guesses) | 1.73x | 10 | 31 min |
| Gemma 4 26B-A4B (MoE) Q4 | MTP | 37.5 | 56.0 (MTP, 3 guesses, cut-off 0.7) | 1.50x | 12 | 7 min |
| Gemma 4 26B-A4B (MoE) Q4 | DFlash | 37.2 | 42.3 (DFlash, 6 guesses, cut-off 0.7) | 1.14x | 13 | 9 min |
| Qwen3.5-35B-A3B (MoE) IQ4_XS | none | 47.2 | 50.2 (`--load-mode none`) | 1.06x | 4 | 2 min |
| Qwen3.8-27B Q4_K_M | none | 4.4 | 5.0 (`--fit-target 256`) | 1.14x | 4 | 12 min |

In these runs, the two models without a helper gained little over the defaults (1.06x and 1.14x); the big wins
came from picking the helper's guess count and cut-off.

**Single setups, measured by hand with the same prompts:**

| Setup | Math | Code | Explain | Writing |
|---|---|---|---|---|
| Gemma 4 31B Q4_K_M, llama-server defaults | 3.4 | 3.4 | 3.4 | 3.4 |
| + DFlash helper, 16 guesses | 7.3 | 6.8 | 2.7 | 2.2 |
| + DFlash helper, 16 guesses, cut-off 0.7 | 8.1 | 8.0 | 5.2 | 4.4 |
| + MTP helper, 6 guesses | 13.4 | 11.4 | 6.5 | 5.6 |
| + MTP helper, 6 guesses, cut-off 0.7 | 12.0 | 11.8 | 7.4 | 6.6 |
| Gemma 4 26B-A4B (MoE) Q4, `--load-mode none --fit` | 40.0 | 40.1 | 40.1 | 40.0 |
| + DFlash helper, 8 guesses | 37.4 | 45.7 | 23.4 | 20.5 |
| + DFlash helper, 8 guesses, cut-off 0.5 | 38.7 | 50.9 | 28.3 | 25.3 |
| + DFlash helper, 2 guesses | 50.8 | 51.8 | 37.7 | 31.8 |

On the 31B, the cut-off roughly doubles prose speed with a helper and lifts math and code too. On the MoE, many
guesses lose even with the cut-off: each extra guess can need different experts, which have to be read from RAM,
so few guesses (or none, for writing) win there. That is why the helper's starting point depends on the model type.
Serving 16 requests at once, the MoE reached 86.5 tokens/sec in total. (MTP behaves differently on the MoE: 3
guesses with the cut-off gave 56.0 tokens/sec, see the table above.)

The answers are checked against the defaults' answers on the test prompts. Separately, on 450 MMLU-Pro and GSM8K
questions, the helper setups scored within about 1 point of the same model fully in VRAM. With the cut-off
(DFlash, 16 guesses, 0.7) it scored 72.0% vs 72.2% fully in VRAM (-0.2 +/-1.0 points, 98.4% same answers).

## Limits

- Tested on Windows 11 with an NVIDIA GPU. Linux should work (overflow there makes the load fail, which
  spilltune also handles); AMD, Intel and Apple GPUs are untested, and overflow detection reads CUDA, ROCm,
  Vulkan, SYCL and Metal log lines but has only been checked against CUDA.
- One GPU. Multi-GPU split tuning is not done yet.
- Tunes at 4096 tokens of context per slot unless you pass `--ctx`. A longer context needs more VRAM, so tune
  with the context you will actually run (`--ctx 32768`). With `--fit` llama.cpp sizes the cache itself; on older
  builds without it spilltune reserves 512 MiB per extra 4K tokens, a bound from four measured models.
- Speeds are for the built-in prompts. Your own text may accept more or fewer guesses.
- A helper only speeds things up when it was made for that exact model; spilltune refuses a helper whose width
  doesn't match.

## If it stops with "started without a GPU backend"

spilltune checks that llama-server actually loaded a GPU backend. If your GPU is present but the server log
shows only the CPU backend, every speed would be wrong, so it stops. On Windows the usual cause is missing CUDA
runtime DLLs (cudart64_*.dll, cublas64_*.dll, cublasLt64_*.dll): put the files from llama.cpp's
cudart-llama-bin-win-cuda-*.zip next to llama-server.exe, or add the folder that holds them to PATH.

## Related

[spillsafe](https://github.com/0xydendrum/spillsafe) runs heavy local AI jobs one at a time, and only when RAM
and VRAM can hold them. Running spilltune inside spillsafe keeps it from colliding with other jobs.

## License

[PolyForm Noncommercial 1.0.0](https://github.com/0xydendrum/spilltune/blob/main/LICENSE). Free for personal, research and non-commercial use.
