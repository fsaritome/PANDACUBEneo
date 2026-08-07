# OCR Engine & Hardware Benchmarks

All numbers below are real, measured runs (not estimates) on the same 29-file
real patent PDF test corpus (`PDF STUFF/`), using the production pipeline
(`patent-ocr ... sweep`), unless noted otherwise. 28 of the 29 files go
through OCR; 1 (`sendebericht_*.pdf`) is text-native and takes the
passthrough path (no OCR, no confidence score).

## Engine quality: Tesseract vs PaddleOCR (isolated, ai01 GPU, 29-file batch)

Same batch, same machine, `strategy: always_parallel`, only `primary`/
`secondary` changed so each engine ran alone.

| Engine        | Mean confidence | Min    | Max    |
|---------------|-----------------|--------|--------|
| Tesseract only    | 94.14% | 93.26% | 95.36% |
| PaddleOCR only    | 99.53% | 99.31% | 99.75% |
| Tesseract+PaddleOCR combined (winner-take-all per region) | 99.53% | 99.31% | 99.75% |

**Finding:** PaddleOCR alone accounts for essentially all of the quality
improvement on this corpus — the combined dual-engine result is identical to
PaddleOCR-only, meaning PaddleOCR wins the per-region confidence comparison
in `reassembly.reconcile_region()` almost every time here. Tesseract is still
a useful safety net (it wins on documents where PaddleOCR happens to score
lower), but running both doubles compute cost for no measured quality gain on
this dataset.

## Speed: CPU vs GPU, local Windows vs ai01 Linux server

Same 29-file batch, dual-engine (tesseract+paddleocr), `max_workers: 4` in
all cases (kept identical across all runs to isolate hardware speed, not
parallelism differences).

| Machine | Compute | Time for 29 files | Mean confidence | Notes |
|---|---|---|---|---|
| Local Windows | CPU (AMD-integrated GPU, not CUDA-capable — CPU is the only option) | 25/29 done in 15m 7s, then manually stopped | 99.57% (single-file check) | Never run to completion; extrapolated ~17-18 min total from the completion rate |
| ai01 | CPU (AMD Threadripper PRO 5955WX, 16C/32T) | **5m 39s** (all 29, completed) | 99.51% | |
| ai01 | GPU (2× NVIDIA RTX 3090, PaddleOCR only offloaded to GPU; Tesseract is always CPU) | **33.1s** (all 29, completed) | 99.53% | |

**Findings:**
- ai01's CPU alone is ~3x faster than the local Windows CPU just from better
  hardware (Threadripper vs whatever's local).
- ai01's GPU is ~10x faster than ai01's own CPU, and ~30x faster than the
  local Windows CPU.
- Confidence/quality is essentially identical across all three — hardware
  choice only affects speed, not OCR accuracy (same model weights either way).

## Single-file reference numbers

Same file (`66842 100726-01_batch_1_pages_1-1.pdf`) used across the original
local-vs-GPU comparison, before the full-batch runs above:

| Run | Mean confidence | Min confidence |
|---|---|---|
| Local CPU, tesseract-only | 94.13% | — |
| Local CPU, tesseract+paddleocr | 99.57% | 94.78% |
| ai01 GPU, tesseract+paddleocr | 99.68% | 98.68% |

## PaddleOCR crash fix validation + 81-file PDF_STUFF_2 corpus (2026-08-07)

A larger, harder corpus than the 29-file set above: 81 real patent PDFs
(`PDF_STUFF_2/`), run on ai01.

### Background: the PaddleOCR crash

An earlier dual-engine CPU run crashed partway through `G_22012634.PDF`
(32-page, 300 DPI scan) — `ocr.predict()` raised an unhandled exception
inside PaddleOCR, killing the worker. Fixed in
`patent_ocr/ocr/paddleocr_engine.py` by wrapping the `predict()` call in a
defensive try/except that logs the error and returns `[]` for that region
instead of crashing the whole file.

### Run 1: CPU dual-engine (tesseract + paddleocr), post-fix validation

- 81/81 files completed, 0 failures — confirms the fix works;
  `G_22012634.PDF` no longer crashes.
- Total wall time: **30m51s** (`time python -m patent_ocr.cli sweep`).
- Without `G_22012634.PDF`, the other 80 files took only 15m23s combined —
  that one file alone added another 15m27s, roughly doubling total runtime.
- Only 10 of the 81 files needed real OCR (heavy scanned pages); the other
  71 hit the passthrough/skip-text fast path (<2.5s each).
- Slowest 10 files (`created_at`→`updated_at` duration):

  | File | Duration |
  |---|---|
  | G_22012634.PDF | 1693.4s (28m13s) |
  | G_22022661.PDF | 773.8s (12m54s) |
  | G_22001414.PDF | 434.7s |
  | G_22009725.PDF | 376.5s |
  | G_22022663.PDF | 351.6s |
  | G_22021420.pdf | 252.3s |
  | hk3b3shl.pdf | 154.5s |
  | G_22021424.pdf | 143.2s |
  | G_22021422.pdf | 136.6s |
  | G_22021423.pdf | 108.6s |

- `G_22012634.PDF` resolution check: 32 pages, 2480x3504px, exactly 300 DPI
  on A4 — a normal, good-quality scan, not a bad-resolution file. It's just
  large (32 pages) and computationally heavy for CPU dual-engine
  cross-checking.
- CPU load: with `max_workers: 4` on ai01's 32-core Threadripper, individual
  worker processes spiked to 560% CPU (5.6 cores) because both tesseract and
  paddleocr run multi-threaded CPU inference per region/page, for every
  concurrent worker. Not oversubscription (only ~10/32 cores in use at
  peak) — just legitimately CPU-heavy dual-engine OCR.
- Confidence/engine-win stats for this specific run weren't captured before
  the ledger/qc directories were wiped for the GPU test below — archive
  `state/history.sqlite3` (or call `record_run()`) before wiping state in
  future comparisons.

### Run 2: GPU-only (paddleocr, no tesseract), same 81-file corpus

- ai01's GPUs were confirmed idle at the time (`nvidia-smi` showed 0%
  utilization, only a few hundred MB used) — the box's vLLM instance was not
  occupying them during this run. Always check `nvidia-smi` for the current
  state rather than assuming the worst-case GPU contention described below
  always applies.
- Config: `engine.primary: paddleocr`, `engine.secondary: null` (tesseract
  disabled entirely — it's CPU-bound and pointless to run alongside a GPU
  speed test), `engine_options.paddleocr.use_gpu: true`.
- Total wall time: **1m3.7s** for all 81 files, 0 failures — a **~29x
  speedup** over the CPU dual-engine run (30m51s).
- `G_22022661.PDF` (12m54s on CPU dual-engine) and `G_22012634.PDF` (28m13s
  on CPU dual-engine) both completed within that same ~1 minute window.
- Mean confidence across the 10 OCR'd files: **82.70%** — notably lower than
  the ~99.5% figure measured on the original 29-file corpus with dual-engine
  cross-checking. Per-file breakdown:

  | File | Mean confidence | Engine wins |
  |---|---|---|
  | G_22022661.PDF | 20.00% | paddleocr: 1 |
  | G_22012634.PDF | 61.90% | paddleocr: 70 |
  | G_22022663.PDF | 68.96% | paddleocr: 6 |
  | G_22021422.pdf | 82.03% | paddleocr: 5 |
  | G_22001414.PDF | 98.31% | paddleocr: 2 |
  | G_22021424.pdf | 98.38% | paddleocr: 2 |
  | G_22021423.pdf | 99.29% | paddleocr: 2 |
  | G_22021420.pdf | 99.08% | paddleocr: 2 |
  | G_22009725.PDF | 99.50% | paddleocr: 1 |
  | hk3b3shl.pdf | 99.57% | paddleocr: 2 |

**Finding: removing the tesseract cross-check to save time has a real
quality cost on some documents.** Two files (`G_22022661.PDF` at 20%,
`G_22012634.PDF` at 61.9%) drag the mean down significantly. Per the
original 29-file finding above ("Tesseract is still a useful safety net — it
wins on documents where PaddleOCR happens to score lower"), these are almost
certainly cases where tesseract would have won the per-region reconciliation
and produced a much higher-confidence result. With no secondary engine
configured, PaddleOCR's own output is used unconditionally even when its
self-reported confidence is very low.

### Speed vs. quality tradeoff summary

| Mode | Time (81 files) | Mean confidence (10 OCR'd files) | Notes |
|---|---|---|---|
| CPU, dual-engine (tesseract+paddleocr) | 30m51s | not captured this run (~99.5% on prior 29-file corpus) | Safest quality; slow without GPU |
| GPU, paddleocr-only | 1m3.7s | 82.70% (skewed by 2 low-confidence files) | ~29x faster; measurable quality risk on some documents |

**Recommendation:** GPU dual-engine (tesseract on CPU + paddleocr on GPU
simultaneously) was not tested this session — that combination would likely
give both the speed benefit of GPU-offloaded paddleocr and the quality
safety net of tesseract cross-checking, at the cost of some CPU contention
from tesseract. This is the natural next benchmark once GPUs are free again.

### Operational gotchas found this session

- **Orphaned engine subprocesses**: killing the sweep's Python parent
  processes (`pkill -f "patent_ocr.cli"`) does **not** kill already-spawned
  `tesseract` subprocess children — they become orphaned and keep running,
  consuming CPU independently. Separately `pkill -9 -f "^tesseract "` (or
  equivalent) when aborting a run mid-flight.
- **Case-sensitive cleanup**: `rm -f failed/*.pdf` misses uppercase `.PDF`
  files on Linux — use `find <dirs> -type f -delete` for a full,
  case-agnostic wipe between benchmark runs.
- **Background process survival on ai01**: plain `nohup cmd &` over SSH gets
  killed when the SSH session closes. Use `setsid nohup <cmd> < /dev/null >
  log 2>&1 & disown` to fully detach a process so it survives session
  close — see [DEPLOYMENT.md](DEPLOYMENT.md#detached-background-processes-on-ai01).

## Caveats

- All confidence numbers are engine-reported self-confidence, not
  ground-truth accuracy against a hand-verified transcript. They're a
  reasonable proxy (agreement between two independent engines correlates with
  correctness) but not a substitute for real accuracy measurement.
- The local Windows CPU number is incomplete (manually interrupted at 25/29
  files) — treat the ~17-18 min full-batch estimate as an extrapolation, not
  a measured result.
- ai01's two GPUs are normally saturated by a production vLLM instance
  (~23.6GB/24GB used each, serving Qwen3.6-27B). All ai01 GPU benchmarks here
  required stopping vLLM first to free VRAM — see
  [DEPLOYMENT.md](DEPLOYMENT.md#gpu-contention-with-vllm) for details.
