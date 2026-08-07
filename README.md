# Patent OCR Pipeline

Self-hosted Kofax replacement: watches a hot folder, OCRs patent PDFs (mixed
EN/DE/FR, mixed scanned/text-native, USPTO/EPO/DPMA/WIPO), and writes them
back as searchable PDFs with a word-level positioned invisible text layer —
mirroring the input directory tree exactly.

Priority: **correctness/quality over everything else.** No destructive
preprocessing (despeckle/denoise) is ever applied.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
```

External dependencies (not pip-installable): [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
and Ghostscript must be on `PATH` (used by OCRmyPDF). GPU engines (PaddleOCR,
Surya) are optional extras: `pip install -e .[paddle]` / `.[surya]`.

## Configure

Copy [config.example.yaml](config.example.yaml) to `config.yaml` and adjust
`watcher.input_root` / `watcher.output_root` at minimum.

## Run

```powershell
# Live: watcher + startup backlog sweep + worker pool (foreground)
patent-ocr --config config.yaml run

# One-off backlog sweep only (no live watch), exits when the queue drains
patent-ocr --config config.yaml sweep

# Ledger status counts
patent-ocr --config config.yaml status

# Files flagged for human review (low-confidence fallback couldn't resolve)
patent-ocr --config config.yaml report-flagged
```

## Architecture

1. **Watcher + backlog sweep** (`watcher.py`, `pipeline.py`) — `watchdog`-based
   hot folder watch, plus an idempotent full-tree sweep on every startup
   (content-hash based, via `ledger.py`'s SQLite state store). Files that
   raise during processing are moved out of `input_root` into `failed_root`
   (mirrored tree), so they aren't retried on every sweep and are easy to find.
2. **Passthrough** (`passthrough.py`) — text-native PDFs with sane extractable
   text are copied through untouched; anything else goes through OCR.
3. **Layout segmentation** (`layout/segmenter.py`) — heuristic column/margin
   detection *before* OCR, fixing reading order for the two-column-with-
   margin-line-numbers patent layout (the #1 accuracy risk).
4. **OCR engines** (`ocr/`) — pluggable: Tesseract, PaddleOCR, Surya, ABBYY
   (stub). Confidence-driven secondary-engine/LLM-fallback strategy in
   `confidence.py` / `fallback/`.
5. **Reassembly + hOCR** (`reassembly.py`, `hocr.py`) — per-region reconciliation,
   final reading order, hOCR document build.
6. **OCRmyPDF plugin** (`ocrmypdf_plugin.py`, `pdf_text_layer.py`) — OCRmyPDF
   still owns rasterization, `--skip-text` page protection, and PDF assembly;
   our plugin supplies the hOCR/text-layer content instead of Tesseract's own.
7. **Compositor** (`compositor.py`) — invokes `ocrmypdf.ocr()`; `clean`,
   `clean_final`, `remove_background` are hard-coded `False` (never
   configurable) — only `deskew` is an opt-in non-destructive preprocessing step.
8. **QC** (`qc.py`) — per-page QC aggregated into a `.qc.json` sidecar written
   under `watcher.qc_root`, mirroring the input tree path-for-path (kept
   separate from `output_root`, which only ever contains OCR'd PDFs); flagged
   files are queryable via `report-flagged`.

## Out of scope

Zonal/field extraction, a validation UI, scanner/ISIS drivers, forms
processing, destructive preprocessing, and cloud OCR APIs are explicitly not
part of this project.

## Admin dashboard

A FastAPI + React dashboard (`patent_ocr/api/`, `dashboard/`) provides live
processing status, per-file failure logs, engine-attribution charts, and
persistent cross-run history. Deployed alongside the pipeline on ai01 at
`http://ai01:8000/`. See
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#admin-dashboard-fastapi--react) for
build/deploy steps.

## Further documentation

- [docs/OPERATIONS.md](docs/OPERATIONS.md) — reprocessing/delete-on-success
  behavior, CLI flags, dual-engine reconciliation.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — deploying to the `ai01` GPU
  server (system deps, Python 3.11 requirement, GPU-enabled PaddlePaddle,
  vLLM GPU contention, detached background processes, admin dashboard).
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — measured OCR quality and speed
  comparisons (Tesseract vs PaddleOCR, local CPU vs ai01 CPU vs ai01 GPU,
  81-file corpus dual-engine vs GPU-only speed/quality tradeoff).
