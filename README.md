# Patent OCR Pipeline

Self-hosted Kofax replacement: watches a hot folder, OCRs patent PDFs (mixed
EN/DE/FR, mixed scanned/text-native, USPTO/EPO/DPMA/WIPO), and writes them
back as searchable PDFs with a word-level positioned invisible text layer —
mirroring the input directory tree exactly. Optionally also emits a `.docx`
per document, built from the same ordered regions so both formats carry
identical text.

Priority: **correctness/quality over everything else.** No destructive
preprocessing (despeckle/denoise) is ever applied.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
```

External dependencies (not pip-installable): Ghostscript must be on `PATH` (used
by OCRmyPDF). The primary OCR engine is classic **PaddleOCR** (PP-OCR) with
`return_word_box` for measured word-level boxes, and layout comes from
PaddleOCR's trained **PP-StructureV3** model — see
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for GPU setup.
GPU extras: `pip install -e .[paddle]`. DOCX output: `pip install -e .[docx]`.

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
   Startup also reconciles rows stranded in `processing` by a previous hard
   kill, and clears orphaned work dirs.
2. **Passthrough** (`passthrough.py`) — text-native PDFs with sane extractable
   text are copied through untouched; anything else goes through OCR.
3. **Layout segmentation** (`layout/`) — two selectable backends via
   `layout.backend`. `ppstructure` (default) uses PaddleOCR's trained
   PP-DocLayout model for regions, reading order, semantic labels and
   table/formula detection; `heuristic` is the original OpenCV projection-
   profile column/margin detector. Reading order is decided *before* OCR,
   fixing the two-column-with-margin-line-numbers patent layout (the #1
   accuracy risk). Patent margin line-numbers are recovered explicitly
   (`layout/line_numbers.py`) because the layout model does not emit a region
   for them.
4. **OCR engines** (`ocr/`) — pluggable: PaddleOCR (primary, PP-OCR with
   measured word boxes), PaddleOCR-VL (VLM, Docker vLLM backend), Surya,
   ABBYY (stub). `OCREngine.operates_on_full_page = True` signals page-level
   engines that receive whole pages instead of pre-cropped regions.
   Confidence-driven secondary-engine/LLM-fallback strategy in
   `confidence.py` / `fallback/`.
5. **Reassembly + hOCR** (`reassembly.py`, `hocr.py`) — per-region reconciliation,
   final reading order, hOCR document build.
6. **OCRmyPDF plugin** (`ocrmypdf_plugin.py`, `pdf_text_layer.py`) — OCRmyPDF
   still owns rasterization, `--skip-text` page protection, and PDF assembly;
   our plugin supplies the hOCR/text-layer content instead of Tesseract's own.
7. **Compositor** (`compositor.py`) — invokes `ocrmypdf.ocr()`; `clean`,
   `clean_final`, `remove_background` are hard-coded `False` (never
   configurable) — only `deskew` is an opt-in non-destructive preprocessing step.
8. **DOCX export** (`docx_export.py`, optional) — assembles the same ordered
   regions into a Word document: tables become real Word tables, figures are
   embedded as cropped images, titles become headings. Enable with
   `watcher.emit_docx`.
9. **QC** (`qc.py`) — per-page QC aggregated into a `.qc.json` sidecar written
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
  behavior, CLI flags, layout backends, DOCX output, dual-engine reconciliation.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — deploying to the `ai01` GPU
  server (system deps, Python 3.11 requirement, GPU-enabled PaddlePaddle,
  vLLM GPU contention, detached background processes, admin dashboard).
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — measured OCR quality and speed
  comparisons, including the 2026-08-14 box-geometry fixes and the
  heuristic-vs-PP-StructureV3 layout A/B over a 40-document corpus sample.

## Author & License

Author: **fsaritome**

Licensed under a custom non-commercial license — free to use, modify, and
distribute for personal, educational, or research purposes only. Commercial
use requires prior written permission from the author. See
[LICENSE](LICENSE) for full terms.
