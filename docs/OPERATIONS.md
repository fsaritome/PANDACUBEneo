# Operational Behavior

## Default: always re-OCR everything dropped into `input/`

By design, the pipeline **always reprocesses** a file when it's dropped into
`input_root` or found there on startup — even if a file with identical
content was already processed before (ledger shows `done`/`flagged`). This
is intentional: a file sitting in `input/` is an explicit instruction to OCR
it, not a request to consult history first.

- Live watcher drops (`watcher.py`'s `_Handler._maybe_submit()`): always
  `force=True`, unconditionally — not configurable. Dropping a file is
  always an explicit action.
- Startup/CLI backlog sweep (`backlog_sweep()`): defaults to `force=True` as
  well. Pass `--skip-done` on the CLI to restore the old idempotent
  behavior (skip files whose content hash already has a `done`/`flagged`
  ledger record) — useful for re-running `sweep` repeatedly over a large
  archive without redoing unchanged work:

  ```bash
  patent-ocr --config config.yaml sweep --skip-done
  patent-ocr --config config.yaml run --skip-done
  ```

## Delete-on-success

Successfully processed files (both the passthrough path and the full-OCR
path) are **deleted** from `input_root` after the output is written —not
archived or moved. This keeps `input/` acting as a true work queue that
drains to empty on success. Combined with the always-reprocess default
above, this also prevents infinite reprocessing loops on every restart
(a file can't be both "always reprocessed" and "left sitting in input/
forever" without one of those causing runaway repeated work).

Files that raise an exception during processing are moved to `failed_root`
(mirrored directory tree) instead of being deleted or left in `input/` — so
failures are inspectable and aren't silently retried on every sweep.

## Interrupted-run reconciliation

A `SIGKILL` (OOM killer, `kill -9`, host reboot) skips `process_file`'s
`except`/`finally`, so its ledger row stays `processing` forever and the
dashboard renders it as live work with an ever-growing elapsed time. Nothing
used to clear these; a single killed run poisoned the dashboard permanently.

`cli._startup_reconcile()` now runs at the top of **both** `run` and `sweep`:
it marks any lingering `processing` row as `failed` with
`interrupted: process exited before completion`, and removes orphaned
`state/work/*` directories left behind by the same kill.

## Dual-engine reconciliation

`engine.secondary` supports exactly **one** secondary engine (not an
arbitrary list). With `engine.strategy: always_parallel`, both the primary
and secondary engine run on every region; `reassembly.reconcile_region()`
picks whichever engine scored higher `mean_conf()` for that entire region and
uses its output — this is a per-region winner-take-all, not a per-word
blend of the two engines' output.

See [BENCHMARKS.md](BENCHMARKS.md) for measured results. Classic **PaddleOCR**
(PP-OCR) is the current primary engine: it has a real text detector, so it
returns measured line/word boxes and genuine per-line confidence. PaddleOCR-VL
does neither — it exposes only block-level boxes and a hardcoded confidence
constant, which silently disables the low-confidence, QC-flagging and
reconciliation machinery.

## Layout backends

`layout.backend` selects how regions and reading order are produced:

| value | behaviour |
|---|---|
| `ppstructure` (default) | PaddleOCR's trained PP-DocLayout model: semantic region labels, reading order, table/formula regions |
| `heuristic` | OpenCV projection-profile column/margin detection |

Measured over a 40-document corpus sample, text extraction is identical
between the two (2085 vs 2086 chars/page), but the heuristic segmenter
averaged **1.2 regions/page** — a single full-page region on ~34 of 40
documents, i.e. effectively no segmentation — versus **17.4** for
PP-StructureV3, at ~1.5x the page time.

`layout.ppstructure_words` controls where words come from under that backend:

- `engine` (default) — the configured OCR engine supplies word-level boxes.
- `builtin` — PP-StructureV3's own OCR, which is **line-level only**;
  `PPStructureV3` rejects `return_word_box` outright, so word geometry is not
  obtainable from it.

## Patent margin line-numbers

The trained layout model emits no region for patent line numbering, and its
text regions frequently swallow the numbers. `layout/line_numbers.py`
therefore detects them page-wide by their *arithmetic* signature — numeric
tokens sharing a stable edge, strictly ascending, constant step (tolerating
gaps where OCR missed one) — rather than by position alone. Position-only
detection misread the vertical sidebar publication number `EP 1 439 083 A3`
as line numbering and was unstable across raster resolution.

`watcher.strip_line_numbers` (default `false`) omits them from the `.docx`.
The searchable PDF **always retains them**: its text layer sits under the
original page image, so removing them there would leave visible digits that
cannot be selected or searched.

## DOCX output

`watcher.emit_docx` (default `false`) writes a `.docx` beside each output PDF,
built from the same ordered regions so both formats carry identical text.
Requires `pip install patent-ocr[docx]`; without it the export logs a warning
and is skipped. DOCX failures are caught so they can never fail a good PDF.

Structure is preserved where the layout model supplies it: recognized tables
become real Word tables, figure regions are embedded as cropped images, and
titles become headings.

## `engine.engine_options`

Per-engine constructor kwargs, keyed by engine name:

```yaml
engine:
  primary: paddleocr
  engine_options:
    paddleocr:
      use_gpu: true
      single_instance: true          # one model for `languages`; avoids 3x VRAM
      use_textline_orientation: true
      return_word_box: true          # real measured per-token boxes
      split_lines_into_words: true   # fallback estimate when the above is unavailable
      text_det_box_thresh: 0.3       # see below
```

`text_det_box_thresh` — PaddleOCR's default (0.6) is too strict for faint,
isolated glyphs. On a real 300dpi EP claims scan it silently dropped the first
two margin line-numbers (`5`, `10`) while keeping `15`–`35`, so the visible
numbering began at 15. `0.3` recovers all of them at a measured cost of
0.3 percentage points of mean confidence and no text inflation.

`use_doc_unwarping` / `use_doc_orientation_classify` are forced `False` and
should stay that way: both geometrically transform the page, and PaddleOCR
then returns boxes in that transformed space while the pipeline sandwiches
them under the *untouched* image. Leaving unwarping on offset every box by a
non-constant 75–95px.

`preprocess.max_side_px` (default 4000) caps the raster's longest side,
because PaddleOCR silently downsamples anything larger — capping it here keeps
the resample under our control, where boxes are correctly rescaled back.
`preprocess.oversample_dpi` is a *minimum* raster DPI; values above ~300 on A4
exceed that 4000px ceiling and cause double transcoding for no gain.
