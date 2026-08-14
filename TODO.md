# TODO

## Resolved (2026-08-14) — box geometry

Three bugs were silently corrupting word-box geometry on every document:

1. **UVDoc unwarping left enabled.** `PaddleOCR(...)` defaults leave
   `use_doc_unwarping` / `use_doc_orientation_classify` on. Unwarping
   geometrically transforms the page and PaddleOCR returns boxes in *that*
   space, but the pipeline sandwiches them under the untouched image — so
   every box was offset by a non-constant 75–95px. Found by rendering
   detected boxes over ground-truth text; numeric assertions alone had
   masked it. Both flags are now forced `False`.
2. **Downscale coordinate mismatch.** `_prepare_page_image()` shrank pages
   over `max_page_megapixels` but boxes stayed in downscaled coordinates
   while the plugin rendered against the original page size. Now returns
   `(array, scale)` and rescales regions before emitting hOCR.
3. **VL block splitting.** `_blocks_to_words` excluded spaces from the
   denominator while advancing the cursor past them (boxes overran the
   block) and gave every word the whole block's vertical extent.

Also: `return_word_box=True` now yields *measured* per-token boxes;
`ocr/wordsplit.py` is the proportional fallback for engines without word
geometry.

## Resolved (2026-08-14) — faint-glyph detection

PaddleOCR's default `text_det_box_thresh` (0.6) is too strict for faint
isolated glyphs: on a real 300dpi EP claims scan it dropped the first two
margin line-numbers while keeping the rest, so numbering appeared to start at
15. Set to `0.3`. Regression-checked over 40 documents: chars/page 2086→2081,
words 431→432, empty pages 0%→0%, confidence 98.9→98.6 (the dip is expected —
the recovered glyphs *are* the faint low-confidence ones).

Note: an earlier hypothesis blamed PaddleOCR's 4000px input cap. That was
**wrong** — `text_det_limit_side_len=4000` still missed the digits.
`preprocess.max_side_px` was added anyway, since silent downsampling is a real
hazard worth bounding, but it was not the fix.

## Resolved (2026-08-14) — concurrency

The per-file QC directory was passed via the `PATENT_OCR_QC_DIR` environment
variable, but GPU mode runs files concurrently in a `ThreadPoolExecutor` whose
threads share `os.environ`. Whichever file started last overwrote it, so one
file's pages could be aggregated into another file's QC sidecar and the
loser's DOCX was silently dropped (reproduced with two documents in one
sweep). The staging directory is now derived deterministically from the output
path (`qc.staging_dir`), resolved by the plugin from `options.output_file`.

**Any concurrent GPU run before 2026-08-14 may have mixed pages between files
in its QC data.**

Related: `write_page_qc` named per-page files with `uuid4`, so
`aggregate_qc_dir`'s `sorted(glob(...))` returned pages in random order and
every index in `flagged_pages` referred to an arbitrary page. Page files now
carry a zero-padded sort key derived from OCRmyPDF's raster filenames.

## Resolved (2026-08-14) — interrupted runs

`Ledger.reconcile_interrupted()` + `cli._startup_reconcile()` clear rows
stranded in `processing` by a hard kill, plus orphaned work dirs. Previously
nothing ever cleared them, so a single `kill -9` left the dashboard showing
phantom "live" work with unbounded elapsed time (8 such rows were found dating
back three days).

## Superseded (2026-08-11 entries)

The PaddleOCR-VL non-determinism and QC/PDF text-layer mismatch entries are no
longer the active configuration. The primary engine is classic **PaddleOCR**,
chosen because PaddleOCR-VL has no text detector at all — verified in
`paddlex/inference/pipelines/paddleocr_vl/result.py`, which exposes only
label/bbox/content per block. Word geometry from it is pure estimation and its
confidence is a hardcoded constant, which silently disables the
low-confidence, QC-flagging and reconciliation machinery. It remains available
via `engine.primary: paddleocr_vl`.

## Open

1. **Line-number detection rate is unmeasured.** `layout/line_numbers.py`
   validates an arithmetic signature and works 4/4 pages on the reference EP
   claims document, but has not been measured across the corpus. Other
   numbering conventions (every line rather than every 5th, right-margin
   numbering) are untested. Measure before treating
   `watcher.strip_line_numbers` as production-ready.
2. **Region *labelling* is unstable across raster resolution.** A sweep at
   200/250/300/350/400dpi showed margin-region assignment flipping — at 350dpi
   zero margin regions were produced even though every word was captured. Word
   capture is solid; classification is not. Affects reading-order grouping
   rather than whether text exists.
3. **Table cell text bypasses the OCR engine.** Table contents come from
   PP-Structure's table recognizer, not from the configured engine's words, so
   within tables the PDF and DOCX could in principle diverge. Not observed,
   but it is the one seam in the "single source of truth" guarantee.
4. **`layout/segmenter.py` and most of `ocr/wordsplit.py` are now largely dead
   weight.** The heuristic backend remains selectable but is measurably inert
   (1.2 regions/page). Kept as a fallback while `ppstructure` beds in; remove
   once it has proven itself in production.
5. **Figure/table/formula region kinds are barely exercised.** `_LABEL_KINDS`
   maps them, but testing only ever surfaced `text`/`header`/`number`/
   `paragraph_title` plus tables and one figure on a single search report.
6. **`disable_passthrough: true` forces re-OCR of everything**, including pages
   that already carry good text. On the RENK corpus 609 of 663 files were pure
   passthrough, so this is the dominant cost driver for full-corpus runs.

## Next steps

- Measure line-number detection across a corpus sample; then decide whether
  `strip_line_numbers` can be recommended to end users.
- Re-run the full STAND_DER_TECHNIK corpus (976 files) end to end and record
  the numbers in `docs/BENCHMARKS.md`.
- Consider deleting the heuristic layout backend once `ppstructure` has run a
  full production corpus without regression.
