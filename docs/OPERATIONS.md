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

## Dual-engine reconciliation

`engine.secondary` supports exactly **one** secondary engine (not an
arbitrary list). With `engine.strategy: always_parallel`, both the primary
and secondary engine run on every region; `reassembly.reconcile_region()`
picks whichever engine scored higher `mean_conf()` for that entire region and
uses its output — this is a per-region winner-take-all, not a per-word
blend of the two engines' output.

See [BENCHMARKS.md](BENCHMARKS.md) for measured evidence that PaddleOCR
currently wins nearly every region against Tesseract on the test corpus, so
the "combined" result is effectively identical to running PaddleOCR alone.

## `engine.engine_options`

Per-engine constructor kwargs, keyed by engine name:

```yaml
engine:
  engine_options:
    paddleocr: { use_gpu: true }   # false on CPU-only machines
```

Currently only `use_gpu` (bool) is consumed by `PaddleOCREngine.__init__`.
