# TODO

## Resolved (2026-08-11)

- **paddleocr_vl via Docker vLLM server** — fully working. 663 RENK files processed,
  0 failures, 63m 9s wall time. See `docs/BENCHMARKS.md` for details.
- **Non-deterministic output** — resolved by switching from direct PaddlePaddle
  local inference to the Docker-hosted vLLM server (stable, deterministic).
- **temperature=0.0 regression** — reverted; the vLLM server handles its own
  sampling correctly. Do not pass `temperature`/`top_p` from the client.
- **Pipeline architecture** — `operates_on_full_page = True` on
  `PaddleOCRVLEngine`: pipeline feeds full page images (not pre-cropped
  regions), then distributes returned words back into layout regions by bbox
  overlap. This matches PaddleOCR-VL's expected input format.

## Still open

1. **QC/PDF text-layer mismatch** (lower priority — production run showed 0
   failures and correct OCR output): at least one file in early testing had
   words in the QC sidecar but an empty text layer in the output PDF.
   Not reproduced in the production 663-file run. Investigate only if it
   resurfaces in real usage — compare `generate_hocr()` vs `generate_pdf()`
   branch in `ocrmypdf_plugin.py` on an affected file.

## Pending improvements

- **GPU split**: run layout model on GPU 1 (`gpu_id: 1` in engine_options) so
  vLLM on GPU 0 has no contention. Currently both share GPU 0. Not yet tested
  but the config knob (`gpu_id`) is already wired in `PaddleOCRVLEngine`.
- **predict() lock removal benchmark**: with the lock removed from `predict()`
  and `vl_rec_max_concurrency: 8`, the vLLM server was at ~26% GPU utilization
  on the second RENK run (vs 0% previously). A full timing comparison vs the
  first run (63m 9s, 8 workers serialized) is pending completion of that run.

- Docker group membership: `sudo usermod -aG docker install`, then reconnect
- genai client plugin: `pip install 'openai>=1.63'` in the pipeline venv
- See `docs/DEPLOYMENT.md#paddleocr-vl-docker-vllm-server` for full setup


1. **Non-deterministic empty output**: rerunning the identical small test
   corpus gave different files zero recognized words on different runs.
   Tried pinning `temperature=0.0, top_p=1.0` in `paddleocr_vl_engine.py`'s
   `predict()` call to force greedy decoding — **this made things worse, not
   better**: a full 663-file RENK ingestion with that change active showed
   nearly every real-OCR call failing with `RuntimeError: int(Tensor) is not
   supported in static graph mode` (confirmed via live run, ~2026-08-11
   08:2x). **Reverted** — back to the pipeline's own (sampling) default,
   which at least succeeds some of the time. Root cause of the original
   non-determinism is still open; whatever fix is tried next must be smoke
   tested on a handful of files before a full corpus run, not the other way
   around.

2. **QC/PDF text-layer mismatch**: at least one file (`G_22021423.pdf`) had
   correct non-zero word counts in its QC sidecar JSON, but the final output
   PDF's text layer was completely empty (`pypdf.extract_text()` returned
   ~0 chars). Recognition succeeded; something between QC recording and PDF
   rendering (hOCR building in `hocr.py`, or `pdf_text_layer.py`'s
   `render_invisible_text_pdf`, or the `generate_hocr` vs `generate_pdf`
   branch in `ocrmypdf_plugin.py`) drops the text. **Likely engine-independent**
   — needs to be reproduced with the plain `paddleocr` engine on the same
   file to confirm before assuming it's paddleocr_vl-specific.

## Full RENK ingestion timing (2026-08-11, exploratory, DO NOT treat as a
   valid production run — output quality is compromised by bug #1 above)

663 files, `disable_passthrough: false` (normal skip-text behavior), 4
workers / 2 GPUs, paddleocr_vl primary, no secondary. Wall time: **1m5s**,
0 files marked "failed" — but that number is misleading:
- **609/663 files** hit the pure passthrough/skip-text fast path (already had
  a usable text layer) — near-instant, not a real OCR throughput measurement.
- Only **~10 files** actually needed real OCR (`Parsing N pages with
  HocrParser` in the log) — consistent with the ~10/81 ratio in the original
  PDF_STUFF_2 baseline.
- Of those ~10, the majority hit `paddleocr_vl predict() failed on a batch,
  skipping` (silently caught, region ends up with zero words, file still
  marked "done" since our pipeline doesn't treat an empty OCR result as a
  hard failure) — this run happened while the broken `temperature=0.0` change
  above was still active, so it's not a valid quality/reliability measurement
  either. **Needs rerunning** now that that regression is reverted, before
  drawing any real conclusion about paddleocr_vl's throughput or reliability
  at RENK scale.

## Next steps

- Reproduce bug #2 in isolation with a single known-bad file, inspecting
  `PageResult.regions_for_render` directly before it reaches
  `render_invisible_text_pdf`/hOCR, to find where the text is dropped.
- Once #1 and #2 are resolved, re-benchmark paddleocr_vl against the
  documented classic-paddleocr baseline in `docs/BENCHMARKS.md`.
- Consider whether `disable_passthrough` interacts with either bug (both
  observed bugs were on `force_ocr` pages).
