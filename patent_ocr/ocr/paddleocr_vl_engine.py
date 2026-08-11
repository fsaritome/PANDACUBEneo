"""PaddleOCR-VL engine (§5.6 stack): VLM-based document parsing model, the
replacement for the Tesseract accuracy floor. Scores far higher than classic
PP-OCR/Tesseract on messy/degraded scans (96.3% on OmniDocBench) since it
recognizes each region as a vision-language model rather than a classic
detector+CRNN pipeline. Heavy dependency (paddlepaddle) — imported lazily so
the rest of the pipeline works even when this engine isn't installed/selected.

PaddleOCR-VL's native output is block-level markdown/text (`parsing_res_list`,
one block per detected layout element), not word-level boxes with per-token
confidence like Tesseract/PP-OCR. `_blocks_to_words` adapts that block-level
content into this pipeline's per-word `Word` contract by splitting each
block's text into whitespace tokens and distributing them proportionally
across the block's bbox width — the same approximation `surya_engine` uses
for engines that don't expose word-level boxes. The VLM exposes no per-token
probability, so every recognized word gets a fixed high confidence; a region
where the model produced nothing still surfaces as low-confidence (no words).
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from patent_ocr.ocr.base import OCREngine, Word

log = logging.getLogger(__name__)

# The VLM recognizer is inherently multilingual and takes no language switch;
# lang_hint is accepted (for interface parity with other engines) but unused.


class PaddleOCRVLEngine(OCREngine):
    name = "paddleocr_vl"
    # Confirmed via live benchmark: fed an already-cropped region (a column or
    # margin-number strip), its internal layout detector finds no blocks and
    # returns zero words. Fed a whole page, it works well. So the page
    # pipeline must call this engine once per page, not once per region.
    operates_on_full_page = True

    # No per-token probability is exposed by the VL recognizer; every word from
    # a successfully produced block gets this fixed confidence.
    DEFAULT_WORD_CONFIDENCE = 95.0

    def __init__(
        self,
        use_gpu: bool = True,
        gpu_id: int | None = None,
        pipeline_version: str = "v1.6",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_chart_recognition: bool = False,
        use_seal_recognition: bool = False,
        use_layout_detection: bool | None = None,
        # When set, delegates the VLM stage to the official genai server
        # (Docker-hosted vLLM) per PaddleOCR docs section 3.2 — the layout
        # analysis model still runs locally, only VLM inference is offloaded.
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
    ):
        self.use_gpu = use_gpu
        # gpu_id=None: let PADDLE_GPU_ID env var or CUDA_VISIBLE_DEVICES decide (set per-worker).
        self.gpu_id = gpu_id
        self.pipeline_version = pipeline_version
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_chart_recognition = use_chart_recognition
        self.use_seal_recognition = use_seal_recognition
        # None: leave layout detection at the pipeline's own default. We already
        # feed it a pre-cropped region from our own layout segmenter, but the
        # region can still contain multiple lines/blocks worth detecting.
        self.use_layout_detection = use_layout_detection
        self.vl_rec_backend = vl_rec_backend
        self.vl_rec_server_url = vl_rec_server_url
        self._instance = None
        # The pipeline (like most PaddleX/cuDNN-backed predictors) is not safe
        # to call concurrently from multiple threads on one shared instance —
        # confirmed via a live benchmark: concurrent calls from a GPU-mode
        # ThreadPoolExecutor raised CUDNN_STATUS_INTERNAL_ERROR. Serialize
        # calls here rather than one instance per thread (which would
        # multiply this multi-GB VLM's memory footprint per worker).
        self._lock = threading.Lock()

    def _get_instance(self):
        if self._instance is None:
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR-VL is not installed or too old. Install extras: "
                    "pip install patent-ocr[paddle] (requires paddleocr>=3.2 for the "
                    "PaddleOCRVL pipeline)."
                ) from exc
            import os

            gpu_id = self.gpu_id
            if gpu_id is None and self.use_gpu:
                try:
                    gpu_id = int(os.environ.get("PADDLE_GPU_ID", "0"))
                except ValueError:
                    gpu_id = 0
            kwargs = dict(
                pipeline_version=self.pipeline_version,
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
                use_chart_recognition=self.use_chart_recognition,
                use_seal_recognition=self.use_seal_recognition,
                use_layout_detection=self.use_layout_detection,
            )
            if self.vl_rec_backend and self.vl_rec_server_url:
                # Delegate VLM inference to the Docker-hosted vLLM server;
                # layout analysis still runs locally on the client GPU.
                kwargs["vl_rec_backend"] = self.vl_rec_backend
                kwargs["vl_rec_server_url"] = self.vl_rec_server_url
            else:
                kwargs["device"] = f"gpu:{gpu_id}" if self.use_gpu else "cpu"
            self._instance = PaddleOCRVL(**kwargs)
        return self._instance

    @staticmethod
    def _blocks_to_words(blocks) -> list[Word]:
        words: list[Word] = []
        for block in blocks:
            text = (getattr(block, "content", "") or "").strip()
            tokens = text.split()
            if not tokens:
                continue
            x0, y0, x1, y1 = block.bbox
            total_chars = sum(len(t) for t in tokens)
            span_width = x1 - x0
            cursor = 0
            for tok in tokens:
                frac_start = cursor / total_chars if total_chars else 0
                frac_end = (cursor + len(tok)) / total_chars if total_chars else 1
                wx0 = x0 + int(frac_start * span_width)
                wx1 = x0 + int(frac_end * span_width)
                words.append(
                    Word(
                        text=tok,
                        bbox=(wx0, y0, wx1, y1),
                        confidence=PaddleOCRVLEngine.DEFAULT_WORD_CONFIDENCE,
                        engine=PaddleOCRVLEngine.name,
                    )
                )
                cursor += len(tok) + 1
        return words

    def _run(self, images: list) -> list[list[Word]]:
        try:
            # Lock covers instance construction too, not just predict() — with
            # multiple file-processing threads calling a fresh engine's first
            # recognize() nearly simultaneously, an unlocked _get_instance()
            # could otherwise race and build/discard a duplicate GPU model.
            with self._lock:
                pipeline = self._get_instance()
                # NOTE: do NOT pass temperature=0.0 here — tried that to fix
                # observed non-determinism (see repo memory), but it triggers
                # `RuntimeError: int(Tensor) is not supported in static graph
                # mode` on nearly every real call, confirmed via a 663-file
                # live run (only ~2/10 real-OCR files succeeded, vs the
                # pipeline's own sampling default which at least sometimes
                # works). Left at the pipeline default until root-caused.
                results = list(pipeline.predict(images))
        except Exception:
            log.warning("paddleocr_vl predict() failed on a batch, skipping", exc_info=True)
            return [[] for _ in images]
        out = [self._blocks_to_words(res["parsing_res_list"]) for res in results]
        while len(out) < len(images):
            out.append([])
        return out

    def recognize(self, region_image, lang_hint: list[str] | None = None) -> list[Word]:
        image = np.asarray(region_image)
        return self._run([image])[0]

    def recognize_batch(self, region_images: list, lang_hint: list[str] | None = None) -> list[list[Word]]:
        """Run one GPU call for all crops; returns one Word list per input image."""
        if not region_images:
            return []
        images = [np.asarray(img) for img in region_images]
        return self._run(images)
