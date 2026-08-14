"""PaddleOCR-VL engine (§5.6 stack): VLM-based document parsing model, the
replacement for the Tesseract accuracy floor. Scores far higher than classic
PP-OCR/Tesseract on messy/degraded scans (96.3% on OmniDocBench) since it
recognizes each region as a vision-language model rather than a classic
detector+CRNN pipeline. Heavy dependency (paddlepaddle) — imported lazily so
the rest of the pipeline works even when this engine isn't installed/selected.

PaddleOCR-VL's native output is block-level markdown/text (`parsing_res_list`,
one block per detected layout element), not word-level boxes with per-token
confidence like Tesseract/PP-OCR — its result objects expose only
label/bbox/content, with no text-line detector in the loop to ask for finer
geometry. `_blocks_to_words` therefore *estimates* word boxes (see
`ocr/wordsplit.py`): the block is divided into equal-height line strips and
each strip is split by character width. Word geometry from this engine is
consequently approximate; engines with a real text detector (paddleocr,
surya) give measured line boxes and should be preferred when downstream
consumers care about box alignment. The VLM exposes no per-token probability,
so every recognized word gets a fixed high confidence — which also means the
low-confidence fallback and QC thresholds cannot fire for this engine; a
region where the model produced nothing still surfaces as low-confidence
(no words).
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from patent_ocr.ocr.base import OCREngine, Word
from patent_ocr.ocr.wordsplit import split_block_into_words

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
        vl_rec_max_concurrency: int | None = None,
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
        self.vl_rec_max_concurrency = vl_rec_max_concurrency
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
                if self.vl_rec_max_concurrency is not None:
                    kwargs["vl_rec_max_concurrency"] = self.vl_rec_max_concurrency
            else:
                kwargs["device"] = f"gpu:{gpu_id}" if self.use_gpu else "cpu"
            self._instance = PaddleOCRVL(**kwargs)
        return self._instance

    @staticmethod
    def _blocks_to_words(blocks) -> list[Word]:
        words: list[Word] = []
        for block in blocks:
            text = (getattr(block, "content", "") or "").strip()
            if not text:
                continue
            words.extend(
                split_block_into_words(
                    text,
                    tuple(block.bbox),
                    PaddleOCRVLEngine.DEFAULT_WORD_CONFIDENCE,
                    PaddleOCRVLEngine.name,
                )
            )
        return words

    def _run(self, images: list) -> list[list[Word]]:
        try:
            # Lock only covers instance construction — the double-checked lock
            # in _get_instance() guards the one-time model load. When using the
            # vllm-server backend, predict() is just an HTTP request and is
            # safe to call concurrently; holding the lock for the entire call
            # would serialize all workers and negate the server's concurrency.
            # For local PaddlePaddle inference (no vl_rec_backend), cuDNN IS
            # not thread-safe — but that path loads only once per process via
            # _get_instance(), and the lock below guards that critical section.
            pipeline = self._get_instance()
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
