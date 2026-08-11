"""PaddleOCR engine (§5.6 stack): free, GPU-accelerated, stronger than Tesseract on
degraded scans. Heavy dependency (paddlepaddle) — imported lazily so the rest of
the pipeline works even when this engine isn't installed/selected.

Targets PaddleOCR >=3.0's pipeline API (predict()/OCRResult), which replaced the
old .ocr(cls=True) list-of-tuples interface used in 2.x.
"""
from __future__ import annotations

import logging

import numpy as np

from patent_ocr.ocr.base import OCREngine, Word

log = logging.getLogger(__name__)

_LANG_MAP = {
    "en": "en", "eng": "en",
    "de": "german", "deu": "german", "ger": "german",
    "fr": "french", "fra": "french",
}


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def __init__(self, use_gpu: bool = True, gpu_id: int | None = None, default_lang: str = "en",
                 single_instance: bool = False, use_textline_orientation: bool = True):
        self.use_gpu = use_gpu
        # gpu_id=None: let PADDLE_GPU_ID env var or CUDA_VISIBLE_DEVICES decide (set per-worker).
        self.gpu_id = gpu_id
        self.default_lang = default_lang
        # single_instance=True: one model loaded once (default_lang only); ignores per-region lang hints.
        # Cuts VRAM from ~6-9GB/worker (3 models) to ~2-3GB/worker (1 model).
        self.single_instance = single_instance
        self.use_textline_orientation = use_textline_orientation
        self._instances: dict[str, object] = {}

    def _get_instance(self, lang: str):
        if lang not in self._instances:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR is not installed. Install extras: pip install patent-ocr[paddle]"
                ) from exc
            import os
            # Honour explicit gpu_id, then fall back to PADDLE_GPU_ID env var.
            gpu_id = self.gpu_id
            if gpu_id is None and self.use_gpu:
                try:
                    gpu_id = int(os.environ.get("PADDLE_GPU_ID", "0"))
                except ValueError:
                    gpu_id = 0
            device = f"gpu:{gpu_id}" if self.use_gpu else "cpu"
            kwargs = dict(
                lang=lang,
                use_textline_orientation=self.use_textline_orientation,
                device=device,
            )
            if not self.use_gpu:
                # PIR/oneDNN executor crashes on CPU builds (ConvertPirAttribute2RuntimeAttribute
                # NotImplementedError) — disabling mkldnn is the known workaround; mkldnn is a
                # CPU-only optimization so it's irrelevant (and not passed) on GPU.
                kwargs["enable_mkldnn"] = False
            self._instances[lang] = PaddleOCR(**kwargs)
        return self._instances[lang]

    def _parse_result(self, res: dict) -> list[Word]:
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        boxes = res.get("rec_boxes")
        polys = res.get("rec_polys")
        words: list[Word] = []
        for i, text in enumerate(texts):
            conf = float(scores[i]) * 100.0 if i < len(scores) else 0.0
            if boxes is not None and i < len(boxes):
                x0, y0, x1, y1 = boxes[i]
                bbox = (int(x0), int(y0), int(x1), int(y1))
            else:
                quad = polys[i]
                xs = [p[0] for p in quad]
                ys = [p[1] for p in quad]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            words.append(Word(text=text, bbox=bbox, confidence=conf, engine=self.name))
        return words

    def recognize(self, region_image, lang_hint: list[str] | None = None) -> list[Word]:
        if self.single_instance:
            lang = _LANG_MAP.get(self.default_lang, self.default_lang)
        else:
            lang = _LANG_MAP.get((lang_hint or [self.default_lang])[0], self.default_lang)
        ocr = self._get_instance(lang)
        image = np.asarray(region_image)
        try:
            results = ocr.predict(image)
        except Exception:
            log.warning("paddleocr recognize() failed on a region, skipping", exc_info=True)
            return []
        words: list[Word] = []
        for res in results or []:
            words.extend(self._parse_result(res))
        return words

    def recognize_batch(self, region_images: list, lang_hint: list[str] | None = None) -> list[list[Word]]:
        """Run one GPU call for all crops; returns one Word list per input image."""
        if not region_images:
            return []
        if self.single_instance:
            lang = _LANG_MAP.get(self.default_lang, self.default_lang)
        else:
            lang = _LANG_MAP.get((lang_hint or [self.default_lang])[0], self.default_lang)
        ocr = self._get_instance(lang)
        images = [np.asarray(img) for img in region_images]
        try:
            # predict() accepts a list of arrays and returns one result dict per image
            results = ocr.predict(images)
        except Exception:
            log.warning("paddleocr recognize_batch() failed, falling back to per-image", exc_info=True)
            return [self.recognize(img, lang_hint) for img in region_images]
        out: list[list[Word]] = []
        for res in results or []:
            out.append(self._parse_result(res))
        # Pad to match input length if predict() dropped any result
        while len(out) < len(region_images):
            out.append([])
        return out
