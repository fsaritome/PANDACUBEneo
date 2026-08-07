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

    def __init__(self, use_gpu: bool = True, default_lang: str = "en"):
        self.use_gpu = use_gpu
        self.default_lang = default_lang
        self._instances: dict[str, object] = {}

    def _get_instance(self, lang: str):
        if lang not in self._instances:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR is not installed. Install extras: pip install patent-ocr[paddle]"
                ) from exc
            kwargs = dict(
                lang=lang,
                use_textline_orientation=True,
                device="gpu" if self.use_gpu else "cpu",
            )
            if not self.use_gpu:
                # PIR/oneDNN executor crashes on CPU builds (ConvertPirAttribute2RuntimeAttribute
                # NotImplementedError) — disabling mkldnn is the known workaround; mkldnn is a
                # CPU-only optimization so it's irrelevant (and not passed) on GPU.
                kwargs["enable_mkldnn"] = False
            self._instances[lang] = PaddleOCR(**kwargs)
        return self._instances[lang]

    def recognize(self, region_image, lang_hint: list[str] | None = None) -> list[Word]:
        lang = _LANG_MAP.get((lang_hint or [self.default_lang])[0], self.default_lang)
        ocr = self._get_instance(lang)

        image = np.asarray(region_image)
        try:
            results = ocr.predict(image)
        except Exception:
            # PaddleX's internal pipeline has known edge cases (e.g. a rec_score
            # coming back as a list instead of a float) that crash predict()
            # outright. Don't let one bad region take down the whole file —
            # degrade to "no secondary output" so reconciliation falls back to
            # the primary engine for this region.
            log.warning("paddleocr recognize() failed on a region, skipping", exc_info=True)
            return []

        words: list[Word] = []
        for res in results or []:
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            boxes = res.get("rec_boxes")
            polys = res.get("rec_polys")
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
