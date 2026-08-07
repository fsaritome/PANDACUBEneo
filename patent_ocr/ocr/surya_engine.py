"""Surya engine (§5.6 stack): free, GPU-accelerated, modern transformer OCR.
Heavy dependency (torch + surya-ocr) — imported lazily.

Surya's native output is line-level. Where the installed version exposes
word-level boxes (`TextLine.words`) we use those directly; otherwise we
approximate word boxes by splitting the line bbox proportionally to each
word's character span. That approximation is flagged via lower confidence
is NOT applied automatically — see reassembly/confidence handling, which
treats these as ordinary word confidences from the underlying recognizer.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from patent_ocr.ocr.base import OCREngine, Word

_LANG_MAP = {
    "en": "en", "eng": "en",
    "de": "de", "deu": "de", "ger": "de",
    "fr": "fr", "fra": "fr",
}


class SuryaEngine(OCREngine):
    name = "surya"

    def __init__(self):
        self._det_predictor = None
        self._rec_predictor = None

    def _ensure_loaded(self):
        if self._rec_predictor is not None:
            return
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise ImportError(
                "surya-ocr is not installed. Install extras: pip install patent-ocr[surya]"
            ) from exc
        self._det_predictor = DetectionPredictor()
        self._rec_predictor = RecognitionPredictor()

    def recognize(self, region_image, lang_hint: list[str] | None = None) -> list[Word]:
        self._ensure_loaded()
        if isinstance(region_image, np.ndarray):
            image = Image.fromarray(region_image)
        else:
            image = region_image

        langs = [_LANG_MAP.get(h, h) for h in (lang_hint or ["en"])]
        predictions = self._rec_predictor([image], [langs], self._det_predictor)

        words: list[Word] = []
        for page_pred in predictions:
            for line in page_pred.text_lines:
                line_words = getattr(line, "words", None)
                if line_words:
                    for w in line_words:
                        words.append(
                            Word(
                                text=w.text,
                                bbox=tuple(int(v) for v in w.bbox),
                                confidence=float(getattr(w, "confidence", line.confidence)) * 100.0,
                                engine=self.name,
                            )
                        )
                else:
                    words.extend(self._split_line_into_words(line))
        return words

    @staticmethod
    def _split_line_into_words(line) -> list[Word]:
        """Approximate per-word boxes by splitting a line bbox proportionally
        to character span, for surya versions that only expose line-level boxes."""
        text = line.text
        tokens = text.split()
        if not tokens:
            return []
        x0, y0, x1, y1 = (int(v) for v in line.bbox)
        total_chars = sum(len(t) for t in tokens)
        words: list[Word] = []
        cursor = 0
        span_width = x1 - x0
        for tok in tokens:
            frac_start = cursor / total_chars if total_chars else 0
            frac_end = (cursor + len(tok)) / total_chars if total_chars else 1
            wx0 = x0 + int(frac_start * span_width)
            wx1 = x0 + int(frac_end * span_width)
            words.append(
                Word(
                    text=tok,
                    bbox=(wx0, y0, wx1, y1),
                    confidence=float(line.confidence) * 100.0,
                    engine="surya",
                )
            )
            cursor += len(tok) + 1
        return words
