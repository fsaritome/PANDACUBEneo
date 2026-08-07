"""Tesseract engine (§5.6 stack): the cheapest baseline, used as the accuracy floor."""
from __future__ import annotations

import numpy as np
import pytesseract
from PIL import Image

from patent_ocr.ocr.base import OCREngine, Word

# Map ISO 639-1/2 hints used elsewhere in the pipeline to tesseract's traineddata names.
_LANG_MAP = {
    "en": "eng", "eng": "eng",
    "de": "deu", "deu": "deu", "ger": "deu",
    "fr": "fra", "fra": "fra",
}


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, config_str: str = "--oem 1 --psm 6"):
        self.config_str = config_str

    def recognize(self, region_image, lang_hint: list[str] | None = None) -> list[Word]:
        if isinstance(region_image, np.ndarray):
            image = Image.fromarray(region_image)
        else:
            image = region_image

        lang = "+".join(dict.fromkeys(_LANG_MAP.get(h, h) for h in (lang_hint or ["eng"])))
        data = pytesseract.image_to_data(
            image, lang=lang, config=self.config_str, output_type=pytesseract.Output.DICT
        )

        words: list[Word] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 0:
                conf = 0.0
            left, top = data["left"][i], data["top"][i]
            width, height = data["width"][i], data["height"][i]
            words.append(
                Word(
                    text=text,
                    bbox=(left, top, left + width, top + height),
                    confidence=conf,
                    engine=self.name,
                )
            )
        return words
