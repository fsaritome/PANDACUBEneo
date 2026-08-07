"""Engine factory: builds `OCREngine` instances by name from config (§5.7).

Keeps the pipeline decoupled from any specific engine — swap primary/secondary
per deployment (or per document) by editing config, not code.
"""
from __future__ import annotations

from patent_ocr.ocr.base import OCREngine

_ENGINES: dict[str, str] = {
    "tesseract": "patent_ocr.ocr.tesseract_engine.TesseractEngine",
    "paddleocr": "patent_ocr.ocr.paddleocr_engine.PaddleOCREngine",
    "surya": "patent_ocr.ocr.surya_engine.SuryaEngine",
    "abbyy": "patent_ocr.ocr.abbyy_engine.AbbyyEngine",
}


def build_engine(name: str, **kwargs) -> OCREngine:
    if name not in _ENGINES:
        raise ValueError(f"Unknown OCR engine '{name}'. Available: {sorted(_ENGINES)}")
    module_path, class_name = _ENGINES[name].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    return engine_cls(**kwargs)
