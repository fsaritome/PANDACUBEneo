"""ABBYY FineReader Engine/Server adapter (§6): commercial, licensed separately.

No ABBYY SDK is vendored here (proprietary, license-gated). This stub defines
the integration point so a deployment with an ABBYY license can drop in the
real client without touching the rest of the pipeline — conforms to the same
`OCREngine` interface as every other engine.

ABBYY's SDKs (FREngine / Cloud OCR SDK / Docker container REST API) all expose
word-level bounding boxes with confidence; wire whichever transport your
license uses into `recognize()` below.
"""
from __future__ import annotations

from patent_ocr.ocr.base import OCREngine, Word


class AbbyyEngine(OCREngine):
    name = "abbyy"

    def __init__(self, server_url: str | None = None, license_path: str | None = None):
        self.server_url = server_url
        self.license_path = license_path

    def recognize(self, region_image, lang_hint: list[str] | None = None) -> list[Word]:
        raise NotImplementedError(
            "ABBYY FineReader Engine/Server integration requires a licensed SDK/container. "
            "Implement recognize() against your deployment's API "
            "(e.g. FREngine COM/API, or the ABBYY Docker/on-prem REST endpoint) "
            "and return word-level Word(text, bbox, confidence) entries."
        )
