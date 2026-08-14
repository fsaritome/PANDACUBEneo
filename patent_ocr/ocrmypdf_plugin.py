"""OCRmyPDF plugin: swaps Tesseract for our full layout-aware pipeline (§5.10).

OCRmyPDF still owns the actual "sandwich" composition (page rasterization,
--skip-text page-level protection, final PDF/image handling) — we only
replace *how the text layer is produced* for pages OCRmyPDF decides to OCR.

`OcrEngine`'s methods are staticmethods per OCRmyPDF's plugin spec, so there's
no instance state to carry a `Config` object through. Instead the compositor
sets `PATENT_OCR_CONFIG_PATH` in the environment before calling
`ocrmypdf.ocr(...)`, and this plugin reads it once per worker process.
"""
from __future__ import annotations

import os
from pathlib import Path

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

from patent_ocr import __version__
from patent_ocr.config import Config, load_config
from patent_ocr.docx_export import write_page_content
from patent_ocr.page_pipeline import PageResult, process_page_image
from patent_ocr.qc import write_page_qc

# One page's worth of work is identical whether ocrmypdf calls generate_hocr()
# or generate_pdf() for it (some versions call one, some the other) — cache by
# input path so we never run the (expensive) OCR pipeline twice for one page.
_PAGE_CACHE: dict[str, PageResult] = {}
_CONFIG_CACHE: Config | None = None


def _get_config() -> Config:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config(os.environ.get("PATENT_OCR_CONFIG_PATH"))
    return _CONFIG_CACHE


def _get_page_result(input_file: Path) -> PageResult:
    key = str(input_file)
    if key not in _PAGE_CACHE:
        result = process_page_image(input_file, _get_config())
        # input_file is OCRmyPDF's page raster; its name carries the page order.
        write_page_qc(result.qc, input_file.name)
        write_page_content(result.regions_for_render, input_file.name)
        _PAGE_CACHE[key] = result
    return _PAGE_CACHE[key]


class PatentOcrEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        return __version__

    @staticmethod
    def creator_tag(options) -> str:
        return f"patent-ocr {__version__}"

    def __str__(self) -> str:
        return f"patent-ocr pipeline {__version__} (layout-aware, pluggable engines)"

    @staticmethod
    def languages(options) -> set[str]:
        return set(_get_config().languages)

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        # Page rotation detection is out of scope for this pipeline (§1); defer
        # to OCRmyPDF/tesseract's own orientation handling by expressing no opinion.
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        result = _get_page_result(Path(input_file))
        Path(output_hocr).write_text(result.hocr_xml, encoding="utf-8")
        Path(output_text).write_text(result.text, encoding="utf-8")

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options) -> None:
        # Import here to keep this module importable even before reportlab/PIL
        # are needed (this staticmethod is only invoked by newer ocrmypdf versions
        # that call generate_pdf directly instead of generate_hocr).
        from PIL import Image

        from patent_ocr.pdf_text_layer import render_invisible_text_pdf

        result = _get_page_result(Path(input_file))
        Path(output_text).write_text(result.text, encoding="utf-8")
        with Image.open(input_file) as im:
            width_px, height_px = im.size
            dpi = im.info.get("dpi", (300, 300))[0]
        render_invisible_text_pdf(result.regions_for_render, width_px, height_px, dpi, output_pdf)


@hookimpl
def get_ocr_engine(options) -> OcrEngine:
    return PatentOcrEngine()
