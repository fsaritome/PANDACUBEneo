"""Invisible, word-positioned PDF text layer (R4).

Builds the text-only PDF directly from our Region/Word data — no
version-fragile dependency on OCRmyPDF's internal hOCR-to-PDF machinery
(that internal API has changed across OCRmyPDF releases; this is the same
"invisible text render mode" technique OCRmyPDF's own hOCR transform has
historically used, just self-contained).
"""
from __future__ import annotations

from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from patent_ocr.layout.types import Region

_FONT_NAME = "Helvetica"


def render_invisible_text_pdf(
    regions: list[Region],
    page_width_px: int,
    page_height_px: int,
    dpi: float,
    output_pdf: str | Path,
) -> None:
    """Write a single-page PDF containing only invisible, positioned text.

    OCRmyPDF grafts this onto the original page image — the image itself is
    untouched (R6), and word boxes here are in the same pixel space as the
    OCR engines produced them, remapped to page coordinates already.
    """
    px_to_pt = 72.0 / dpi
    width_pt = page_width_px * px_to_pt
    height_pt = page_height_px * px_to_pt

    c = canvas.Canvas(str(output_pdf), pagesize=(width_pt, height_pt))
    for region in regions:
        for word in region.words:
            text = word.text.strip()
            if not text:
                continue
            x0, y0, x1, y1 = word.bbox
            box_w_pt = (x1 - x0) * px_to_pt
            box_h_pt = (y1 - y0) * px_to_pt
            if box_w_pt <= 0 or box_h_pt <= 0:
                continue

            font_size = max(1.0, box_h_pt * 0.88)
            natural_width = stringWidth(text, _FONT_NAME, font_size) or 1.0
            h_scale = (box_w_pt / natural_width) * 100.0

            text_x = x0 * px_to_pt
            text_y = height_pt - (y1 * px_to_pt) + font_size * 0.15

            c.saveState()
            c.setFont(_FONT_NAME, font_size)
            c.setHorizScale(h_scale)
            c._code.append("3 Tr")  # invisible text render mode
            c.drawString(text_x, text_y, text)
            c._code.append("0 Tr")
            c.restoreState()
    c.showPage()
    c.save()
