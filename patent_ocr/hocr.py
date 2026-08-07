"""hOCR document builder.

Converts the final, reading-order-correct region/word stream into hOCR XML so
OCRmyPDF's own HocrTransform can composite it as an invisible, word-positioned
text layer under the untouched page image (§5.10) — we don't reinvent that
compositing step, only feed it correctly-ordered input.
"""
from __future__ import annotations

from xml.sax.saxutils import escape

from patent_ocr.layout.types import Region
from patent_ocr.ocr.base import Word

_HOCR_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" \
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
<title></title>
<meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
<meta name='ocr-system' content='patent-ocr' />
<meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word'/>
</head>
<body>
"""


def group_words_into_lines(words: list[Word]) -> list[list[Word]]:
    """Cluster a flat word list into text lines by vertical overlap, since our
    OCR engines return words, not pre-grouped lines."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    lines: list[list[Word]] = []
    current: list[Word] = [ordered[0]]
    current_top = ordered[0].bbox[1]
    current_height = ordered[0].bbox[3] - ordered[0].bbox[1]
    for w in ordered[1:]:
        threshold = max(4, 0.6 * current_height)
        if abs(w.bbox[1] - current_top) <= threshold:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: x.bbox[0]))
            current = [w]
            current_top = w.bbox[1]
            current_height = w.bbox[3] - w.bbox[1]
    lines.append(sorted(current, key=lambda x: x.bbox[0]))
    return lines


def _bbox_str(bbox: tuple[int, int, int, int]) -> str:
    return f"bbox {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"


def build_hocr(page_width: int, page_height: int, regions: list[Region], dpi: int = 300) -> str:
    """Build a full hOCR document for one page. `regions` must already be in
    final reading order (see reassembly.assemble_page)."""
    parts = [_HOCR_HEADER]
    parts.append(
        f"<div class='ocr_page' id='page_1' title='image \"page\"; {_bbox_str((0, 0, page_width, page_height))}; ppageno 0'>\n"
    )
    for r_idx, region in enumerate(regions):
        parts.append(f"<div class='ocr_carea' id='block_1_{r_idx}' title=\"{_bbox_str(region.bbox)}\">\n")
        parts.append(f"<p class='ocr_par' id='par_1_{r_idx}' title=\"{_bbox_str(region.bbox)}\">\n")
        lines = group_words_into_lines(region.words)
        for l_idx, line_words in enumerate(lines):
            line_bbox = (
                min(w.bbox[0] for w in line_words),
                min(w.bbox[1] for w in line_words),
                max(w.bbox[2] for w in line_words),
                max(w.bbox[3] for w in line_words),
            )
            parts.append(
                f"<span class='ocr_line' id='line_1_{r_idx}_{l_idx}' title=\"{_bbox_str(line_bbox)}\">\n"
            )
            for w_idx, word in enumerate(line_words):
                conf = max(0, min(100, int(round(word.confidence))))
                parts.append(
                    f"<span class='ocrx_word' id='word_1_{r_idx}_{l_idx}_{w_idx}' "
                    f"title=\"{_bbox_str(word.bbox)}; x_wconf {conf}\">"
                    f"{escape(word.text)}</span>\n"
                )
            parts.append("</span>\n")
        parts.append("</p>\n</div>\n")
    parts.append("</div>\n</body>\n</html>\n")
    return "".join(parts)
