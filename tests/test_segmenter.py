import numpy as np

from patent_ocr.config import LayoutConfig
from patent_ocr.layout.segmenter import segment_page
from patent_ocr.layout.types import LayoutType, RegionKind


def _make_two_column_page(width=1000, height=600):
    # White background, ink = 0 (black) after RGB->gray, matches typical scanned text.
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    col_a = (20, 450)
    col_b = (550, 980)
    for col in (col_a, col_b):
        y = 20
        while y < height - 20:
            img[y : y + 4, col[0] : col[1]] = 0  # a thin "text line"
            y += 30
    return img


def test_segment_page_detects_two_columns_no_margin():
    img = _make_two_column_page()
    layout_type, regions = segment_page(img, LayoutConfig())
    kinds = [r.kind for r in regions]
    assert kinds.count(RegionKind.COLUMN) == 2
    # Reading order left-to-right
    assert regions[0].bbox[0] < regions[1].bbox[0]


def test_segment_page_single_column_when_no_gap():
    width, height = 800, 400
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    img[100:104, 20:780] = 0
    layout_type, regions = segment_page(img, LayoutConfig())
    assert layout_type == LayoutType.SINGLE_COLUMN
    assert len(regions) == 1
    assert regions[0].kind == RegionKind.FULL_PAGE
