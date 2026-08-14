"""Layout region types shared by the segmenter and reassembly stages (§5.6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from patent_ocr.ocr.base import BBox, Word


class RegionKind(str, Enum):
    MARGIN_NUMBERS = "margin_numbers"
    COLUMN = "column"
    FIGURE = "figure"
    FORMULA = "formula"
    TABLE = "table"
    TITLE = "title"
    FULL_PAGE = "full_page"
    OTHER = "other"


class LayoutType(str, Enum):
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN_MARGIN = "two_column_margin"
    OTHER = "other"


@dataclass
class Region:
    kind: RegionKind
    bbox: BBox  # left, top, right, bottom in *page* pixel coordinates
    order_index: int  # fixed reading-order position: margin -> col A -> col B -> figures
    column_index: int | None = None  # 0 = leftmost column, 1 = next, ...
    words: list[Word] = field(default_factory=list)  # populated after OCR + remap
    # Structured representation when the layout model provides one (table HTML).
    html: str | None = None

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
