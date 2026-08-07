"""Heuristic layout segmenter (§5.6).

Column-gap + margin-strip detection via OpenCV projection profiles. This is
intentionally rule-based, not learned — good enough for the dominant patent
claims-page shape (margin line numbers + two body columns), upgradable to a
learned layout model later if real documents show it doesn't hold up.

The segmenter's output ordering IS the reading order used downstream: margin
numbers first, then columns left-to-right, then figure/formula blocks last.
This directly addresses §3.2 (reading order is the #1 accuracy risk) by
making region order an explicit, inspectable pipeline artifact instead of an
incidental side effect of raw OCR.
"""
from __future__ import annotations

import cv2
import numpy as np

from patent_ocr.config import LayoutConfig
from patent_ocr.layout.types import LayoutType, Region, RegionKind


def _binarize(image: np.ndarray) -> np.ndarray:
    """Return a 0/1 ink mask (1 = ink) via Otsu threshold. Read-only, non-destructive."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _find_gap_runs(profile: np.ndarray, density_threshold: float, min_width_px: int) -> list[tuple[int, int]]:
    """Contiguous index ranges where `profile` stays below `density_threshold`."""
    peak = profile.max() if profile.size else 0
    if peak <= 0:
        return []
    is_gap = profile < (density_threshold * peak)
    runs: list[tuple[int, int]] = []
    start = None
    for i, gap in enumerate(is_gap):
        if gap and start is None:
            start = i
        elif not gap and start is not None:
            if i - start >= min_width_px:
                runs.append((start, i))
            start = None
    if start is not None and len(is_gap) - start >= min_width_px:
        runs.append((start, len(is_gap)))
    return runs


def _bands_from_gaps(gaps: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    """Content bands are whatever lies between (and around) the detected gaps."""
    bands: list[tuple[int, int]] = []
    cursor = 0
    for gstart, gend in gaps:
        if gstart > cursor:
            bands.append((cursor, gstart))
        cursor = gend
    if cursor < total:
        bands.append((cursor, total))
    return bands


def _band_has_content(binary: np.ndarray, x0: int, x1: int, min_ink_ratio: float = 0.001) -> bool:
    region = binary[:, x0:x1]
    if region.size == 0:
        return False
    return (region.sum() / region.size) > min_ink_ratio


def _detect_line_bands(binary_band: np.ndarray, min_gap_px: int = 3) -> list[tuple[int, int]]:
    """Row bands with ink, used to estimate whether a block is text (many thin,
    evenly spaced lines) or a figure/formula (few, irregular dense blocks)."""
    row_profile = binary_band.sum(axis=1).astype(np.float64)
    gaps = _find_gap_runs(row_profile, density_threshold=0.05, min_width_px=min_gap_px)
    return _bands_from_gaps(gaps, len(row_profile))


def _classify_band_kind(binary_band: np.ndarray) -> RegionKind:
    """Figures/formulas tend to be a few dense, unevenly-spaced ink blocks rather
    than many thin, evenly spaced text lines. Heuristic v1 per §5.9."""
    h, w = binary_band.shape
    if h == 0 or w == 0:
        return RegionKind.COLUMN
    line_bands = _detect_line_bands(binary_band)
    fill_ratio = binary_band.sum() / (h * w)
    avg_line_height = (h / len(line_bands)) if line_bands else h
    # Many short, thin bands => body text. Few bands and/or high fill => figure/formula.
    if len(line_bands) >= 4 and avg_line_height < 0.15 * h:
        return RegionKind.COLUMN
    if fill_ratio > 0.12 or len(line_bands) <= 2:
        return RegionKind.FIGURE
    return RegionKind.COLUMN


def segment_page(image: np.ndarray, config: LayoutConfig | None = None) -> tuple[LayoutType, list[Region]]:
    """Segment one page image into ordered regions.

    Returns (layout_type, regions) where regions are already assigned
    order_index per the fixed reading order: margin -> columns (L to R) -> figures.
    """
    config = config or LayoutConfig()
    h, w = image.shape[:2]
    binary = _binarize(image)

    col_profile = binary.sum(axis=0).astype(np.float64)
    min_gap_px = max(1, int(config.min_gap_fraction * w))
    gaps = _find_gap_runs(col_profile, config.gap_density_threshold, min_gap_px)
    bands = [
        (x0, x1) for (x0, x1) in _bands_from_gaps(gaps, w)
        if _band_has_content(binary, x0, x1)
    ]

    if len(bands) <= 1:
        region = Region(kind=RegionKind.FULL_PAGE, bbox=(0, 0, w, h), order_index=0)
        return LayoutType.SINGLE_COLUMN, [region]

    margin_region: Region | None = None
    column_bands = list(bands)
    x0, x1 = bands[0]
    band_width_fraction = (x1 - x0) / w
    if band_width_fraction <= config.margin_max_width_fraction:
        margin_region = Region(kind=RegionKind.MARGIN_NUMBERS, bbox=(x0, 0, x1, h), order_index=0)
        column_bands = bands[1:]

    regions: list[Region] = []
    order = 0
    if margin_region is not None:
        margin_region.order_index = order
        regions.append(margin_region)
        order += 1

    figure_regions: list[Region] = []
    col_idx = 0
    for x0, x1 in column_bands:
        band_binary = binary[:, x0:x1]
        kind = _classify_band_kind(band_binary)
        if kind == RegionKind.FIGURE:
            figure_regions.append(Region(kind=RegionKind.FIGURE, bbox=(x0, 0, x1, h), order_index=-1))
            continue
        regions.append(Region(kind=RegionKind.COLUMN, bbox=(x0, 0, x1, h), order_index=order, column_index=col_idx))
        order += 1
        col_idx += 1

    for fig in figure_regions:
        fig.order_index = order
        regions.append(fig)
        order += 1

    layout_type = LayoutType.TWO_COLUMN_MARGIN if margin_region is not None else LayoutType.OTHER
    return layout_type, regions
