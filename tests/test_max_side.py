import pytest
from PIL import Image

from patent_ocr.config import Config
from patent_ocr.page_pipeline import _prepare_page_image


def _cfg(max_mp: float = 40.0, max_side: int = 4000) -> Config:
    cfg = Config()
    cfg.preprocess.max_page_megapixels = max_mp
    cfg.preprocess.max_side_px = max_side
    return cfg


def test_long_side_is_capped_below_paddleocrs_internal_limit():
    """OCRmyPDF rasterized A4 at 400dpi (3306x4678); PaddleOCR silently
    downsamples anything over 4000px/side and drops small marginalia."""
    array, scale = _prepare_page_image(Image.new("RGB", (3306, 4678)), _cfg())
    height, width = array.shape[:2]
    assert max(width, height) <= 4000
    # Scale reports the resize actually applied, so it absorbs integer rounding.
    assert scale == pytest.approx(4000 / 4678, abs=1e-3)


def test_pages_under_the_cap_are_untouched():
    array, scale = _prepare_page_image(Image.new("RGB", (2480, 3508)), _cfg())
    assert scale == 1.0
    assert array.shape[:2] == (3508, 2480)


def test_megapixel_cap_still_applies_and_wins_when_stricter():
    # 6000x6000 = 36MP: under 40MP, but far over the side cap.
    array, scale = _prepare_page_image(Image.new("RGB", (6000, 6000)), _cfg())
    assert max(array.shape[:2]) <= 4000
    assert scale == pytest.approx(4000 / 6000, abs=1e-3)


def test_side_cap_can_be_disabled():
    array, scale = _prepare_page_image(Image.new("RGB", (5000, 5000)), _cfg(max_side=0))
    assert scale == 1.0
    assert array.shape[:2] == (5000, 5000)
