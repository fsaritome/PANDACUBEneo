from PIL import Image

from patent_ocr.config import Config
from patent_ocr.layout.types import Region, RegionKind
from patent_ocr.ocr.base import Word
from patent_ocr.page_pipeline import _prepare_page_image, _rescale_regions


def _config(max_mp: float) -> Config:
    cfg = Config()
    cfg.preprocess.max_page_megapixels = max_mp
    return cfg


def test_no_downscale_reports_unit_scale():
    image = Image.new("RGB", (1000, 1000))
    array, scale = _prepare_page_image(image, _config(40.0))
    assert scale == 1.0
    assert array.shape[:2] == (1000, 1000)


def test_downscale_reports_the_applied_scale():
    image = Image.new("RGB", (4000, 4000))  # 16 MP, capped to 4 MP
    array, scale = _prepare_page_image(image, _config(4.0))
    height, width = array.shape[:2]
    assert width < 4000
    assert scale == width / 4000


def test_rescaling_maps_boxes_back_to_original_page_space():
    """Regression: boxes were left in downscaled coordinates while the PDF text
    layer was rendered against the original page size, offsetting every word."""
    image = Image.new("RGB", (4000, 4000))
    _, scale = _prepare_page_image(image, _config(4.0))
    region = Region(kind=RegionKind.COLUMN, bbox=(0, 0, 100, 100), order_index=0)
    region.words = [Word(text="x", bbox=(10, 20, 30, 40), confidence=90.0, engine="e")]

    _rescale_regions([region], 1.0 / scale)

    assert region.bbox[2] == round(100 / scale)
    assert region.words[0].bbox[0] == round(10 / scale)
    assert region.words[0].bbox[3] == round(40 / scale)


def test_rescaling_preserves_word_text_and_confidence():
    region = Region(kind=RegionKind.COLUMN, bbox=(0, 0, 10, 10), order_index=0)
    region.words = [Word(text="hello", bbox=(1, 2, 3, 4), confidence=77.5, engine="paddleocr")]
    _rescale_regions([region], 2.0)
    word = region.words[0]
    assert word.text == "hello"
    assert word.confidence == 77.5
    assert word.engine == "paddleocr"
    assert word.bbox == (2, 4, 6, 8)
