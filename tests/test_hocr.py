from patent_ocr.hocr import build_hocr, group_words_into_lines
from patent_ocr.layout.types import Region, RegionKind
from patent_ocr.ocr.base import Word


def _word(text, x0, y0, x1, y1, conf=90.0):
    return Word(text=text, bbox=(x0, y0, x1, y1), confidence=conf)


def test_group_words_into_lines_clusters_by_row():
    words = [
        _word("hello", 0, 0, 40, 20),
        _word("world", 45, 2, 90, 22),
        _word("second", 0, 30, 50, 50),
        _word("line", 55, 31, 90, 51),
    ]
    lines = group_words_into_lines(words)
    assert len(lines) == 2
    assert [w.text for w in lines[0]] == ["hello", "world"]
    assert [w.text for w in lines[1]] == ["second", "line"]


def test_build_hocr_contains_words_in_order():
    region = Region(kind=RegionKind.FULL_PAGE, bbox=(0, 0, 100, 100), order_index=0)
    region.words = [_word("Claim", 0, 0, 40, 20), _word("one", 45, 0, 70, 20)]
    xml = build_hocr(100, 100, [region])
    assert "ocrx_word" in xml
    assert xml.index("Claim") < xml.index("one")
    assert "ocr_page" in xml
