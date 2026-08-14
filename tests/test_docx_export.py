import json

import pytest

from patent_ocr.docx_export import read_pages, write_docx
from patent_ocr.layout.types import Region, RegionKind
from patent_ocr.ocr.base import Word
from patent_ocr.qc import page_sort_key


def _region(kind: RegionKind, *words: str) -> Region:
    region = Region(kind=kind, bbox=(0, 0, 10, 10), order_index=0)
    region.words = [Word(w, (0, 0, 5, 5), 99.0, "e") for w in words]
    return region


def test_page_sort_key_orders_by_ocrmypdf_page_number():
    """uuid4 filenames sorted randomly, scrambling page order and every
    flagged_pages index."""
    keys = [page_sort_key(n) for n in
            ("000010_rasterize.png", "000002_rasterize.png", "000001_rasterize.png")]
    assert sorted(keys) == [page_sort_key("000001_rasterize.png"),
                            page_sort_key("000002_rasterize.png"),
                            page_sort_key("000010_rasterize.png")]


def test_page_sort_key_falls_back_when_there_is_no_number():
    assert page_sort_key("") == "99999999"
    assert page_sort_key("page.png") == "99999999"


def test_read_pages_returns_pages_in_order(tmp_path):
    for name, text in (("000003_a.page.json", "third"),
                       ("000001_b.page.json", "first"),
                       ("000002_c.page.json", "second")):
        (tmp_path / name).write_text(
            json.dumps({"blocks": [{"kind": "column", "text": text}]}), encoding="utf-8"
        )
    assert [p[0]["text"] for p in read_pages(tmp_path)] == ["first", "second", "third"]


def test_read_pages_skips_corrupt_files(tmp_path):
    (tmp_path / "000001_a.page.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "000002_b.page.json").write_text(
        json.dumps({"blocks": [{"kind": "column", "text": "ok"}]}), encoding="utf-8"
    )
    assert [p[0]["text"] for p in read_pages(tmp_path)] == ["ok"]


def test_write_docx_reports_false_with_no_pages(tmp_path):
    assert write_docx(tmp_path, tmp_path / "out.docx") is False


def test_write_docx_produces_a_readable_document(tmp_path):
    pytest.importorskip("docx")
    from docx import Document

    (tmp_path / "000001_a.page.json").write_text(
        json.dumps({"blocks": [
            {"kind": "margin_numbers", "text": "5 10 15"},
            {"kind": "column", "text": "A spinal bone fastener assembly"},
        ]}),
        encoding="utf-8",
    )
    out = tmp_path / "out.docx"
    assert write_docx(tmp_path, out, title="claims") is True
    assert out.exists()

    text = "\n".join(p.text for p in Document(str(out)).paragraphs)
    assert "A spinal bone fastener assembly" in text
    assert "5 10 15" in text


def test_write_page_content_is_a_noop_without_a_staging_dir(monkeypatch):
    from patent_ocr.docx_export import write_page_content

    monkeypatch.delenv("PATENT_OCR_QC_DIR", raising=False)
    write_page_content([_region(RegionKind.COLUMN, "hello")])  # must not raise
