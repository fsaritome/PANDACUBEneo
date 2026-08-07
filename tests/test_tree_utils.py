import pytest

from patent_ocr.tree_utils import output_path_for, iter_input_files
from pathlib import Path


def test_output_path_for_mirrors_relative_structure(tmp_path):
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    nested = input_root / "USPTO" / "2024" / "doc.pdf"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"%PDF-1.4\n")

    result = output_path_for(nested, input_root, output_root)
    assert result == (output_root / "USPTO" / "2024" / "doc.pdf").resolve()


def test_iter_input_files_filters_by_extension(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.PDF").write_bytes(b"x")

    files = iter_input_files(tmp_path, [".pdf"])
    names = sorted(p.name for p in files)
    assert names == ["a.pdf", "c.PDF"]
