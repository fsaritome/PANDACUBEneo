"""DOCX export (optional second output format alongside the searchable PDF).

Built from the pipeline's own ordered regions rather than PaddleX's built-in
`save_to_word`, because that would serialize PP-StructureV3's internal OCR
text while the PDF carries our configured engine's words - two outputs of the
same document disagreeing on their contents is not acceptable for documents
that get cited. Here both formats are rendered from one source of truth.

Page content is staged per page during OCR (the plugin runs per page, often in
another process) and assembled once the sandwich finishes.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from patent_ocr.layout.types import RegionKind
from patent_ocr.qc import page_sort_key

log = logging.getLogger(__name__)

_CONTENT_SUFFIX = ".page.json"

# Region kinds whose text is body prose rather than furniture.
_BODY_KINDS = {RegionKind.COLUMN.value, RegionKind.FULL_PAGE.value}


def write_page_content(regions, source_name: str = "") -> None:
    """Stage one page's ordered text blocks next to its QC file."""
    qc_dir = os.environ.get("PATENT_OCR_QC_DIR")
    if not qc_dir:
        return
    blocks = []
    for region in regions:
        text = " ".join(w.text for w in region.words).strip()
        if text:
            blocks.append({"kind": region.kind.value, "text": text})
    if not blocks:
        return
    path = Path(qc_dir)
    path.mkdir(parents=True, exist_ok=True)
    name = f"{page_sort_key(source_name)}_{uuid.uuid4().hex}{_CONTENT_SUFFIX}"
    (path / name).write_text(json.dumps({"blocks": blocks}), encoding="utf-8")


def read_pages(work_dir: Path) -> list[list[dict]]:
    """All staged pages, in page order (filenames carry the sort prefix)."""
    if not work_dir.exists():
        return []
    pages = []
    for file in sorted(work_dir.glob(f"*{_CONTENT_SUFFIX}")):
        try:
            pages.append(json.loads(file.read_text(encoding="utf-8"))["blocks"])
        except (OSError, ValueError, KeyError):
            log.warning("skipping unreadable page content file: %s", file)
    return pages


def write_docx(work_dir: Path, output_path: Path, title: str = "") -> bool:
    """Assemble staged pages into one .docx. Returns False if nothing was written."""
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt
    except ImportError:
        log.warning("python-docx not installed; skipping DOCX output (pip install python-docx)")
        return False

    pages = read_pages(work_dir)
    if not pages:
        return False

    document = Document()
    if title:
        document.add_heading(title, level=1)

    for index, blocks in enumerate(pages):
        if index:
            document.add_page_break()
        for block in blocks:
            kind = block["kind"]
            paragraph = document.add_paragraph()
            run = paragraph.add_run(block["text"])
            if kind == RegionKind.MARGIN_NUMBERS.value:
                # Patent line numbers: keep them, but visually subordinate so
                # they cannot be mistaken for claim text.
                run.font.size = Pt(8)
                run.italic = True
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            elif kind not in _BODY_KINDS:
                run.font.size = Pt(9)
                run.italic = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output_path))
    return True
