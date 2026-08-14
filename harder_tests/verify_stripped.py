"""Paragraph-level check: are margin line-number blocks gone from the DOCX?

Raw substring matching cannot answer this - '10' also appears inside claim
text as the reference numeral '( 10 )'. Only whole paragraphs consisting
purely of an ascending numeric run are line numbering.
"""
import re
import sys
from pathlib import Path

import pypdf
from docx import Document

result = Path("harder_tests/result")
docx_path = result / (sys.argv[1] if len(sys.argv) > 1 else "EN Ansprueche_test.docx")
pdf_path = result / "EN Ansprueche_test_SEARCHABLE.pdf"

paras = [p.text.strip() for p in Document(str(docx_path)).paragraphs if p.text.strip()]

NUMRUN = re.compile(r"^\d+(\s+\d+)*$")
numeric_paras = [p for p in paras if NUMRUN.match(p)]

print(f"docx paragraphs            : {len(paras)}")
print(f"pure-numeric paragraphs    : {len(numeric_paras)}")
for p in numeric_paras:
    print(f"    LEFTOVER: {p!r}")

body_ok = any("spinal bone fastener assembly" in p for p in paras)
claims = [p for p in paras if re.match(r"^\d+\s*\.", p)]
print(f"\nbody text intact           : {body_ok}")
print(f"numbered claim paragraphs  : {len(claims)}")

pdf_text = " ".join((p.extract_text() or "") for p in pypdf.PdfReader(str(pdf_path)).pages)
pdf_numeric = [ln.strip() for ln in pdf_text.splitlines()
               if ln.strip() and NUMRUN.match(ln.strip())]
print(f"PDF numeric-only lines     : {len(pdf_numeric)}  (line numbers must survive in the PDF)")

print("\n--- first 8 docx paragraphs ---")
for p in paras[:8]:
    print("   ", p[:84].encode("ascii", "replace").decode())
