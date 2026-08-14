"""Verify the DOCX carries the same text as the PDF, in the same order."""
import re
from pathlib import Path

import pypdf
from docx import Document

result = Path("harder_tests/result")
docx_path = result / "claims_test.docx"
pdf_path = result / "claims_test_SEARCHABLE.pdf"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


doc = Document(str(docx_path))
paras = [p.text for p in doc.paragraphs if p.text.strip()]
docx_text = norm(" ".join(paras))
# The title heading is added by the exporter and has no PDF counterpart.
body_text = norm(" ".join(paras[1:]))

pdf_text = norm(" ".join((p.extract_text() or "") for p in pypdf.PdfReader(str(pdf_path)).pages))

print(f"docx paragraphs : {len(paras)}")
print(f"docx chars      : {len(docx_text)}  (body only: {len(body_text)})")
print(f"pdf  chars      : {len(pdf_text)}")
print(f"body identical  : {body_text == pdf_text}")
from collections import Counter

docx_tokens = Counter(body_text.split())
pdf_tokens = Counter(pdf_text.split())
print(f"token multiset identical : {docx_tokens == pdf_tokens}")
only_docx = docx_tokens - pdf_tokens
only_pdf = pdf_tokens - docx_tokens
if only_docx or only_pdf:
    print(f"  only in docx: {dict(list(only_docx.items())[:8])}")
    print(f"  only in pdf : {dict(list(only_pdf.items())[:8])}")
if body_text != pdf_text:
    for i, (a, b) in enumerate(zip(body_text, pdf_text)):
        if a != b:
            print(f"  first char diff at {i}: docx={a!r} pdf={b!r}")
            break

for target in ("5", "10", "15", "20", "25", "30", "35"):
    print(f"  margin {target:>2} in docx: {bool(re.search(rf'(?<!\d){target}(?!\d)', docx_text))}")

print("\n--- first 8 docx paragraphs ---")
for p in paras[:8]:
    print("   ", p[:88].encode("ascii", "replace").decode())
