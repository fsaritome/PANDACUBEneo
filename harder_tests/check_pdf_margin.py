"""Which margin numbers actually made it into the delivered PDF's text layer?"""
from pathlib import Path

import pypdf

pdf = Path(r"harder_tests/result/claims_test_SEARCHABLE.pdf")
reader = pypdf.PdfReader(str(pdf))

page = reader.pages[0]
runs = []
page.extract_text(
    visitor_text=lambda t, cm, tm, fd, fs: runs.append((t.strip(), round(tm[4]), round(tm[5])))
    if t.strip() else None
)

print(f"page 1: {len(runs)} text runs")
print("\n--- runs in the left margin (x < 60pt) ---")
for text, x, y in runs:
    if x < 60 and len(text) < 12:
        print(f"  x={x:4} y={y:4}  {text!r}")

print("\n--- every run that is purely a number ---")
for text, x, y in runs:
    if text.isdigit():
        print(f"  x={x:4} y={y:4}  {text!r}")

full = page.extract_text() or ""
print("\n--- first 6 lines of extracted text ---")
for line in full.splitlines()[:6]:
    print("   ", line.encode("ascii", "replace").decode())

for target in ("5", "10", "15", "20", "25", "30", "35"):
    present = any(t == target for t, _, _ in runs)
    print(f"  standalone {target!r} present as its own run: {present}")
