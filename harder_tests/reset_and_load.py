"""Reset ai01 to a clean slate and load the STAND_DER_TECHNIK corpus into the hot folder."""
import shutil
import subprocess
from pathlib import Path

root = Path("/home/install/patent_ocr")
staging = root / "sdt_staging"

print("before:")
for d in ("input", "output", "qc", "failed"):
    p = root / d
    print(f"  {d}: {len(list(p.iterdir())) if p.exists() else 0}")

for d in ("output", "qc", "failed"):
    shutil.rmtree(root / d, ignore_errors=True)
    (root / d).mkdir(parents=True, exist_ok=True)

for f in ("ledger.sqlite3", "ledger.sqlite3-wal", "ledger.sqlite3-shm"):
    (root / "state" / f).unlink(missing_ok=True)
shutil.rmtree(root / "state" / "work", ignore_errors=True)
(root / "state").mkdir(parents=True, exist_ok=True)

inp = root / "input"
inp.mkdir(parents=True, exist_ok=True)
for leftover in inp.iterdir():
    leftover.unlink()

# Copy (never move) so the staging corpus survives the consuming pipeline.
count = 0
for src in sorted(staging.rglob("*")):
    if src.is_file() and src.suffix.lower() == ".pdf":
        shutil.copy2(src, inp / src.name)
        count += 1

print(f"\ncopied {count} pdfs into input/")
print("after:")
for d in ("input", "output", "qc", "failed"):
    p = root / d
    print(f"  {d}: {len(list(p.iterdir()))}")
print("staging intact:", len(list(staging.rglob('*.PDF'))) + len(list(staging.rglob('*.pdf'))))
print("free disk:", subprocess.run(["df", "-h", "/home"], capture_output=True, text=True).stdout.splitlines()[-1])
