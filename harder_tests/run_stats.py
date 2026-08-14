"""Aggregate stats for the sidecars written by today's run only."""
import json
import time
from pathlib import Path

CUTOFF = time.mktime(time.strptime("2026-08-14 10:50", "%Y-%m-%d %H:%M"))
qc = Path("/home/install/patent_ocr/qc")

confs, pages, flagged, layouts, engines = [], 0, 0, {}, {}
docs = 0
for f in qc.glob("*.json"):
    if f.stat().st_mtime < CUTOFF:
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    docs += 1
    if d.get("mean_confidence") is not None:
        confs.append(d["mean_confidence"])
    pages += len(d.get("pages", []))
    flagged += 1 if d.get("flagged") else 0
    for lt in d.get("layout_types", []):
        layouts[lt] = layouts.get(lt, 0) + 1
    for e in d.get("engines_used", []):
        engines[e] = engines.get(e, 0) + 1

print(f"documents        : {docs}")
print(f"pages            : {pages}")
print(f"mean confidence  : {sum(confs)/len(confs):.2f}" if confs else "n/a")
print(f"min doc conf     : {min(confs):.2f}" if confs else "")
print(f"flagged docs     : {flagged}")
print(f"engines          : {engines}")
print(f"layout types     : {layouts}")
