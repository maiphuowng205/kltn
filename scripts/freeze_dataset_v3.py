"""Write the immutable inventory/checksums for the current Vietnam V3 dataset."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/lseg_v3'
RF=ROOT/'data/external/risk_free_daily.parquet'

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    files=sorted(p for p in DATA.rglob('*') if p.is_file() and p.name!='freeze_manifest_v3.json')+[RF]
    items=[]
    for p in files:
        items.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)})
    report={'freeze_id':'vn_v3_lseg_2026-08-03','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_ACTIVE','scope':'Vietnam-only main study; US dataset excluded','claim_policy':'Not survivorship-bias-free; no complete delisting ledger; corporate-action semantics and historical bid/ask remain limited.','file_count':len(items),'files':items}
    out=DATA/'reports/freeze_manifest_v3.json'; out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'freeze_id':report['freeze_id'],'file_count':len(items),'manifest':str(out)},indent=2))
if __name__=='__main__': main()
