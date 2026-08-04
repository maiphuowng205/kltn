"""Compare byte-level artifacts from two deterministic run directories."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--main-run',type=Path,required=True); p.add_argument('--repeat-run',type=Path,required=True); p.add_argument('--files',nargs='+',required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args(); report={}
    for name in a.files:
        left=a.main_run/name; right=a.repeat_run/name
        if not left.exists() or not right.exists(): report[name]={'identical':False,'reason':'missing'}; continue
        l=digest(left); r=digest(right); report[name]={'main':l,'repeat':r,'identical':l==r}
    output={'status':'PASS' if all(v.get('identical') for v in report.values()) else 'FAIL','files':report}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(output,indent=2),encoding='utf-8'); print(json.dumps(output,indent=2));
    if output['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
