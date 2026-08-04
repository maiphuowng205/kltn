"""Download a reproducible LSEG corporate-action ledger for Vietnam V3."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import lseg.data as ld

ROOT=Path(__file__).resolve().parents[1]
INP=ROOT/'data/lseg_v3/raw/historical_universe_monthly.parquet'
OUT=ROOT/'data/lseg_v3/raw/corporate_actions'
FIELDS=['TR.CACorpActEventType','TR.CACorpActDesc','TR.CATermsOldShares','TR.CATermsNewShares','TR.CAAnnouncementDate','TR.CAExDate','TR.CARecordDate','TR.CAEffectiveDate','TR.CAIsRescinded','TR.CAAdjustmentFactor']

def chunks(x,n):
    for i in range(0,len(x),n): yield x[i:i+n]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rics=sorted(pd.read_parquet(INP,columns=['ric']).ric.dropna().astype(str).str.strip().loc[lambda x:x.ne('')].unique())
    ld.open_session(name='desktop.workspace')
    try:
        for no,batch in enumerate(chunks(rics,50),1):
            path=OUT/f'corporate_actions_batch_{no:02d}.parquet'
            if path.exists():
                print('SKIP',path.name); continue
            last=None
            for attempt in range(4):
                try:
                    df=ld.get_data(universe=batch,fields=FIELDS,parameters={'SDate':'2010-01-01','EDate':'2025-12-31'})
                    df['extract_timestamp_utc']=datetime.now(timezone.utc).isoformat()
                    df.to_parquet(path,index=False); print('WROTE',path.name,'rows',len(df)); break
                except Exception as exc:
                    last=exc; time.sleep(2**attempt)
            else: raise last
        manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'source':'LSEG Workspace Corporate Actions','ric_count':len(rics),'fields':FIELDS,'period':['2010-01-01','2025-12-31'],'partitions':len(list(OUT.glob('corporate_actions_batch_*.parquet')))}
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    finally:
        ld.close_session()
if __name__=='__main__': main()
