"""Normalise LSEG Vietnam corporate actions and audit daily price discontinuities."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/lseg_v3'

def main():
    raw=DATA/'raw/corporate_actions'
    files=sorted(raw.glob('corporate_actions_batch_*.parquet'))
    if len(files)!=19: raise RuntimeError(f'Expected 19 action partitions, found {len(files)}')
    x=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    rename={'Instrument':'ric','Corporate Change Event Type':'event_type','Corporate Action Description':'event_description','Terms Old Shares':'terms_old_shares','Terms New Shares':'terms_new_shares','Capital Change Announcement Date':'announcement_date','Capital Change Ex Date':'ex_date','Capital Change Record Date':'record_date','Capital Change Effective Date':'effective_date','Capital Change Is Rescinded':'is_rescinded','Adjustment Factor':'adjustment_factor'}
    x=x.rename(columns=rename)
    x['ric']=x.ric.fillna('').astype(str).str.strip(); x['event_type']=x.event_type.fillna('').astype(str).str.strip()
    ledger=x.loc[x.ric.ne('') & x.event_type.ne('')].copy()
    for c in ['announcement_date','ex_date','record_date','effective_date']:
        ledger[c]=pd.to_datetime(ledger[c],errors='coerce').dt.normalize()
    for c in ['terms_old_shares','terms_new_shares','adjustment_factor']:
        ledger[c]=pd.to_numeric(ledger[c],errors='coerce')
    ledger['is_rescinded']=ledger['is_rescinded'].fillna(False).astype(bool)
    key=ledger[['ric','event_type','event_description','announcement_date','ex_date','record_date','effective_date','terms_old_shares','terms_new_shares','adjustment_factor']].astype('string').fillna('').agg('|'.join,axis=1)
    ledger['event_id']=key.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:20])
    ledger=ledger.drop_duplicates('event_id').sort_values(['ric','effective_date','ex_date','announcement_date','event_type'])
    curated=DATA/'curated'; reports=DATA/'reports'; curated.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    ledger.to_parquet(curated/'corporate_action_ledger.parquet',index=False)

    # Daily data is already deduplicated/cleaned by the V3 builder.  Identify
    # large one-session close moves and link them only to known event dates;
    # unmatched jumps remain review items, not inferred corporate actions.
    daily=pd.read_parquet(curated/'daily_panel.parquet',columns=['ric','date','close','total_return_pct'])
    daily=daily.sort_values(['ric','date']).drop_duplicates(['ric','date'])
    daily['close_return_1d']=daily.groupby('ric')['close'].pct_change(fill_method=None)
    daily['previous_date']=daily.groupby('ric')['date'].shift(1)
    jumps=daily.loc[daily.close_return_1d.abs().ge(.15),['ric','date','previous_date','close','close_return_1d','total_return_pct']].copy()
    event_dates=ledger.melt(id_vars=['event_id','ric','event_type','event_description','is_rescinded','adjustment_factor'],value_vars=['ex_date','effective_date'],var_name='event_date_kind',value_name='event_date').dropna(subset=['event_date'])
    candidates=jumps.merge(event_dates,on='ric',how='left')
    candidates['calendar_day_gap']=(candidates.date-candidates.event_date).dt.days.abs()
    candidates=candidates.loc[candidates.calendar_day_gap.le(3)].sort_values(['ric','date','calendar_day_gap','event_id'])
    matches=candidates.drop_duplicates(['ric','date'],keep='first')
    audit=jumps.merge(matches[['ric','date','event_id','event_type','event_description','event_date_kind','event_date','calendar_day_gap','is_rescinded','adjustment_factor']],on=['ric','date'],how='left')
    audit['event_match_status']=np.where(audit.event_id.notna(),'matched_within_3_calendar_days','unmatched_review_required')
    audit.to_parquet(reports/'corporate_action_price_jump_audit.parquet',index=False)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'ledger_rows':int(len(ledger)),'rics_with_events':int(ledger.ric.nunique()),'event_type_counts':{str(k):int(v) for k,v in ledger.event_type.value_counts().items()},'rescinded_events':int(ledger.is_rescinded.sum()),'price_jumps_abs_15pct':int(len(audit)),'matched_price_jumps':int(audit.event_id.notna().sum()),'unmatched_price_jumps':int(audit.event_id.isna().sum()),'match_rule':'same RIC and ex/effective date within 3 calendar days; unmatched events are not inferred'}
    (reports/'corporate_action_audit_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
