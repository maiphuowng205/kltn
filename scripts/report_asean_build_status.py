"""Print resumable ASEAN extraction progress and field coverage."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / "artifacts/asean_v1/raw/daily_history"
expected = {"Singapore": 4, "Malaysia": 6, "Indonesia": 5, "Thailand": 5, "Philippines": 2}
files = sorted(raw.glob("daily_*.parquet"))
rows = []
for p in files:
    try:
        x = pd.read_parquet(p, columns=["country", "close", "volume", "bid", "ask", "total_return_pct"])
        country = str(x["country"].dropna().iloc[0]) if len(x) else p.name
        rows.append({"file": p.name, "country": country, "rows": len(x), "close": x["close"].notna().mean() if len(x) else 0, "volume": x["volume"].notna().mean() if len(x) else 0, "bid": x["bid"].notna().mean() if len(x) else 0, "ask": x["ask"].notna().mean() if len(x) else 0, "total_return": x["total_return_pct"].notna().mean() if len(x) else 0})
    except Exception as exc:
        rows.append({"file": p.name, "error": str(exc)})
df = pd.DataFrame(rows)
counts = df.groupby("country").size().to_dict() if len(df) else {}
summary = {"daily_files": len(files), "expected_files": sum(expected.values()) * 10, "progress_pct": round(100 * len(files) / (sum(expected.values()) * 10), 2), "files_by_country": counts, "marketcap_files": len(list((ROOT / "artifacts/asean_v1/raw/market_cap_monthly").glob("market_cap_*.parquet"))), "rf_exists": (ROOT / "artifacts/asean_v1/raw/risk_free_daily.parquet").exists()}
print(json.dumps(summary, indent=2, default=float))
if len(df):
    print(df.groupby("country")[["rows", "close", "volume", "bid", "ask", "total_return"]].mean().round(4).to_string())
