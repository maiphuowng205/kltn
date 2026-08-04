"""Validate the immutable V3 freeze manifest and core row contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    manifest_path = (args.manifest or root / "data/lseg_v3/reports/freeze_manifest_v3.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        relative = Path(item["path"].replace("\\", "/"))
        path = root / relative
        if not path.exists():
            failures.append({"path": str(relative), "reason": "missing"})
            continue
        if path.stat().st_size != item["bytes"]:
            failures.append({"path": str(relative), "reason": "byte_count_mismatch"})
            continue
        if digest(path) != item["sha256"]:
            failures.append({"path": str(relative), "reason": "sha256_mismatch"})

    weekly = pd.read_parquet(root / "data/lseg_v3/curated/universe_weekly.parquet")
    model = pd.read_parquet(root / "data/lseg_v3/model_ready/weekly_features_targets.parquet")
    rows_by_date = weekly.groupby("date").size()
    contract = {
        "freeze_id": manifest.get("freeze_id"),
        "file_count_expected": int(manifest["file_count"]),
        "file_count_checked": len(manifest["files"]),
        "checksum_failures": failures,
        "weekly_rows": int(len(weekly)),
        "model_ready_rows": int(len(model)),
        "dates": int(rows_by_date.size),
        "assets_per_date_ok": bool(rows_by_date.eq(100).all()),
        "status": "PASS" if not failures and rows_by_date.eq(100).all() else "FAIL",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))
    if contract["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
