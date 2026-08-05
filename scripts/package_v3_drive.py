"""Package the frozen V3 inputs for manual upload to Google Drive.

The package contains licensed/raw data only in a local ignored archive; it is
never added to Git.  The archive is intentionally independent of LSEG APIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/vn_v3_lseg_2026-08-03_drive.zip")
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    output = args.output.resolve()
    manifest_path = root / "data/lseg_v3/reports/freeze_manifest_v3.json"
    validator = ROOT / "scripts/validate_v3_contract.py"
    subprocess.run([sys.executable, str(validator), "--workspace-root", str(root)], check=True)
    required = [root / "data/lseg_v3", root / "data/external/risk_free_daily.parquet", root / "data/external/risk_free_daily.metadata.json", root / "FROZEN_DATASET_V3.md"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"required package paths missing: {missing}")
    members = []
    for path in sorted((root / "data/lseg_v3").rglob("*")):
        if path.is_file():
            members.append(path)
    members.extend([root / "data/external/risk_free_daily.parquet", root / "data/external/risk_free_daily.metadata.json", root / "FROZEN_DATASET_V3.md"])
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            relative = path.relative_to(root).as_posix()
            archive.write(path, relative)
            records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_id": manifest.get("freeze_id"), "archive": str(output), "archive_bytes": output.stat().st_size, "file_count": len(records), "files": records, "status": "PACKAGE_READY"}
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "freeze_id": report["freeze_id"], "file_count": report["file_count"], "archive": str(output), "sidecar": str(sidecar)}, indent=2))


if __name__ == "__main__":
    main()
