"""Assemble the P2 extension artifacts without mutating the frozen V3 handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    parser.add_argument("--extension-root", type=Path, default=ROOT / "runs/v3_extension_p2")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/v3_p2_handoff")
    args = parser.parse_args()
    extension_root = args.extension_root.resolve()
    output = args.output.resolve()
    if not extension_root.exists():
        raise FileNotFoundError(extension_root)
    output.mkdir(parents=True, exist_ok=True)

    copied = []
    for source in sorted(extension_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(extension_root)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({
            "path": relative.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })

    freeze_manifest = ROOT / "data/lseg_v3/reports/freeze_manifest_v3.json"
    freeze_id = None
    if freeze_manifest.exists():
        freeze_id = json.loads(freeze_manifest.read_text(encoding="utf-8")).get("freeze_id")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCAL_EXTENSION_HANDOFF_READY",
        "extension": "V3 P2 quarterly walk-forward and transaction-cost availability audit",
        "freeze_id": freeze_id,
        "source": str(extension_root),
        "output": str(output),
        "files": copied,
        "final_handoff_mutated": False,
        "cost_validation_status": "BLOCKED_NO_OBSERVED_QUOTES",
    }
    (output / "handoff_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "file_count": len(copied), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
