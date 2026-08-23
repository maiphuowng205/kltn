"""Create a content-addressed freeze manifest for completed ASEAN results.

The script deliberately does not copy or alter raw LSEG data.  It records a
recursive SHA-256 inventory of a result handoff so that V1 evidence can remain
auditable after development of the V2 protocol begins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Completed V1 result directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the immutable JSON manifest.")
    parser.add_argument("--label", default="ASEAN_V1_PRE_PROTOCOL_V2")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    files = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "freeze_label": args.label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "file_count": len(files),
        "files": files,
        "interpretation": (
            "This is pre-V2 exploratory/out-of-sample evidence. It must not be "
            "used as an untouched final test set for configurations selected after this freeze."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "files": len(files), "source": str(source)}, indent=2))


if __name__ == "__main__":
    main()
