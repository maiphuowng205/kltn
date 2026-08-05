"""Copy a completed V3 handoff and exact notebook outputs to persistent storage.

The script is intentionally filesystem-only.  In Colab, Google Drive must be
mounted by the notebook before invoking it; no credentials or network calls are
performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(source: Path, destination: Path) -> list[dict[str, object]]:
    if not source.exists():
        raise FileNotFoundError(source)
    copied: list[dict[str, object]] = []
    for source_file in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = source_file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        copied.append({"path": str(relative).replace("\\", "/"), "bytes": target.stat().st_size, "sha256": sha256_file(target)})
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Completed local runs/v3_final_handoff directory")
    parser.add_argument("--destination", type=Path, required=True, help="Persistent Drive destination")
    parser.add_argument("--notebook-root", type=Path, default=None, help="Optional cloned-repo notebooks directory")
    parser.add_argument("--archive-root", type=Path, default=None, help="Optional destination for exact notebook copies")
    args = parser.parse_args()
    files = copy_tree(args.source, args.destination)
    notebooks: list[dict[str, object]] = []
    if args.notebook_root is not None and args.archive_root is not None:
        notebooks = copy_tree(args.notebook_root, args.archive_root)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "destination": str(args.destination),
        "handoff_files": files,
        "notebook_archive": notebooks,
        "status": "SYNC_COMPLETE",
    }
    (args.destination / "sync_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "handoff_files": len(files), "notebook_files": len(notebooks), "manifest": str(args.destination / 'sync_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
