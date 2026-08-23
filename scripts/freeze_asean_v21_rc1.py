"""Freeze an ASEAN V2.1 release-candidate handoff without overwriting it."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio-root", type=Path, required=True, help="v2_1_portfolio directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forecast-root", type=Path, default=None, help="optional v2_1_ensemble directory")
    parser.add_argument("--data-reports", type=Path, default=None, help="optional V2 reports directory")
    args = parser.parse_args()
    if not args.portfolio_root.exists():
        raise FileNotFoundError(args.portfolio_root)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    shutil.copytree(args.portfolio_root, args.output_dir / "portfolio")
    if args.forecast_root and args.forecast_root.exists():
        shutil.copytree(args.forecast_root, args.output_dir / "forecast")
    if args.data_reports and args.data_reports.exists():
        shutil.copytree(args.data_reports, args.output_dir / "data_reports")

    files = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "freeze_manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(args.output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "status": "ASEAN_V21_RC1_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": "V2.1-RC1",
        "do_not_treat_as_final_holdout": True,
        "known_review_items": [
            "Philippines turnover decomposition and implementation audit",
            "Thailand realistic-cost fragility",
            "V1 replay through the corrected V2.1 portfolio engine",
            "final holdout evaluation after audit completion",
        ],
        "files": files,
    }
    (args.output_dir / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
