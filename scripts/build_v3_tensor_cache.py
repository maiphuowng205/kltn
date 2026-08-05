"""Build the shared V3 tensor cache with train-only preprocessing provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v3_method import FEATURES, build_tensor_bundle


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/lseg_v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/v3_tensor_cache")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.data_root / "reports" / "freeze_manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, median, scale = build_tensor_bundle(args.data_root, "train")
    validation, _, _ = build_tensor_bundle(args.data_root, "validation", median, scale)
    test, _, _ = build_tensor_bundle(args.data_root, "test", median, scale)
    bundles = {"train": train, "validation": validation, "test": test}
    files: dict[str, dict[str, object]] = {}
    for name, bundle in bundles.items():
        assert bundle.x.shape[1:] == (100, 60, len(FEATURES))
        assert np.isfinite(bundle.x).all()
        target = args.output_dir / f"{name}.npz"
        np.savez_compressed(target, dates=bundle.dates, rics=bundle.rics, x=bundle.x, y=bundle.y, mask=bundle.mask, execution_dates=bundle.execution_dates)
        files[target.name] = {
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "shape": list(bundle.x.shape),
            "dates": int(len(bundle.dates)),
            "target_available_fraction": float(bundle.mask.mean()),
        }
    preprocessing = {"features": FEATURES, "lookback_sessions": 60, "median": median.tolist(), "iqr": scale.tolist(), "clip": [-10.0, 10.0], "fit_split": "train"}
    (args.output_dir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_id": manifest.get("freeze_id"),
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "data_root": str(args.data_root),
        "features": FEATURES,
        "lookback_sessions": 60,
        "assets_per_date": 100,
        "preprocessing_fit_split": "train",
        "files": files,
        "status": "PASS",
    }
    (args.output_dir / "cache_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
