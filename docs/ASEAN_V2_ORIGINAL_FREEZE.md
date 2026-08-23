# ASEAN V2-original freeze

The forecast-evaluation handoff received on 2026-08-23 is frozen as
`V2-original`. It contains the five-seed development evaluation, not an
untouched final test. The V2.1 ensemble/calibration/cost runs must be kept in
separate output directories and cannot overwrite these files.

Source handoff:

`forecast_evaluation-20260823T055759Z-1-001.zip`

SHA-256:

`DB9CD6E54BD35B98DA45B5FFFB5545A4445BD1F0776FF1991824B5F76181F267`

The content-addressed local manifest is generated with:

```powershell
python scripts/freeze_asean_results.py `
  --source _handoff_inspect/forecast_evaluation_20260823/forecast_evaluation `
  --output artifacts/asean_v1_aggregate/freeze/asean_v2_original_forecast_manifest.json `
  --label ASEAN_V2_ORIGINAL_FORECAST_EVALUATION
```

The V2.1 final evaluation remains development-only until a new, pre-registered
holdout is available.  The previously viewed 2024–2025 period cannot be
called a pristine final test.
