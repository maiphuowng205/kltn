# ASEAN V1 result freeze

The ASEAN V1 results are frozen before implementation of the protocol-V2
development work.  V1 used a fixed, availability-aware 100-name universe,
raw five-session excess-return forecasts, a strict 252-session covariance
window, and 2024–2025 as the reported out-of-sample period.

The V1 results remain reproducible evidence, but the reported 2024–2025
outcomes are now **observed development evidence**, not an untouched final
test set.  No V2 configuration may be selected by repeatedly optimizing these
V1 tables.

The local, content-addressed manifest is generated with:

```powershell
python scripts/freeze_asean_results.py `
  --source artifacts/asean_v1_aggregate `
  --output artifacts/asean_v1_aggregate/freeze/asean_v1_pre_v2_manifest.json
```

The canonical V1 handoffs supplied for review are:

- `asean_v1_colab_results-20260818T020147Z-1-001.zip`
- `asean_v1_extension_results-20260818T020307Z-1-001.zip`

The V2 final holdout is intentionally not defined until 2026 data are
available.  If 2026 cannot be obtained, V2 results must be described as
purged walk-forward development evidence rather than a pristine holdout test.
