# Dataset Freeze V3 — Vietnam main study

Status: **FROZEN / ACTIVE**

Scope: HOSE/HNX historical monthly universe, LSEG daily OHLCV/total-return data,
LSEG corporate-action ledger, VND1MD= daily cash-proxy RF, V3 curated panel and
weekly model-ready dataset.

The immutable file inventory and SHA-256 checksums are in
`data/lseg_v3/reports/freeze_manifest_v3.json`.

Allowed interpretation: relative forecast and portfolio performance within the
frozen Vietnam panel, using stated cost sensitivity.

Not allowed: claims that the panel is survivorship-bias-free, has a complete
delisting/listing ledger, has fully verified corporate-action adjustment
semantics, or uses observed historical bid/ask transaction costs.

Any data or protocol revision creates V4; it must not overwrite V3.
