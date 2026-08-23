# Tổng hợp toàn bộ phương pháp và kết quả thực nghiệm ASEAN

**Ngày tổng hợp:** 23/08/2026  
**Phạm vi:** ASEAN-5 gồm Indonesia, Malaysia, Philippines, Singapore và Thailand.  
**Bản kết quả được ưu tiên khi kết luận:** `V2.1-RC3 final audit`. Các bảng V1 và V2.1-RC2 được giữ lại để mô tả tiến trình và làm benchmark, không được trộn trực tiếp với RC3 vì engine và protocol khác nhau.

## 1. Kết luận ngắn gọn

Audit RC3 cho thấy pipeline chạy được và có tính tái lập tốt, nhưng **chưa có bằng chứng rằng forecast PTCST tạo thêm giá trị đầu tư ổn định so với risk-only allocation**.

- 5 seed được đối chiếu trên 374.000 quan sát: khớp 100%, sai khác dự báo lớn nhất bằng 0.
- Universe giữa các seed dùng trong ensemble có intersection ratio bằng 1, tức các forecast được ghép trên cùng universe.
- Ensemble được chọn hoàn toàn trên validation là `E3 – IC-weighted`.
- Calibration chỉ giữ tín hiệu dương ở Philippines và Thailand. Indonesia, Malaysia và Singapore có calibration beta âm nên bị đưa về `mu = 0`, tức danh mục thực tế là risk-only.
- Ở Philippines và Thailand, PTCST có Sharpe thấp hơn risk-only trong cả C0, C1 và C2.
- Vì vậy kết luận hợp lệ là **negative incremental alpha finding**: risk/cost-aware portfolio engine hoạt động, nhưng forecast Transformer chưa chứng minh được giá trị gia tăng bền vững.

Không nên viết rằng PTCST “đánh bại các benchmark” hoặc “cải thiện Sharpe ở cả ASEAN”.

## 2. Nguồn kết quả và thứ tự ưu tiên

| Gói | Nội dung | Vai trò |
|---|---|---|
| `asean_v1_colab_results-20260818...zip` | V1 availability-aware Top-100, forecast baselines, Temporal Transformer, PatchTST, PTCST, MVO và cost audit | Benchmark lịch sử |
| `asean_v1_extension_results-20260818...zip` | Tổng hợp BID/ASK cuối ngày và observed-cost sensitivity V1 | Benchmark chi phí |
| `asean_v2_development-20260823T052605...zip` | 5 seed PTCST-v2, training history, validation/development predictions | Training/development evidence |
| `forecast_evaluation-20260823T055759...zip` | Forecast evaluation của ensemble V2.1 trước RC3 | Kết quả trung gian |
| `v2_1_portfolio-20260823T093201...zip` | V2.1-RC2 portfolio, lambda selection, C0/C1/C2 | Kết quả trung gian |
| `asean_v2-20260823T110436...zip` + `asean_v2_development-20260823T110810...zip` | Dataset V2 và 5 seed runs dùng cho audit RC3 | Nguồn RC3 |
| `D:\kltn\_rc3_local\audit2` | Kết quả audit RC3 mới nhất | **Nguồn kết luận cuối** |

Gói kết quả audit RC3 đầy đủ: [v2_1_rc3_final_audit.zip](D:\kltn\_rc3_local\v2_1_rc3_final_audit.zip).

## 3. Dữ liệu và protocol

### 3.1 Thị trường và universe

Nghiên cứu dùng 5 thị trường ASEAN: Indonesia, Malaysia, Philippines, Singapore và Thailand.

V1 dùng universe **availability-aware Top-100**. V2 chuyển sang **pure lagged market-cap Top-100 với variable-N**: cổ phiếu không đủ dữ liệu bị loại, không kéo mã đứng hạng 101–300 vào để lấp chỗ. Vì vậy số mã thực tế mỗi ngày nhỏ hơn 100.

### 3.2 Timing và target V2

Protocol V2 được ghi trong `protocol.json` và dataset report:

```text
signal: close t
execution: close t+1
target/P&L: close t+2 đến close t+6
```

Target là **cross-sectional demeaned five-session excess return**, tức lợi suất 5 phiên sau execution, đã loại thành phần trung bình chéo của các cổ phiếu trong cùng thị trường/ngày.

Split:

- Train: 2019–2022.
- Validation: 2023.
- Development evidence: 2024–2025.
- Chưa có holdout 2026 chưa nhìn thấy. Do đó 2024–2025 không nên gọi là final untouched test.

Purging: 657 dòng có thể chạm boundary đã được loại khỏi split; audit V2 ghi nhận không còn leakage được giữ lại.

### 3.3 Dataset V2 coverage

| Country | Weekly dates | Mean assets/date | Min assets/date | Target coverage |
|---|---:|---:|---:|---:|
| Indonesia | 489 | 74.74 | 57 | 98.75% |
| Malaysia | 496 | 82.07 | 67 | 98.96% |
| Philippines | 411 | 70.96 | 62 | 97.78% |
| Singapore | 470 | 75.03 | 1 | 98.58% |
| Thailand | 495 | 85.67 | 80 | 99.10% |

Risk eligibility yêu cầu tối thiểu 126 phiên lịch sử; covariance dùng tối đa 252 phiên.

## 4. Các mô hình và portfolio strategy

### 4.1 Forecast models trong V1

- `Zero`: dự báo bằng 0; không có ranking hợp lệ.
- `Historical Mean`: trung bình lịch sử.
- `Ridge`: linear regression có regularization L2.
- `XGBoost`: gradient-boosted trees.
- `Temporal Transformer`: Transformer theo chuỗi thời gian.
- `PatchTST`: chia chuỗi thời gian thành các patch rồi đưa vào Transformer.
- `PTCST`: Patch/Temporal/Cross-Sectional Transformer, kết hợp temporal modeling và cross-sectional attention.

### 4.2 Portfolio engine

Sau forecast, pipeline dùng covariance Ledoit–Wolf và constrained mean–variance optimization (MVO). Các thành phần chính:

- Risk aversion được chọn trên validation; V2.1 chọn `lambda = 50`.
- Turnover cap: `0.40` theo full L1 turnover.
- Initial deployment được xử lý riêng, không còn bị ghi nhận là solver failure.
- Cost scenarios:
  - `C0`: chi phí cố định 10 bps.
  - `C1`: median observed country half-spread.
  - `C2`: security-specific lagged half-spread.
- `risk-only`: cùng covariance, constraints và cost nhưng `mu = 0`; đây là benchmark bắt buộc để đo incremental alpha của forecast.

## 5. Kết quả V1: forecast baselines

Spearman IC là mức liên hệ thứ hạng giữa dự báo và excess return thực tế. V1 test IC của các baseline:

| Country | Historical Mean | Ridge | XGBoost | Zero |
|---|---:|---:|---:|---:|
| Indonesia | 0.0312 | 0.0078 | 0.0134 | N/A |
| Malaysia | 0.0421 | 0.0314 | -0.0105 | N/A |
| Philippines | 0.0532 | 0.0602 | 0.0519 | N/A |
| Singapore | 0.0261 | 0.0088 | -0.0055 | N/A |
| Thailand | 0.0204 | -0.0039 | -0.0019 | N/A |

Các IC này nhìn chung nhỏ. Zero có MAE/RMSE đôi khi thấp vì target thường gần 0, nhưng Zero không tạo ranking nên không được coi là có stock-selection skill.

## 6. Kết quả V1: deep model và PTCST ablation

### 6.1 Net annualized Sharpe của deep models

| Country | Temporal Transformer | PatchTST | PTCST |
|---|---:|---:|---:|
| Indonesia | -0.030 | -0.138 | -0.013 |
| Malaysia | 0.475 | 0.668 | 0.619 |
| Philippines | N/A | N/A | N/A |
| Singapore | 0.703 | 0.690 | 0.685 |
| Thailand | -0.999 | -1.041 | -1.073 |

Philippines khi đó không có đủ risk coverage để tạo portfolio hợp lệ. Đây là một trong các lý do V2 phải sửa risk eligibility.

### 6.2 PTCST portfolio ablation V1

| Country | PTCST-Top20 Sharpe / turnover | PTCST-MVO Sharpe / turnover | PTCST-CA-MVO Sharpe / turnover |
|---|---:|---:|---:|
| Indonesia | 0.409 / 74.89% | -0.617 / 26.07% | -0.013 / 0.94% |
| Malaysia | 0.024 / 126.57% | 1.232 / 35.38% | 0.619 / 0.93% |
| Philippines | -0.109 / 93.84% | N/A / 0% | N/A / 0% |
| Singapore | 0.305 / 63.25% | 0.742 / 21.11% | 0.685 / 0.76% |
| Thailand | -0.778 / 53.46% | -0.907 / 19.33% | -1.073 / 0.55% |

Điểm nổi bật của V1 là cost-aware MVO giảm turnover rất mạnh, nhưng không đồng nghĩa forecast tốt hơn.

## 7. V1: chi phí và risk coverage

### 7.1 Observed BID/ASK cuối ngày

| Country | Median half-spread | P90 median half-spread | Mean quote coverage | Dates half-spread >10 bps |
|---|---:|---:|---:|---:|
| Indonesia | 21.88 bps | 25.16 bps | 98.81% | 99.80% |
| Malaysia | 26.12 bps | 29.86 bps | 83.63% | 83.73% |
| Philippines | 26.00 bps | 33.82 bps | 88.38% | 88.50% |
| Singapore | 31.95 bps | 35.34 bps | 82.95% | 83.10% |
| Thailand | 29.67 bps | 32.37 bps | 97.96% | 98.04% |

BID/ASK là dữ liệu cuối ngày, không phải tick-level và không đo implementation shortfall trong từng giao dịch.

### 7.2 V1 covariance fallback

| Country | Fallback dates | Fallback fraction |
|---|---:|---:|
| Indonesia | 0/259 | 0% |
| Malaysia | 0/261 | 0% |
| Philippines | 81/212 | 38.21% |
| Singapore | 40/242 | 16.53% |
| Thailand | 0/261 | 0% |

Đây là vấn đề đã được xử lý trong V2 bằng risk history tối thiểu 126 phiên và full risk eligibility.

## 8. V2 development: 5 seed PTCST

Năm seed được train trên cùng protocol:

| Seed | Best validation IC | Best epoch | Development IC | Calibration beta |
|---:|---:|---:|---:|---:|
| 7 | 0.0472 | 8 | 0.0351 | 0.001200 |
| 19 | 0.0445 | 8 | 0.0234 | 0.000993 |
| 31 | 0.0397 | 7 | 0.0417 | 0.000775 |
| 43 | 0.0388 | 10 | 0.0157 | 0.000942 |
| 59 | 0.0397 | 8 | 0.0249 | 0.000947 |

Kết quả này chỉ là seed-level development evidence; không được dùng để chọn “best seed” theo development/test.

## 9. V2.1-RC2: kết quả trung gian trước audit RC3

### 9.1 Ensemble mean-rank trước RC3

| Country | Mean IC | 95% CI thấp | 95% CI cao |
|---|---:|---:|---:|
| Indonesia | -0.0073 | -0.0297 | 0.0150 |
| Malaysia | 0.0046 | -0.0184 | 0.0279 |
| Philippines | 0.0017 | -0.0229 | 0.0272 |
| Singapore | 0.0144 | -0.0077 | 0.0382 |
| Thailand | 0.0108 | -0.0134 | 0.0343 |

Đây là lý do phải audit ensemble: kết quả mean-rank làm mất phần lớn tín hiệu seed-level ở Philippines.

### 9.2 Lambda selection trên validation

| Lambda | Mean-country validation Sharpe |
|---:|---:|
| 2 | 0.873 |
| 5 | 0.878 |
| 10 | 0.906 |
| 20 | 0.924 |
| **50** | **1.003** |

Lambda 50 được giữ vì được chọn trên validation, không phải vì nhìn test rồi tune.

### 9.3 V2.1-RC2 cost sensitivity

| Country | C0 Sharpe | C1 Sharpe | C2 Sharpe | Nhận xét |
|---|---:|---:|---:|---|
| Indonesia | 1.397 | 1.336 | 0.830 | Tương đối robust |
| Malaysia | 1.644 | 1.571 | 1.843 | Robust |
| Philippines | 0.518 | 0.432 | 0.633 | Turnover rất cao |
| Singapore | 2.403 | 2.359 | 2.106 | Mạnh nhất |
| Thailand | 0.168 | -0.081 | -0.166 | Nhạy với chi phí |

Các số RC2 này gần RC3 ở những nước risk-only, nhưng RC3 là bản dùng để kết luận vì đã bổ sung ensemble audit và attribution.

## 10. Audit RC3: kiểm tra ensemble cuối

### 10.1 Integrity và alignment

- Forecast input reconciliation: `374.000/374.000` dòng khớp.
- Maximum absolute prediction difference: `0`.
- Common-universe intersection ratio: `1.0` ở validation và development.
- Pairwise rank correlation giữa seed: khoảng `0.50–0.66`, nghĩa là các seed có mức đồng thuận vừa phải, không hoàn toàn giống nhau.
- Seed list: `7, 19, 31, 43, 59`.

Lưu ý: hai ZIP được cung cấp không chứa một bản “original forecast” độc lập riêng; vì vậy reconciliation hiện là self-reconciliation trong cùng run root, không phải so sánh độc lập giữa hai package khác nhau.

### 10.2 So sánh bốn ensemble candidate trên validation

| Candidate | Mean IC | Median IC | ICIR | Hit rate | Top-bottom |
|---|---:|---:|---:|---:|---:|
| E1 mean rank | 0.0082 | 0.0123 | 0.043 | 52.58% | 1.20 bps |
| E2 median rank | 0.0123 | 0.0143 | 0.077 | 52.58% | 6.86 bps |
| **E3 IC-weighted** | **0.0149** | **0.0223** | **0.113** | **52.96%** | 3.17 bps |
| E4 TB-weighted | 0.0120 | 0.0089 | 0.125 | 53.37% | 11.07 bps |

E3 được chọn vì vượt E1 ít nhất 0.005 Mean IC và thỏa điều kiện validation. E4 có trọng số seed tối đa 71.6%, vượt giới hạn 50%, nên không được chọn.

### 10.3 Trọng số E3 và calibration beta

Trọng số E3, fit trên validation:

```text
seed 7:  13.08%
seed 19:  0.00%
seed 31: 45.25%
seed 43: 23.22%
seed 59: 18.44%
```

Calibration beta theo country:

| Country | Raw beta | Positive beta dùng trong portfolio | Có alpha dương? |
|---|---:|---:|---|
| Indonesia | -0.000827 | 0 | Không |
| Malaysia | -0.000497 | 0 | Không |
| Philippines | 0.002213 | 0.002213 | Có |
| Singapore | -0.000206 | 0 | Không |
| Thailand | 0.000494 | 0.000494 | Có |

Quy tắc `beta_positive = max(0, beta)` là rule đã khóa trước, không flip tín hiệu âm để làm kết quả đẹp hơn.

## 11. RC3 portfolio: PTCST so với risk-only

Đây là bảng quan trọng nhất vì tách giá trị của forecast khỏi giá trị của covariance/optimizer. Sharpe annualize theo daily state engine.

### 11.1 C0 – chi phí cố định 10 bps

| Country | Strategy | Net return | Volatility | Sharpe | Max drawdown | Mean turnover/rebalance |
|---|---|---:|---:|---:|---:|---:|
| Indonesia | PTCST | 14.12% | 10.11% | 1.397 | -17.09% | 3.40% |
| Indonesia | Risk-only | 14.12% | 10.11% | 1.397 | -17.09% | 3.40% |
| Malaysia | PTCST | 12.54% | 7.63% | 1.644 | -9.39% | 1.81% |
| Malaysia | Risk-only | 12.54% | 7.63% | 1.644 | -9.39% | 1.81% |
| Philippines | PTCST | 6.71% | 9.91% | 0.677 | -10.25% | **39.91%** |
| Philippines | Risk-only | 11.02% | 7.08% | **1.557** | -8.67% | 1.63% |
| Singapore | PTCST | 19.44% | 8.09% | 2.403 | -9.92% | 1.88% |
| Singapore | Risk-only | 19.44% | 8.09% | 2.403 | -9.92% | 1.88% |
| Thailand | PTCST | 1.29% | 9.27% | 0.140 | -16.78% | 5.43% |
| Thailand | Risk-only | 2.39% | 9.37% | **0.255** | -16.09% | 2.17% |

### 11.2 Incremental Sharpe của PTCST so với risk-only

| Country | C0 ΔSharpe | C1 ΔSharpe | C2 ΔSharpe |
|---|---:|---:|---:|
| Indonesia | 0.000 | 0.000 | 0.000 |
| Malaysia | 0.000 | 0.000 | 0.000 |
| Philippines | -0.879 | -1.022 | -0.858 |
| Singapore | 0.000 | 0.000 | 0.000 |
| Thailand | -0.116 | -0.150 | -0.173 |

Kết quả C1 và C2 vẫn giữ cùng pattern: PTCST không cải thiện 3 nước có beta âm, và làm giảm Sharpe ở Philippines/Thailand.

### 11.3 Bootstrap incremental return và Sharpe

Bootstrap dùng block thời gian dài 5, 2.000 draws.

- Philippines: mean delta return âm trong cả C0/C1/C2; C1 có khoảng tin cậy delta Sharpe hoàn toàn âm (`-1.985` đến `-0.108`).
- Thailand: mean delta return âm; khoảng tin cậy rộng và thường chứa 0, nên không khẳng định khác biệt có ý nghĩa thống kê mạnh.
- Indonesia, Malaysia và Singapore: PTCST và risk-only giống hệt vì alpha đã bị clip về 0.

### 11.4 Cost scenarios RC3 cho PTCST

| Country | C0 Sharpe | C1 Sharpe | C2 Sharpe | Diễn giải |
|---|---:|---:|---:|---|
| Indonesia | 1.397 | 1.336 | 0.830 | Dương nhưng giảm khi cost thực tế hơn |
| Malaysia | 1.644 | 1.571 | 1.843 | Dương và khá robust |
| Philippines | 0.677 | 0.469 | 0.845 | Alpha không bù được turnover |
| Singapore | 2.403 | 2.359 | 2.106 | Kết quả portfolio tốt nhất |
| Thailand | 0.140 | -0.080 | -0.165 | Fragile dưới realistic cost |

### 11.5 Reliability RC3

Ở các run RC3:

- Evaluation coverage: 100%.
- Covariance fallback: 0%.
- Solver fallback: 0%.
- Missing valuation tính theo asset-day fraction khoảng 0.08%–0.34% tùy nước, không phải tỷ lệ số ngày bị thiếu.

## 12. Diễn giải kinh tế

### Indonesia, Malaysia và Singapore

Ba nước có portfolio Sharpe tốt, nhưng trong RC3 điều này đến từ covariance, diversification, constraints và optimizer vì forecast alpha bị loại bởi beta âm. Do đó chỉ được claim:

> Risk-aware allocation hoạt động tốt trên các thị trường này trong sample development.

Không được claim PTCST forecast tạo ra Sharpe đó.

### Philippines

Philippines có forecast signal validation dương mạnh nhất trong E3, nhưng portfolio turnover trung bình gần chạm cap 40%. PTCST có Sharpe thấp hơn risk-only và cost drag cao. Đây là ví dụ cho thấy forecast ranking tốt chưa chắc chuyển thành portfolio tốt nếu magnitude calibration và transaction cost không phù hợp.

### Thailand

Thailand có beta dương nhỏ, nhưng PTCST vẫn kém risk-only và trở nên âm dưới C1/C2. Đây là kết quả economically fragile, không nên gọi là thành công.

## 13. Kết luận cho luận văn

Kết luận an toàn và đúng với toàn bộ evidence:

> Nghiên cứu xây dựng và kiểm tra một pipeline tích hợp forecast lợi suất, risk estimation bằng Ledoit–Wolf covariance và cost-aware constrained portfolio optimization trên 5 thị trường ASEAN. Các kiểm tra RC3 cho thấy pipeline có alignment, coverage và reproducibility tốt. Tuy nhiên, sau khi lựa chọn ensemble trên validation và calibration beta không âm, PTCST chưa tạo ra incremental portfolio value ổn định so với risk-only allocation. Ở Indonesia, Malaysia và Singapore, tín hiệu bị loại do calibration beta âm; ở Philippines và Thailand, việc sử dụng alpha làm giảm Sharpe và tăng turnover. Vì vậy đóng góp được hỗ trợ mạnh nhất là protocol thực nghiệm và portfolio engine có kiểm soát rủi ro/chi phí, không phải claim rằng PTCST là mô hình dự báo vượt trội.

## 14. Hạn chế cần ghi rõ

1. 2024–2025 đã được dùng làm development evidence; chưa có 2026 untouched holdout.
2. Reconciliation RC3 hiện là self-reconciliation vì chưa có bản original 5-seed package độc lập.
3. BID/ASK là EOD, không phải tick-level; chưa đo implementation shortfall.
4. Các nước có lịch giao dịch, coverage và liquidity khác nhau; không nên pool mechanically mọi ngày quan sát.
5. Kết quả risk-only tốt không chứng minh forecast có alpha.
6. V1 và V2/RC3 dùng engine khác nhau, nên không so Sharpe trực tiếp nếu không replay V1 qua engine RC3.

## 15. File kết quả chính

### RC3 final audit

- `audit_manifest.json`
- `ensemble_input_reconciliation.csv`
- `ensemble_universe_audit.csv`
- `single_seed_recomputed_metrics.csv`
- `seed_pairwise_rank_corr.csv`
- `seed_consensus_diagnostics.csv`
- `ensemble_validation_selection.csv`
- `ensemble_final_config.yaml`
- `ensemble_calibration_summary.csv`
- `ensemble_decile_validation.csv`
- `ensemble_portfolio_summary.csv`
- `risk_only_summary.csv`
- `incremental_value_summary.csv`
- `incremental_bootstrap_ci.csv`
- `portfolio_summary.csv`

### V1/V2 intermediate

- `forecast_summary.csv`
- `deep_model_summary.csv`
- `portfolio_benchmarks.csv`
- `ptcst_ablations.csv`
- `ptcst_observed_cost_sensitivity.csv`
- `quote_cost_summary.csv`
- `risk_coverage_summary.csv`
- `validation_lambda_selection.csv`
- `validation_lambda_by_country.csv`
- `cost_sensitivity_summary.csv`

## 16. Trạng thái cuối

**V2.1-RC3: audit complete, negative incremental-alpha finding, suitable for thesis Results/Discussion with cautious claims.**

Không nên chạy thêm tuning trên cùng development period. Nếu muốn nâng độ mạnh của luận văn, bước bổ sung hợp lệ nhất là:

1. lấy một holdout 2026 chưa nhìn thấy; hoặc
2. replay V1 forecasts qua đúng RC3 portfolio engine để so sánh apples-to-apples.

