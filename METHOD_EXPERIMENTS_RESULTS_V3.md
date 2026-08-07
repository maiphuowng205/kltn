# Tổng hợp phương pháp, thiết kế thực nghiệm và kết quả — Vietnam V3

## 1. Mục đích tài liệu

Tài liệu này tổng hợp trong một nơi duy nhất:

- phương pháp đề xuất **PTCST-CA-MVO**;
- bộ dữ liệu và protocol thực nghiệm đã khóa;
- các baseline dự báo và baseline danh mục;
- các kịch bản ablation, seed sweep, robustness và kiểm định thống kê;
- kết quả thực tế từ bộ Colab final handoff ngày 2026-08-06;
- cách diễn giải kết quả, giới hạn claim và những phần chưa có kết quả cuối.

Nguồn đối chiếu chính gồm `deep-research-report.md`, `IMPLEMENTATION_PLAN_V3.md`, `configs/v3_main.yaml`, `configs/v3_robustness.yaml`, implementation trong `src/v3_method.py` và các artifact trong `v3_final_handoff`.

## 2. Tóm tắt điều hành

Nghiên cứu sử dụng thiết kế hai giai đoạn:

1. PTCST dự báo excess return 5 phiên cho 100 cổ phiếu Việt Nam tại từng ngày tín hiệu tuần.
2. Dự báo được kết hợp với covariance Ledoit–Wolf và cost-aware long-only mean–variance optimization để tạo trọng số danh mục.

Kết quả chính của seed 7 trên test 2024–2025:

| Chỉ số | Kết quả |
|---|---:|
| Validation Spearman IC tốt nhất | 0.081834 |
| Mean net excess return 5 phiên | 0.371854% = 37.185 bps |
| Net Sharpe annualized | 1.435876 |
| Mean L1 turnover | 0.951605% |
| Test dates | 104 |
| Evaluation dates đầy đủ | 92 |

Kết quả có ba điểm nổi bật:

- PTCST-CA-MVO đạt lợi nhuận excess ròng dương, Sharpe khá tốt và turnover rất thấp.
- Kết quả ổn định qua năm seed; Sharpe trung bình là 1.384708 với độ lệch chuẩn 0.064309.
- Robustness vẫn cho lợi nhuận dương khi cost giả định tăng tới 50 bps và khi thay đổi covariance, lookback, giới hạn trọng số và risk aversion.

Điểm cần trình bày trung thực là test Spearman IC của PTCST chỉ khoảng 0.00137, gần bằng 0. Do đó, kết quả không hỗ trợ claim rằng PTCST có khả năng xếp hạng cổ phiếu vượt trội rõ ràng trên test. Đóng góp được hỗ trợ tốt hơn nằm ở sự tích hợp forecast, risk model, constraints và transaction-cost-aware optimization để tạo một danh mục có hiệu suất tương đối tốt với turnover thấp.

## 3. Câu hỏi nghiên cứu và đóng góp

### 3.1. Câu hỏi nghiên cứu

Câu hỏi trung tâm là:

> Liệu một Transformer kết hợp quan hệ theo thời gian và quan hệ chéo giữa cổ phiếu có tạo ra giá trị kinh tế sau khi dự báo đi qua covariance estimation, ràng buộc long-only, turnover control và transaction costs trên thị trường chứng khoán Việt Nam hay không?

### 3.2. Khoảng trống và đóng góp

Khoảng trống phù hợp nhất không phải là “Transformer đầu tiên cho portfolio optimization”. Đóng góp nằm ở việc kiểm tra đồng thời:

- dự báo lợi suất theo thời gian bằng patch-based temporal attention;
- tương tác chéo giữa 100 cổ phiếu bằng cross-sectional attention;
- covariance estimation có shrinkage;
- long-only mean–variance optimization với giới hạn trọng số;
- turnover penalty và transaction cost tại từng lần tái cân bằng;
- protocol theo thời gian, leakage checks, seed sensitivity và robustness.

Đây là đóng góp về **tính tích hợp và kỷ luật thực nghiệm**, không phải claim rằng mọi thành phần kiến trúc đều hoàn toàn mới.

### 3.3. Các giả thuyết thực nghiệm

| Giả thuyết | So sánh chính | Kết quả |
|---|---|---|
| H1: PTCST cải thiện chất lượng forecast | PTCST so với historical mean, Ridge và XGBoost | Hỗn hợp: validation IC tốt nhưng test rank IC gần 0; squared-error test ủng hộ PTCST so với baseline gộp |
| H2: MVO tạo danh mục tốt hơn chọn Top-20 trực tiếp | PTCST-MVO so với PTCST-Top20 | Được ủng hộ về Sharpe: 2.055 so với 0.959 |
| H3: Cost-aware optimization làm danh mục dễ giao dịch hơn | PTCST-CA-MVO so với PTCST-MVO | Được ủng hộ mạnh về turnover: 0.95% so với 17.04%, nhưng Sharpe giảm |
| H4: Kết quả không phụ thuộc một cấu hình duy nhất | Seed sweep và robustness grid | Được ủng hộ trong phạm vi các kịch bản đã chạy |

## 4. Pipeline tổng thể

```mermaid
flowchart LR
    A[Frozen Vietnam V3] --> B[Tensor 100 x 60 x 17]
    B --> C[PTCST forecast 5-session excess return]
    A --> D[252-session past returns]
    D --> E[Ledoit-Wolf covariance]
    C --> F[Cost-aware long-only MVO]
    E --> F
    F --> G[Weights and trades]
    G --> H[Gross return - assumed cost]
    H --> I[Net portfolio metrics]
```

## 5. Dataset và protocol đã khóa

### 5.1. Dataset freeze

| Thuộc tính | Giá trị |
|---|---|
| Scope | Vietnam-only; dữ liệu Mỹ không dùng trong nghiên cứu chính |
| Freeze ID | `vn_v3_lseg_2026-08-03` |
| Frozen files | 276 |
| Weekly dates | 399 |
| Universe | 100 RIC mỗi ngày tín hiệu |
| Weekly rows | 39,900 |
| Exact target rows | 39,642 |
| Masked target rows | 258 |
| Daily panel rows | 1,058,331 |
| Market-calendar rows | 1,999 |
| Target | `target_excess_return_5d_bps` |
| Signal frequency | Tuần, ngày quan sát cuối cùng của W-FRI week |
| Execution | Close của master-market session kế tiếp |
| Holding horizon | 5 master-market sessions |

`target_available=False` chỉ là mask cho loss/evaluation; nó không được dùng để thay đổi ex-post universe 100 mã.

### 5.2. Chia mẫu

| Split | Thời gian | Weekly dates/rows |
|---|---|---:|
| Train | 2019-01-01 đến 2022-12-31 | 207 dates / 20,700 rows |
| Validation | 2023-01-01 đến 2023-12-31 | 52 dates / 5,200 rows |
| Test | 2024-01-01 đến 2025-12-31 | 104 dates / 10,400 rows |
| Warm-up/excluded | Trước train hoặc không đủ lookback | 3,600 rows |

Toàn bộ preprocessing được fit trên train. Validation được dùng để chọn checkpoint. Test chỉ được mở sau khi protocol đã khóa.

### 5.3. Input features

Mỗi quan sát có tensor `[100 cổ phiếu, 60 phiên, 17 features]`.

| Nhóm | Features |
|---|---|
| Return/momentum | `return_1d`, `return_5d`, `return_10d`, `return_20d`, `return_60d` |
| Volatility | `vol_5d`, `vol_20d`, `vol_60d` |
| Liquidity/scale | `log_volume`, `log_dollar_volume`, `log_price`, `log_market_cap`, `amihud` |
| Range | `high_low_proxy` |
| Calendar | `day_of_week`, `is_month_end`, `is_quarter_end` |

### 5.4. Missing values và normalization

- Missing inputs được impute bằng median fit trên train.
- Mỗi feature được robust-scale bằng train median và train IQR.
- Giá trị sau scale được clip vào `[-10, 10]`.
- Không fit lại scaler/imputer bằng validation hoặc test.
- Target thiếu được loại khỏi loss bằng `target_available` mask.

## 6. Phương pháp đề xuất PTCST

PTCST là viết tắt của **Patch-based Temporal and Cross-Sectional Transformer**.

### 6.1. Temporal patching

Mỗi cổ phiếu có chuỗi 60 phiên và 17 features. Chuỗi được chia thành patch dài 5 phiên:

- số patch: `60 / 5 = 12`;
- mỗi patch ban đầu có `5 × 17 = 85` giá trị;
- linear projection chuyển mỗi patch sang embedding kích thước 64;
- positional embedding được thêm vào 12 temporal tokens.

Hai temporal Transformer encoder layers học quan hệ giữa các patch theo thời gian. Embedding của patch cuối được dùng làm biểu diễn temporal cho từng cổ phiếu.

### 6.2. Cross-sectional attention

Sau temporal encoder, mỗi ngày có 100 stock embeddings kích thước 64. Một cross-sectional Transformer layer cho phép mỗi cổ phiếu attention đến các cổ phiếu khác trong cùng ngày.

Mục tiêu của tầng này là học các quan hệ tương đối, đồng biến và cấu trúc thị trường mà một temporal-only model không biểu diễn trực tiếp.

### 6.3. Prediction head

Mỗi stock embedding đi qua:

`LayerNorm → Linear(64, 32) → GELU → Dropout → Linear(32, 1)`

Output là dự báo excess return 5 phiên, đơn vị basis points.

### 6.4. Cấu hình huấn luyện

| Hyperparameter | Giá trị |
|---|---:|
| `d_model` | 64 |
| Temporal layers | 2 |
| Cross-sectional layers | 1 |
| Attention heads | 4 |
| FFN dimension | 128 |
| Dropout | 0.10 |
| Loss | Huber |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Gradient clip norm | 1.0 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Selection metric | Validation mean weekly Spearman IC |
| Seeds | 7, 19, 43, 71, 101 |

## 7. Risk model và portfolio optimizer

### 7.1. Covariance

Covariance chính được ước lượng từ 252 phiên quá khứ, kết thúc tại signal date, bằng `sklearn.covariance.LedoitWolf`.

Một cổ phiếu không có đủ covariance history sẽ giữ pre-trade weight và không được thay bằng một cổ phiếu xếp hạng thấp hơn ngoài universe. Nếu có dưới 20 tài sản hợp lệ, policy fallback là giữ `w_pre`.

### 7.2. Cost-aware long-only MVO

Tại mỗi ngày tái cân bằng, optimizer giải:

$$
\max_w \quad w^\top \hat{\mu}
- \frac{\lambda}{2}w^\top\hat{\Sigma}w
- c\lVert w-w_{pre}\rVert_1
$$

với các ràng buộc:

$$
\sum_i w_i=1, \qquad 0\leq w_i\leq 0.05,
\qquad \lVert w-w_{pre}\rVert_1\leq0.40.
$$

Cấu hình chính:

| Thành phần | Giá trị |
|---|---:|
| Risk aversion $\lambda$ | 10 |
| Assumed proportional cost $c$ | 0.001 = 10 bps |
| Maximum asset weight | 5% |
| L1 turnover cap | 40% |
| Solver order | CLARABEL, sau đó OSQP |

### 7.2.1. Kiểm tra đơn vị đầu vào optimizer

Target và output forecast được lưu ở basis points, nhưng optimizer sử dụng decimal-return units. Trong implementation chính, dự báo được đổi scale trước khi gọi optimizer:

```text
mu_decimal = prediction_bps / 10,000
```

Covariance được fit từ daily `return` ở decimal units; vì vậy expected-return term, covariance risk term và transaction-cost term cùng nằm trên một scale kinh tế. `cost=0.001` tương ứng 10 bps trên mỗi đơn vị L1 turnover theo protocol.

Để tăng khả năng audit khi chạy lại, nên lưu thêm diagnostic cho ít nhất một signal date: median/std của `mu_decimal`, mean diagonal của covariance, expected-return term, risk term và turnover-penalty term. Diagnostic này không thay đổi kết quả hiện tại; nó chỉ chứng minh rõ rằng `risk_aversion=10` không đang bù cho một lỗi scale.

“Cost-aware” nghĩa là optimizer nhìn thấy turnover penalty trước khi chọn giao dịch. Tất cả strategy, kể cả cost-unaware, vẫn bị trừ realized assumed cost khi đánh giá.

### 7.3. Accounting

Turnover và net return được tính như sau:

$$
Turnover_t=\sum_i|w_{i,t}-w_{i,t}^{pre}|+ExitedWeight_t
$$

$$
NetExcessReturn_{t,t+5}=w_t^\top r_{t,t+5}^{excess}-0.001\times Turnover_t.
$$

Sau mỗi kỳ giữ, trọng số được drift theo raw return rồi mới dùng làm pre-trade weights của lần tái cân bằng kế tiếp.

Trong code hiện tại, `w_pre` được tạo trên universe hiện tại `U_t`; phần weight của các mã đã rời universe được lưu riêng trong `exited_weight`. Vì vậy công thức L1 cộng thêm exited weight là nhất quán với implementation hiện tại. Cách viết tổng quát, dễ audit hơn là:

$$
Turnover_t=
\sum_{i\in U_t}|w_{i,t}-\tilde w^{pre}_{i,t}|
 +
\sum_{i\in U_{t-1}\setminus U_t}w^{pre}_{i,t}.
$$

Nếu sau này đưa cả hai vector lên union universe `U_t\cup U_{t-1}`, chỉ được dùng một tổng L1 duy nhất để tránh double count. Nên bổ sung một unit test với universe thay đổi thủ công.

## 8. Baseline và kịch bản thực nghiệm

### 8.1. Forecast baselines đã chạy

| Model | Mục đích |
|---|---|
| Zero | Sanity check; luôn dự báo 0 |
| Expanding historical mean | Dự báo từ trung bình target lịch sử đã quan sát |
| Pooled Ridge | Linear baseline, flatten chuỗi 60 phiên |
| XGBoost | Nonlinear tabular baseline dùng last step và sequence summaries |

Vanilla Temporal Transformer và PatchTST đã có implementation, nhưng final Colab handoff không chứa bảng kết quả deep-baseline hoàn chỉnh. Notebook đã để `RUN_DEEP_BASELINES=False`; vì vậy không được tự tạo hoặc suy diễn kết quả của hai model này.

### 8.2. Portfolio baselines đã chạy

| Strategy | Forecast | Allocation | Cost-aware decision |
|---|---|---|---|
| EW | Không | Weekly equal weight | Không |
| EW-BH | Không | Equal weight ban đầu, sau đó drift | Không |
| MinVar | Không | Ledoit–Wolf minimum variance | Không |
| HM-MVO | Historical mean | Frictionless MVO | Không |
| Ridge-MVO | Ridge | Frictionless MVO | Không |
| XGB-MVO | XGBoost | Frictionless MVO | Không |
| XGB-CA-MVO | XGBoost | Cost-aware MVO | Có |

### 8.3. PTCST ablations

| Strategy | Vai trò |
|---|---|
| PTCST-Top20 | Kiểm tra chọn trực tiếp top 20 dự báo |
| PTCST-MVO | Kiểm tra giá trị của frictionless MVO |
| PTCST-CA-MVO | Phương pháp chính; kiểm tra cost awareness |

### 8.4. Seed sweep

PTCST được chạy với năm seed `7, 19, 43, 71, 101`. Kết quả được tổng hợp bằng mean và sample standard deviation. Seed 7 là protocol-locked main run; không chọn seed tốt nhất dựa trên test.

### 8.5. Robustness grid

Robustness là one-dimension-at-a-time, tổng cộng 21 rows trong output:

| Dimension | Giá trị kiểm tra |
|---|---|
| Cost | 0, 5, 10, 20, 30, 50 bps |
| Covariance | Ledoit–Wolf, sample, EWMA half-life 60 |
| Covariance lookback | 20, 60, 120 sessions; base 252 xuất hiện trong các nhóm khác |
| Turnover cap | 20%, 40%, 80% |
| Max weight | 3%, 5%, 10% |
| Risk aversion | 5, 10, 20 |

Robustness chỉ dùng để kiểm tra độ nhạy, không dùng để chọn lại winner sau khi đã xem test.

### 8.6. Các kiểm tra chất lượng

- Freeze/checksum validation.
- Tensor and timestamp contract validation.
- Leakage report.
- Risk coverage report trên train/validation.
- Determinism repeat cho forecast và portfolio baselines.
- DM/HLN-style forecast-loss test.
- Paired block bootstrap theo forecast date.
- Solver/fallback logging.

## 9. Metrics

### 9.1. Forecast metrics

| Metric | Ý nghĩa | Tốt hơn khi |
|---|---|---|
| Spearman IC | Tương quan thứ hạng giữa prediction và realized return | Cao và dương |
| Pearson IC | Tương quan tuyến tính prediction–target | Cao và dương |
| MAE | Sai số tuyệt đối trung bình, bps | Thấp |
| RMSE | Căn sai số bình phương trung bình, bps | Thấp |
| Directional accuracy | Tỷ lệ đúng dấu return | Cao hơn 50% |
| Top-minus-bottom spread | Realized return top 20% trừ bottom 20% theo prediction | Cao và dương |

### 9.2. Portfolio metrics

| Metric | Ý nghĩa | Tốt hơn khi |
|---|---|---|
| Mean net excess return 5d | Excess return sau assumed transaction cost | Cao |
| Net Sharpe annualized | Mean/std nhân $\sqrt{52}$ | Cao |
| Mean L1 turnover | Tổng thay đổi tuyệt đối của weights mỗi rebalance | Thấp, nếu performance được giữ |
| Evaluation dates | Số kỳ có đầy đủ realized five-session outcomes | Cao và minh bạch |

## 10. Kết quả huấn luyện PTCST chính

| Chỉ số | Giá trị | Diễn giải |
|---|---:|---|
| Best epoch | 4 | Checkpoint tốt nhất xuất hiện sớm |
| Epochs completed | 14 | Early stopping sau 10 epoch không cải thiện |
| Best validation Spearman IC | 0.081834 | Tín hiệu validation dương và đáng chú ý |
| Device | CUDA | Thông tin runtime, không phải performance metric |
| Seed | 7 | Main protocol seed |

Validation IC tăng từ 0.069661 ở epoch 1 lên 0.081834 ở epoch 4, sau đó suy giảm và còn 0.060533 ở epoch 14. Early stopping đã chọn đúng checkpoint tốt nhất theo metric đã khóa.

## 11. Kết quả forecast

### 11.1. Baseline validation

| Model | Spearman IC | Pearson IC | MAE bps | RMSE bps | Directional accuracy | Top-bottom bps |
|---|---:|---:|---:|---:|---:|---:|
| Historical mean | 0.028004 | 0.015162 | 321.883 | 427.292 | 50.612% | 22.497 |
| Ridge | -0.018946 | -0.020047 | 506.203 | 599.131 | 50.386% | -19.150 |
| XGBoost | 0.010128 | 0.008805 | 335.631 | 442.807 | 48.958% | 18.218 |
| Zero | N/A | N/A | 318.376 | 425.462 | N/A* | N/A† |
| **PTCST** | **0.081834** | **0.083511** | **318.318** | **425.382** | **51.956%** | **81.121** |

`*` Zero dự báo đúng bằng 0 nên directional accuracy theo cách tính sign không có ý nghĩa thực tế. `†` Top-minus-bottom của forecast hằng là không xác định vì mọi cổ phiếu bị tie; giá trị raw output 66.294 bps chỉ phản ánh tie-breaking theo thứ tự hàng và không được xem là forecast skill.

Trên validation, PTCST có Spearman IC và top-minus-bottom spread tốt hơn các baseline đã chạy.

### 11.2. Test forecast

`forecast_summary.csv` chỉ chứa baseline. Dòng PTCST dưới đây được tổng hợp từ `forecast_metrics_by_date.parquet` của main method run.

| Model | Spearman IC | Pearson IC | MAE bps | RMSE bps | Directional accuracy | Top-bottom bps |
|---|---:|---:|---:|---:|---:|---:|
| Historical mean | -0.016390 | -0.003070 | 329.100 | 455.991 | 51.557% | -20.547 |
| Ridge | 0.038463 | 0.038595 | 447.712 | 563.839 | 52.820% | 50.997 |
| XGBoost | 0.001838 | 0.033137 | 333.990 | 459.868 | 48.767% | 26.241 |
| Zero | N/A | N/A | 325.048 | 454.029 | N/A* | N/A† |
| **PTCST** | **0.001370** | **0.025842** | **325.052** | **454.029** | **49.903%** | **28.252** |

Diễn giải:

- Test Spearman IC của PTCST gần bằng 0, nên không có bằng chứng mạnh về khả năng xếp hạng out-of-sample.
- Test directional accuracy 49.9% cũng gần mức ngẫu nhiên.
- PTCST có top-minus-bottom spread dương 28.25 bps, gần XGBoost nhưng thấp hơn Ridge.
- MAE/RMSE của PTCST gần zero forecast. Vì vậy, không nên xem error metric này là một cải thiện kinh tế lớn so với zero.
- Validation tốt nhưng test yếu gợi ý khả năng tổng quát hóa forecast còn hạn chế hoặc thị trường đã thay đổi regime.

Với forecast hằng như Zero, quantile spread và directional accuracy được đánh dấu `N/A`; chúng không phải các metric hợp lệ nếu chưa định nghĩa một tie-breaking policy có ý nghĩa kinh tế.

## 12. Kết quả portfolio chính

Main PTCST-CA-MVO seed 7:

| Chỉ số | Giá trị |
|---|---:|
| Mean net excess return 5d | 0.371854% = 37.185 bps |
| Net Sharpe annualized | 1.435876 |
| Mean turnover | 0.951605% |
| Test dates | 104 |
| Complete evaluation dates | 92 |
| Excluded incomplete dates | 12 |
| Evaluation coverage | 88.462% |

Quy đổi minh họa `0.371854% × 52` cho annualized arithmetic excess return khoảng 19.34%; compounding cùng một mean rate cho khoảng 21.29%. Đây không phải CAGR chính thức và không nên dùng thay cho đường wealth thực tế.

### 12.1. So sánh với portfolio baselines

| Strategy | Evaluation dates | Net excess return 5d | Net Sharpe | Mean turnover |
|---|---:|---:|---:|---:|
| **PTCST-CA-MVO** | **92** | **0.3719%** | **1.4359** | **0.9516%** |
| EW | 92 | 0.3176% | 1.0242 | 3.6524% |
| EW-BH | 92 | 0.3671% | 1.1466 | 0.2878% |
| HM-MVO | 92 | 0.0842% | 0.2652 | 12.4268% |
| MinVar | 92 | 0.3113% | 1.6360 | 9.2027% |
| Ridge-MVO | 92 | 0.4817% | 1.3902 | 40.0000% |
| XGB-CA-MVO | 92 | 0.4577% | 1.2085 | 37.9512% |
| XGB-MVO | 92 | 0.4377% | 1.1520 | 39.9457% |

Kết luận từ bảng:

- PTCST-CA-MVO không có mean return cao nhất; Ridge-MVO cao hơn nhưng chạm turnover cap 40%.
- PTCST-CA-MVO không có Sharpe cao nhất trong nhóm baseline; MinVar đạt 1.6360.
- So với EW, PTCST-CA-MVO có return và Sharpe cao hơn, đồng thời turnover thấp hơn.
- So với EW-BH, PTCST-CA-MVO có mean return gần tương đương, Sharpe cao hơn nhưng turnover cũng cao hơn.
- So với Ridge/XGB MVO, PTCST-CA-MVO có turnover thấp hơn rất lớn. Đây là lợi thế thực dụng nổi bật nhất.

## 13. Kết quả ablation

| Strategy | Net excess return 5d | Net Sharpe | Mean turnover |
|---|---:|---:|---:|
| PTCST-CA-MVO | 0.3719% | 1.4359 | 0.9516% |
| PTCST-MVO | 0.4313% | 2.0553 | 17.0417% |
| PTCST-Top20 | 0.4269% | 0.9590 | 36.9075% |

### 13.1. Giá trị của optimizer

PTCST-MVO so với PTCST-Top20:

- net return gần nhau: 0.4313% so với 0.4269%;
- Sharpe tăng mạnh: 2.0553 so với 0.9590;
- turnover giảm từ 36.91% xuống 17.04%.

Điều này hỗ trợ vai trò của covariance-aware MVO thay vì chỉ chọn top dự báo.

### 13.2. Giá trị của cost awareness

PTCST-CA-MVO so với PTCST-MVO:

- turnover giảm khoảng 94.4%, từ 17.04% xuống 0.95%;
- mean net return giảm từ 0.4313% xuống 0.3719%;
- Sharpe giảm từ 2.0553 xuống 1.4359.

Cost awareness tạo trade-off rõ ràng: hy sinh một phần backtest performance để có danh mục ít giao dịch và dễ triển khai hơn.

## 14. Kết quả seed sweep

| Seed | Validation IC | Net excess return 5d | Net Sharpe | Mean turnover |
|---:|---:|---:|---:|---:|
| 7 | 0.081834 | 0.3719% | 1.4359 | 0.9516% |
| 19 | 0.081694 | 0.3507% | 1.3745 | 0.8907% |
| 43 | 0.075220 | 0.3421% | 1.3353 | 0.8646% |
| 71 | 0.083836 | 0.3384% | 1.3136 | 0.8732% |
| 101 | 0.078609 | 0.3771% | 1.4642 | 0.9544% |
| **Mean** | **0.080239** | **0.3561%** | **1.3847** | **0.9069%** |
| **Std** | **0.003371** | **0.0175 percentage point** | **0.0643** | **0.0431 percentage point** |

Cả năm seed đều có return dương và Sharpe từ 1.3136 đến 1.4642. Kết quả ít nhạy với random initialization. Tuy nhiên, tất cả seed dùng cùng test period nên seed stability không đo được time-sample uncertainty.

## 15. Kết quả robustness

### 15.1. Cost sensitivity

| Assumed cost | Net excess return 5d | Net Sharpe | Turnover |
|---:|---:|---:|---:|
| 0 bps | 0.4484% | 2.1357 | 16.7665% |
| 5 bps | 0.4238% | 1.8297 | 1.7580% |
| 10 bps | 0.3719% | 1.4359 | 0.9970% |
| 20 bps | 0.3208% | 1.0944 | 0.4896% |
| 30 bps | 0.3204% | 1.0932 | 0.4896% |
| 50 bps | 0.3197% | 1.0909 | 0.4897% |

Performance vẫn dương ở cost 50 bps. Khi optimizer nhìn thấy cost cao hơn, nó tự giảm turnover, nên net result không giảm tuyến tính theo cost.

### 15.2. Covariance estimator

| Covariance | Net excess return 5d | Net Sharpe | Turnover |
|---|---:|---:|---:|
| Ledoit–Wolf | 0.3719% | 1.4359 | 0.9970% |
| Sample | 0.3720% | 1.4443 | 1.0119% |
| EWMA half-life 60 | 0.4092% | 1.6226 | 1.2522% |

Ba estimator đều dương; kết quả không phụ thuộc hoàn toàn vào Ledoit–Wolf. EWMA tốt hơn trong sample này nhưng không được dùng để thay đổi main specification sau khi xem test.

### 15.3. Lookback sensitivity

| Lookback | Net excess return 5d | Net Sharpe | Turnover |
|---:|---:|---:|---:|
| 20 sessions | 0.4360% | 1.7860 | 2.3328% |
| 60 sessions | 0.3380% | 1.3375 | 1.3107% |
| 120 sessions | 0.3631% | 1.4522 | 1.2763% |

Tất cả lookback đều dương. Lookback 20 tốt nhất trong grid nhưng đây là sensitivity result, không phải lý do hợp lệ để thay main protocol.

### 15.4. Turnover cap

| Turnover cap | Net excess return 5d | Net Sharpe | Turnover |
|---:|---:|---:|---:|
| 20% | 0.3716% | 1.4348 | 0.9967% |
| 40% | 0.3719% | 1.4359 | 0.9970% |
| 80% | 0.3719% | 1.4359 | 0.9970% |

Kết quả gần như giống nhau, cho thấy turnover cap không binding trong main cost-aware solution.

### 15.5. Maximum weight

| Max weight | Net excess return 5d | Net Sharpe | Turnover |
|---:|---:|---:|---:|
| 3% | 0.3902% | 1.4722 | 1.3084% |
| 5% | 0.3719% | 1.4359 | 0.9970% |
| 10% | 0.3085% | 1.2289 | 0.8526% |

Cho phép concentration tới 10% làm performance xấu hơn trong sample; diversification constraint có giá trị.

### 15.6. Risk aversion

| Risk aversion | Net excess return 5d | Net Sharpe | Turnover |
|---:|---:|---:|---:|
| 5 | 0.3493% | 1.1915 | 0.5629% |
| 10 | 0.3719% | 1.4359 | 0.9970% |
| 20 | 0.4017% | 1.7731 | 1.6379% |

Cả ba mức đều dương. Không có solver fallback được ghi nhận trong robustness output.

## 16. Risk coverage

Covariance dùng cửa sổ 252 phiên và risk-coverage audit chỉ mở train/validation trước test.

| Split | Dates | Mean valid assets | Min valid assets | Fallback dates |
|---|---:|---:|---:|---:|
| Train | 207 | 88.845 | 80 | 0 |
| Validation | 52 | 96.500 | 95 | 0 |
| Tổng | 259 | — | — | 0 |

Fallback fraction bằng 0. Coverage không phải 100% tài sản ở mọi ngày, nhưng số tài sản hợp lệ đủ cao và không buộc risk engine phải dùng fallback.

## 17. Kiểm định thống kê

Forecast-loss comparison dùng squared error theo bps, tổng hợp ở cấp forecast date:

| Thành phần | Kết quả |
|---|---:|
| Comparison | Baseline vs PTCST |
| Inference dates | 102 |
| HAC lag | 4 |
| Mean loss difference | 35,448.241 bps² |
| DM/HLN t-statistic | 4.0104 |
| DM/HLN p-value | 0.000116 |
| Block-bootstrap repetitions | 2,000 |
| Block length | 12 dates |
| Bootstrap 95% CI | [19,044.282; 55,188.284] bps² |
| Bootstrap empirical p-value | 0.0 |

Loss difference được tính là `baseline squared error - PTCST squared error`; giá trị dương ủng hộ PTCST. Confidence interval không chứa 0 và DM/HLN p-value nhỏ hơn 0.001.

Giới hạn diễn giải:

- kiểm định này nói về squared forecast error, không phải Sharpe hoặc return;
- nhãn `baseline` trong statistical run chứa toàn bộ baseline forecast file, nên đây là baseline gộp chứ không phải kiểm định riêng cho Ridge, XGBoost hoặc historical mean;
- bootstrap `p_value=0.0` nghĩa là không có lần lặp nào trong 2,000 repetitions cho statistic cực đoan tương ứng, không phải xác suất toán học tuyệt đối bằng 0;
- kết quả không chứng minh test rank IC của PTCST cao, vì rank IC và squared error đo hai thuộc tính khác nhau.

## 18. Leakage, determinism và reproducibility

### 18.1. Leakage report

`v3_leakage_report.json` có trạng thái `PASS`:

| Check | Kết quả |
|---|---|
| Train scaler mutation check | PASS |
| Timestamp check | PASS |
| Target-mask/universe check | PASS |
| Tensor shape | 100 assets × 60 sessions × 17 features |

### 18.2. Determinism report

Forecast baseline và portfolio benchmark được chạy lặp lại. Hash của các artifact chính giống nhau giữa main và repeat, gồm:

- forecasts và forecast metrics;
- portfolio returns;
- weights và trades;
- solver log;
- portfolio summary.

### 18.3. Run identity

| Thành phần | Giá trị |
|---|---|
| Method-run Git commit | `c52e00ebab2cb0025881e6a4b80b9ed672edbf06` |
| Checkpoint SHA-256 | `9415415a58dd51eca1e1e65b348a3733d44890e38420526feb230bb0213f7429` |
| Training cutoff | 2022-12-30 |
| Validation/selection cutoff | 2023-12-29 |
| Freeze manifest SHA-256 | `125c4a61da4ab8718710eeddbde8bd9d4c364e53e052d80dad83d7cb2dd68b2d` |
| Final Drive sync | `SYNC_COMPLETE` |

Final sync chứa 38 handoff files và archive của 8 notebooks.

## 19. Review follow-up và các bước cần làm trước khi chốt luận văn

### 19.1. Đã xác minh từ implementation

- `scripts/run_v3_ptcst_method.py` chuyển forecast từ bps sang decimal bằng `pred / 10000.0` trước khi gọi optimizer.
- Covariance dùng daily decimal returns, phù hợp với scale của `mu_decimal`.
- Turnover code tính L1 trên universe hiện tại và cộng riêng `exited_weight`; không double-count trong implementation hiện tại.
- Zero forecast có tie toàn bộ tài sản; raw top-bottom spread trong CSV không có ý nghĩa và đã được đánh dấu `N/A` trong bảng diễn giải ở tài liệu này.

### 19.2. Các việc cần rerun, chưa được thực hiện trong final handoff

Các hạng mục dưới đây không được suy diễn từ kết quả hiện tại. Mỗi mục phải tạo run/artifact mới; không ghi đè `v3_final_handoff`.

| Ưu tiên | Hạng mục | Entrypoint dự kiến | Artifact bắt buộc | Trạng thái |
|---|---|---|---|---|
| P0 | Forecast pairwise inference với Zero, Historical Mean, Ridge, XGBoost; DM/HLN, block bootstrap và Holm correction | `scripts/run_v3_pairwise_forecast_tests.py` | `runs/v3_extension_p0/pairwise_forecast_tests/forecast_pairwise_tests.json` | Đã chạy |
| P0 | Optimizer unit/scale audit | `scripts/audit_v3_optimizer_scale.py` | `runs/v3_extension_p0/optimizer_scale_diagnostic.json` | Đã chạy |
| P0 | Turnover edge-case với asset rời universe | `tests/test_v3_kernels.py` | 5/5 tests passed | Đã chạy |
| P0 | Architecture ablation: temporal-only, no-patching, PatchTST-style, full PTCST | `scripts/generate_v3_architecture_ablation.py` | `runs/v3_extension_p0/architecture_ablation/architecture_ablation_summary.csv` | Đã đóng gói từ verified local runs |
| P1 | Paired bootstrap cho net return, Sharpe, turnover, drawdown | `scripts/run_v3_portfolio_pairwise_tests.py` | `runs/v3_extension_p1/portfolio_inference/portfolio_pairwise_tests.json` | Đã chạy |
| P1 | Wealth curve và economic metrics | `scripts/generate_v3_economic_metrics.py` | `runs/v3_extension_p1/economic_metrics/economic_metrics.csv` và `portfolio_wealth.parquet` | Đã chạy |
| P1 | Tách performance 2024 và 2025 | `scripts/generate_v3_economic_metrics.py` | `runs/v3_extension_p1/economic_metrics/year_split_portfolio_summary.csv` | Đã chạy |
| P2 | Expanding quarterly walk-forward và cutoff audit | `scripts/run_v3_walk_forward.py` | `runs/v3_extension_p2/quarterly_walk_forward/` | Đã chạy trong handoff extension riêng; không ghi đè final handoff |
| P2 | Transaction-cost validation bằng bid/ask hoặc tick data | dataset version mới | dataset freeze và handoff mới | Phụ thuộc dữ liệu |

### 19.3. Kết quả extension đã chạy

### Pairwise forecast inference

Sau Holm correction trên bốn comparison:

| Baseline so với PTCST | Mean loss difference (bps²) | Holm DM/HLN p-value | Bootstrap CI (bps²) |
|---|---:|---:|---:|
| Zero | 15.15 | 0.6759 | [-49.77; 93.51] |
| Historical mean | 2,347.17 | 0.4946 | [-811.67; 5,529.59] |
| Ridge | 134,070.34 | 0.0011 | [65,368.08; 216,051.30] |
| XGBoost | 5,115.83 | 0.3072 | [-543.23; 11,575.04] |

Trong extension này chỉ Ridge có khác biệt squared-error có ý nghĩa sau điều chỉnh multiple comparisons. Vì vậy, kết quả trước đó không được dùng để claim PTCST thắng từng baseline.

### Architecture ablation

Các run local cùng frozen data, seed 7 và budget 100 epochs cho kết quả:

| Model | Validation IC | Test Spearman IC | Test RMSE bps | Net Sharpe | Turnover |
|---|---:|---:|---:|---:|---:|
| TemporalTransformer | 0.0864 | 0.0149 | 454.047 | 1.3210 | 0.8763% |
| PatchTST | 0.0841 | 0.0014 | 454.036 | 1.3454 | 0.8755% |
| PTCST | 0.0813 | -0.0019 | 454.020 | 1.4232 | 0.9416% |

Đây là extension local, không thay thế main Colab handoff. Trong sample này, PTCST có Sharpe portfolio cao nhất trong ba architecture run nhưng không có test rank IC cao; architecture claim nên tiếp tục được diễn giải thận trọng.

### Portfolio inference và economic metrics

Portfolio paired bootstrap có 92 evaluation dates, circular block length 11 và 2,000 repetitions. Kết quả mean-return sign-flip sau Holm không có comparison nào đạt ý nghĩa thống kê ở mức thông thường. Một số khoảng tin cậy Sharpe/turnover/drawdown vẫn có ý nghĩa mô tả, nhưng không được chuyển thành claim nhân quả.

Extension đã tạo thêm:

- `portfolio_pairwise_tests.json` và `portfolio_pairwise_by_date.parquet`;
- `economic_metrics.csv`;
- `year_split_portfolio_summary.csv`;
- `portfolio_wealth.parquet`;
- manifest riêng cho architecture và economic extension.

Các artifact extension dùng verified local runs và được lưu dưới `runs/v3_extension_p0`/`runs/v3_extension_p1`; chúng không ghi đè `v3_final_handoff`.

### 19.4. Tiêu chí hoàn thành extension

- Mỗi extension giữ nguyên freeze ID, target, feature list và main test protocol, hoặc tạo freeze/protocol version mới nếu thay đổi.
- Không chọn lại cấu hình bằng test labels.
- Mọi output ghi seed, Git commit, config hash, freeze ID và evaluation dates.
- Kết quả mới được lưu trong handoff version riêng và không sửa số liệu V3 hiện tại.
- Các P0/P1 trong bảng trên đã được chạy hoặc đóng gói từ verified local runs. Quarterly walk-forward và cutoff audit P2 đã chạy trong thư mục extension riêng; transaction-cost validation bằng bid/ask vẫn pending vì frozen V3 không có observed quotes.

## 20. Những phần đã hoàn thành và chưa có kết quả cuối

| Hạng mục | Trạng thái final handoff |
|---|---|
| Dataset validation và tensor cache | Hoàn thành |
| Zero/HM/Ridge/XGBoost forecast baselines | Hoàn thành |
| Risk coverage và portfolio baselines | Hoàn thành |
| PTCST main seed 7 | Hoàn thành |
| PTCST five-seed sweep | Hoàn thành |
| PTCST Top20/MVO/CA-MVO ablations | Hoàn thành |
| Statistical tests | Hoàn thành |
| 21-row robustness output | Hoàn thành |
| Leakage/determinism checks | Hoàn thành |
| Final handoff và Drive sync | Hoàn thành |
| Vanilla Transformer/PatchTST final comparison | Chưa có trong final handoff; optional cell đã skip |
| Quarterly expanding walk-forward retraining | Có artifact trong `runs/v3_extension_p2`; không đưa vào final handoff V3 |
| Sortino, maximum drawdown, Calmar, CVaR, CE | Được nêu trong plan nhưng không có trong bảng final hiện tại |
| Statistical test cho portfolio-return/Sharpe differences | Chưa có trong final statistical file |

Các phần chưa có không được mô tả là đã chạy trong luận văn. Nếu cần bổ sung, chúng phải được chạy như experiment extension và lưu trong một handoff version mới.

## 21. Giới hạn dữ liệu và giới hạn claim

Dataset freeze ghi rõ:

- monthly historical constituents giảm nhưng không loại bỏ hoàn toàn survivorship bias;
- không có complete delisting ledger với đầy đủ ngày và lý do hủy niêm yết;
- corporate-action/total-return semantics vẫn cần thận trọng khi diễn giải kinh tế;
- risk-free rate là VND one-month deposit cash proxy và historical publication timestamp chưa đầy đủ;
- transaction cost là giả định vì không có historical bid/ask hoặc tick data;
- 12/104 test dates bị loại khỏi portfolio metrics do five-session outcomes không đầy đủ;
- test portfolio chỉ có 92 evaluation dates, nên precision của Sharpe/return còn hạn chế;
- robustness dùng cùng test sample và không phải một out-of-sample test độc lập mới.

Không nên dùng các claim:

- “survivorship-bias-free”;
- “đầy đủ mọi cổ phiếu đã hủy niêm yết”;
- “transaction cost thực tế đã được quan sát”;
- “PTCST vượt trội mọi baseline”;
- “forecast PTCST có rank IC mạnh trên test”;
- “kết quả đảm bảo hiệu quả giao dịch thực tế”.

## 22. Kết luận được hỗ trợ bởi kết quả

Kết luận phù hợp nhất là:

> Trên frozen Vietnam V3 experimental panel, PTCST-CA-MVO tạo mean net excess return 37.19 bps mỗi chu kỳ 5 phiên, net Sharpe annualized 1.436 và mean turnover khoảng 0.95%. Kết quả duy trì dương qua năm random seeds và các sensitivity scenarios về cost, covariance, lookback, constraints và risk aversion. PTCST đạt validation rank IC dương 0.0818, nhưng test rank IC gần bằng 0, vì vậy lợi thế của phương pháp không nên được mô tả như sự vượt trội tuyệt đối về stock ranking. Bằng chứng mạnh hơn nằm ở việc kết hợp forecast với covariance estimation và cost-aware optimization để đạt sự cân bằng giữa hiệu suất, rủi ro và khả năng giao dịch.

### 21.1. Cách trả lời từng giả thuyết

- **Forecast value:** chỉ được hỗ trợ một phần. Validation và squared-error inference tích cực, nhưng test rank IC yếu.
- **Optimizer value:** được hỗ trợ. PTCST-MVO có Sharpe cao hơn rõ rệt so với PTCST-Top20.
- **Cost-awareness value:** được hỗ trợ mạnh về turnover và tính thực dụng, nhưng có chi phí cơ hội về return/Sharpe.
- **Overall dominance:** không được hỗ trợ. MinVar có Sharpe cao hơn main PTCST-CA-MVO; Ridge-MVO có mean return cao hơn nhưng turnover cực cao.
- **Robustness:** được hỗ trợ trong grid đã định trước, nhưng chưa thay thế cho validation trên giai đoạn hoặc thị trường khác.

## 23. Bản đồ artifact kết quả

| Artifact | Nội dung |
|---|---|
| `metrics.json` | Main PTCST training và portfolio metrics |
| `forecast_metrics_by_date.parquet` | PTCST forecast metrics theo ngày |
| `forecast_summary.csv` | Zero/HM/Ridge/XGBoost forecast summary |
| `portfolio_summary.csv` | Non-deep portfolio benchmark summary |
| `ptcst_ablation_summary.csv` | PTCST-Top20/MVO/CA-MVO comparison |
| `seed_summary.parquet` | Kết quả từng seed |
| `seed_mean_std.parquet` | Mean/std của seed sweep |
| `robustness_results.parquet` | 21 robustness rows |
| `risk_coverage_metrics.json` | Risk-history coverage và fallbacks |
| `statistical_tests.json` | DM/HLN và block bootstrap |
| `portfolio_returns.parquet` | Return/cost/turnover theo test date |
| `weights.parquet` | Target và pre-trade weights |
| `trades.parquet` | Trades theo ngày và RIC |
| `solver_log.parquet` | Solver/risk status theo ngày |
| `best.pt` | Main seed-7 PTCST checkpoint |
| `protocol_lock.json` | Protocol được khóa trước test |
| `v3_leakage_report.json` | Leakage checks |
| `v3_determinism_report.json` | Repeat-run identity checks |
| `sync_manifest.json` | SHA-256 và trạng thái final Drive sync |
| `runs/v3_extension_p0/pairwise_forecast_tests/forecast_pairwise_tests.json` | Forecast inference riêng từng baseline và Holm correction |
| `runs/v3_extension_p0/optimizer_scale_diagnostic.json` | Diagnostic đơn vị và magnitude của optimizer objective |
| `runs/v3_extension_p0/architecture_ablation/architecture_ablation_summary.csv` | So sánh TemporalTransformer, PatchTST và PTCST |
| `runs/v3_extension_p1/portfolio_inference/portfolio_pairwise_tests.json` | Paired block bootstrap cho portfolio comparisons |
| `runs/v3_extension_p1/economic_metrics/economic_metrics.csv` | Wealth/economic metrics mở rộng |
| `runs/v3_extension_p1/economic_metrics/year_split_portfolio_summary.csv` | Tách kết quả 2024 và 2025 |
| `runs/v3_extension_p1/economic_metrics/portfolio_wealth.parquet` | Wealth index theo ngày và strategy |

## 24. Notebook execution map

| Notebook | Vai trò |
|---|---|
| `00_colab_setup_and_validate.ipynb` | Clone code, mount Drive, restore và validate frozen data |
| `01_build_tensor_cache.ipynb` | Tạo tensor cache và leakage report |
| `02_run_forecast_baselines.ipynb` | Chạy forecast baselines |
| `03_run_risk_optimizer_backtest.ipynb` | Risk coverage và portfolio benchmarks |
| `04_train_ptcst.ipynb` | Huấn luyện PTCST và seed sweep |
| `05_run_main_ablations.ipynb` | Chạy PTCST portfolio ablations |
| `06_locked_test_and_inference.ipynb` | Locked test, statistical tests và robustness |
| `07_generate_tables.ipynb` | Tạo report tables, handoff và sync lên Drive |

---

Tài liệu này phản ánh đúng các artifact hiện có tại thời điểm final sync 2026-08-06. Mọi thay đổi dataset, target, universe, cost rule, test period hoặc model selection sau thời điểm này phải tạo một experiment version và handoff mới thay vì sửa ngầm kết quả V3.

## 25. P2 extension update (2026-08-07)

The remaining walk-forward extension was implemented and run separately from
the frozen final handoff:

| Item | Result | Artifact |
|---|---|---|
| Expanding quarterly Ridge walk-forward | 104 test dates, 8 quarterly retraining points | `runs/v3_p2_handoff/quarterly_walk_forward/` |
| Realized-label cutoff audit | PASS; 0 future rows used | `runs/v3_p2_handoff/quarterly_walk_forward/cutoff_audit.json` and `.parquet` |
| Transaction-cost data availability audit | BLOCKED_NO_OBSERVED_QUOTES; 0 complete bid/ask rows, 100% imputed cost | `runs/v3_p2_handoff/cost_validation/transaction_cost_validation.json` |

The cost audit deliberately does not substitute `high_low_proxy` or `amihud`
for observed spreads. Consequently, validation against historical bid/ask or
tick data remains pending and requires a new licensed-data freeze; the V3 cost
assumption and its limitation are unchanged.

The P2 extension is archived under `runs/v3_p2_handoff/handoff_manifest.json`
with SHA-256 records. The frozen `runs/v3_final_handoff` directory was not
modified.
