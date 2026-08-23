# Phương pháp luận chi tiết: ASEAN PTCST V2.1-RC3

**Trạng thái:** protocol hiện tại đã được chạy và audit ở mức `V2.1-RC3`.  
**Phạm vi:** Indonesia, Malaysia, Philippines, Singapore và Thailand.  
**Mục tiêu:** đánh giá xem forecast cross-sectional từ PTCST, khi kết hợp với mô hình rủi ro và tối ưu danh mục có xét chi phí giao dịch, có tạo ra giá trị gia tăng so với một danh mục cùng engine nhưng không dùng forecast alpha (`risk-only`) hay không.

Tài liệu này mô tả **đúng phương pháp đã triển khai**, không mô tả một phương pháp dự kiến trong tương lai. Các kết quả 2024–2025 được xem là *development evidence*, không phải holdout cuối cùng chưa nhìn thấy.

---

## 1. Câu hỏi nghiên cứu và logic đánh giá

Phương pháp RC3 tách bài toán thành ba lớp:

1. **Forecast/ranking:** PTCST có xếp hạng cổ phiếu theo excess return tương lai tốt hơn ngẫu nhiên không?
2. **Calibration:** forecast score có thể chuyển thành expected return cùng đơn vị với covariance hay không?
3. **Portfolio attribution:** sau khi kiểm soát cùng covariance, cùng constraints và cùng transaction cost, forecast PTCST có làm danh mục tốt hơn `risk-only` không?

Việc dùng risk-only là quan trọng. Một danh mục có Sharpe cao chưa đủ chứng minh Transformer tốt: Sharpe đó có thể đến từ covariance Ledoit–Wolf, diversification, long-only constraint hoặc turnover cap. Chỉ chênh lệch **PTCST minus risk-only** mới đo đóng góp tăng thêm của forecast alpha.

---

## 2. Dữ liệu, thị trường và universe

### 2.1 Thị trường

Năm thị trường được xử lý độc lập ở tầng portfolio:

- Indonesia;
- Malaysia;
- Philippines;
- Singapore;
- Thailand.

Mô hình forecast dùng một encoder chung cho ASEAN, nhưng attention chéo cổ phiếu và việc tối ưu danh mục luôn diễn ra **trong từng country-date**, không trộn cổ phiếu giữa các nước vào cùng một portfolio.

### 2.2 Nguồn dữ liệu và đơn vị quan sát

Nguồn đầu vào đã frozen gồm dữ liệu daily từ LSEG Workspace: giá/return, volume, market capitalization, BID/ASK cuối ngày, cùng chuỗi risk-free theo thị trường. RIC là định danh chứng khoán.

Một observation forecast là một **weekly signal date × country × stock**. Mỗi stock có chuỗi đặc trưng 60 phiên trước ngày signal. Danh mục được rebalanced theo các signal weekly, nhưng P&L được tính daily theo trạng thái holdings thực tế.

### 2.3 Pure point-in-time Top-100 và variable-N

Ở mỗi weekly signal date của mỗi quốc gia:

1. Dùng market-cap ranking có độ trễ theo thời điểm có thể biết được.
2. Chỉ xét các mã thuộc Top-100 market-cap.
3. Kiểm tra feature eligibility và risk-history eligibility.
4. Giữ các mã hợp lệ; không thay mã thiếu bằng mã hạng 101 trở xuống.

Do đó số tài sản thực tế là biến thiên, ký hiệu `N_t <= 100`. Tensor vẫn có sức chứa 100 mã nhưng các ô padding được đánh dấu bởi `asset_mask` và không tham gia loss, ranking, covariance hay optimizer.

Universe này khác V1 availability-aware Top-100. RC3 dùng **pure Top-100 variable-N** để tránh việc thay thành phần universe ex post chỉ nhằm giữ đủ 100 mã.

### 2.4 Điều kiện investable

Một mã tại ngày signal chỉ được đưa vào model-ready universe khi đồng thời có:

- các feature bắt buộc: `return_60d`, `vol_60d`, `log_dollar_volume`, `log_market_cap`;
- ít nhất 126 phiên lịch sử liên tục để ước lượng risk.

Mức 126 được gọi là `risk_min_history`; covariance có thể dùng tối đa 252 phiên. Quy tắc này đồng bộ điều kiện vào universe với điều kiện rủi ro, nhằm không lặp lại tình trạng có feature nhưng không thể tạo covariance.

---

## 3. Timing, execution và target

### 3.1 Chuỗi thời điểm

Để tránh look-ahead với dữ liệu EOD, protocol khóa trình tự:

```text
close t:      quan sát feature và tạo signal
close t+1:    thực hiện lệnh/rebalance
t+2 đến t+6: return được dùng làm target 5 phiên và P&L của holdings mới
```

Weights mới không được nhận return từ close `t` đến close `t+1`, vì lệnh chỉ được thực hiện ở close `t+1`.

### 3.2 Excess return 5 phiên

Với stock `i` tại signal date `t`, raw excess return được xác định trên năm phiên sau execution:

```text
ER(i,t) = product[k=2..6] (1 + r(i,t+k))
          - product[k=2..6] (1 + rf(t+k))
```

Trong đó `r(i,t+k)` là stock return và `rf(t+k)` là risk-free return daily. Target được lưu theo basis points (bps):

```text
target_excess_return_5d_bps = 10,000 × ER(i,t)
```

Chỉ target có đủ đúng năm phiên liên tiếp mới được đánh dấu `target_available = True`; target thiếu chỉ bị mask, không làm thay đổi universe sau khi đã chọn.

### 3.3 Cross-sectional demeaned target

Mục tiêu forecast của PTCST là lựa chọn tương đối giữa các cổ phiếu, vì vậy raw excess return được demean trong cùng country-date trên universe investable:

```text
y(i,t) = ER(i,t) - mean[j in U(t)] ER(j,t)
```

`y(i,t)` là target mà mô hình học. Cách xây dựng này giảm ảnh hưởng của hướng đi chung của thị trường và đặt trọng tâm vào stock selection.

### 3.4 Temporal split và purge

| Split | Khoảng signal date | Vai trò |
|---|---|---|
| Train | 2019–2022 | Fit model và preprocessing |
| Validation | 2023 | Chọn checkpoint, ensemble và calibration |
| Development | 2024–2025 | Báo cáo bằng chứng thực nghiệm |

Nếu target của một train sample kéo dài sang 2023, sample đó bị purge. Điều kiện là:

```text
label_end(train) < 2023-01-01
```

`split_audit.csv` lưu signal date, execution date, target start/end, split và leakage flag cho từng sample. Trong dataset V2, 657 dòng chạm boundary đã được purge; không giữ lại observation leakage trong tập train.

**Giới hạn:** 2024–2025 đã được xem trong quá trình phát triển RC3. Vì không có 2026 holdout trong source, đây chưa phải final out-of-sample test hoàn toàn độc lập.

---

## 4. Đặc trưng đầu vào và preprocessing

### 4.1 Chuỗi feature của từng stock

Mỗi asset có 60 daily sessions liên tiếp, gồm 27 feature:

**Stock-level features (18):**

```text
return_1d, return_5d, return_10d, return_20d, return_60d,
vol_5d, vol_20d, vol_60d,
log_volume, log_dollar_volume, log_price, log_market_cap,
high_low_proxy, amihud, quoted_spread_bps,
day_of_week, is_month_end, is_quarter_end
```

**Market-regime features (9):**

```text
market_return_5d, market_return_20d, market_return_60d,
market_vol_20d, market_vol_60d,
cross_sectional_dispersion, market_breadth,
median_stock_vol_20d, market_distance_200d
```

Các market-regime feature được xây theo từng quốc gia từ dữ liệu quá khứ tới ngày signal, sau đó gắn vào stock-level sequence trong cùng country/date.

### 4.2 Robust scaling không leakage

Preprocessing chỉ fit trên train:

```text
median_f = median(train feature f)
IQR_f    = P75(train feature f) - P25(train feature f)
x_scaled = clip((x - median_f) / IQR_f, -10, 10)
```

Nếu feature thiếu, giá trị được thay bằng train median trước khi scale. IQR không hợp lệ hoặc quá nhỏ được thay bằng 1. Validation/development dùng đúng median và IQR từ train, không refit.

---

## 5. Mô hình ASEAN-PTCST

### 5.1 Input và output

Input tensor có dạng:

```text
B × N × 60 × 27
```

Trong đó:

- `B`: số country-date trong batch;
- `N`: tối đa 100 stocks, có asset mask;
- `60`: số phiên lookback;
- `27`: số feature.

Output là một raw score cho mỗi stock hợp lệ. Raw score chưa phải expected return; nó chủ yếu được dùng để ranking.

### 5.2 Patch temporal encoder

60 phiên được chia thành 12 patch, mỗi patch 5 phiên. Mỗi patch được flatten rồi chiếu vào vector chiều 64:

```text
patch embedding: 5 × 27 -> d_model = 64
```

Sau khi thêm temporal positional embedding, mỗi stock đi qua 2 Transformer encoder layers theo chiều thời gian:

- 4 attention heads;
- feed-forward dimension 128;
- dropout 0.10;
- pre-layer normalization.

Representation của patch cuối là summary temporal của stock đó.

### 5.3 Cross-sectional attention

Các representation stock trong cùng country-date được cộng:

- cross-sectional position embedding;
- country embedding.

Sau đó chúng đi qua 1 Transformer encoder layer theo chiều asset. `asset_mask` đảm bảo padding không ảnh hưởng attention. Vì mỗi input row chỉ chứa một country/date, cross-sectional attention không học quan hệ trực tiếp giữa cổ phiếu của các quốc gia khác nhau.

### 5.4 Country-specific prediction heads

Sau shared encoder, model dùng một prediction head riêng cho từng nước:

```text
LayerNorm -> Linear(64, 32) -> GELU -> Linear(32, 1)
```

Thiết kế này chia sẻ thông tin temporal chung ASEAN nhưng cho phép mapping từ representation sang score khác nhau theo mỗi thị trường.

---

## 6. Training protocol

### 6.1 Hybrid loss

Loss là tổ hợp giữa point prediction và ranking:

```text
L = 0.25 × L_Huber + 0.75 × L_rank
```

Trong đó:

- `L_Huber` tính trên target bps đã chia 100 để ổn định scale;
- `L_rank` là pairwise logistic ranking loss giữa các cặp stock hợp lệ trong cùng country-date:

```text
L_rank = mean softplus[-sign(y_i - y_j) × (s_i - s_j)]
```

Loss ranking có trọng số lớn hơn vì mục tiêu cuối là sắp hạng cổ phiếu để phân bổ danh mục, không chỉ giảm MAE từng stock.

### 6.2 Optimisation và checkpoint

- Optimizer: AdamW;
- learning rate: `3e-4`;
- weight decay: `1e-4`;
- batch size: 16 country-date;
- gradient clipping norm: 1.0;
- tối đa 100 epochs;
- early stopping sau 12 epochs không cải thiện;
- checkpoint selection: **validation daily Spearman IC cao nhất**.

Năm seed được chạy: `7, 19, 31, 43, 59`. Không chọn best seed dựa trên development; các seed được dùng làm input cho ensemble audit.

---

## 7. Ensemble audit và lựa chọn ensemble

### 7.1 Reconciliation và common universe

Trước khi ensemble, RC3 kiểm tra các file prediction của 5 seed:

- dates, countries, RIC grid và target phải giống nhau;
- raw score được đối chiếu trong tolerance `1e-8`;
- mỗi date chỉ dùng intersection universe giữa các seed.

Tại mỗi date, rank của từng seed được **tính lại trên chính common universe**, tránh distortion nếu một seed có số asset khác seed khác.

### 7.2 Rank normalization

Với score `s(i,t,m)` của seed `m`, model thay scale raw bằng rank-normalized Gaussian score:

```text
p(i,t,m) = (rank(s(i,t,m)) - 0.5) / N_t
z(i,t,m) = Phi_inverse(p(i,t,m))
```

Điều này giúp các seed có scale raw khác nhau vẫn đóng góp theo thứ hạng tương đối.

### 7.3 Bốn candidate được giới hạn trước

RC3 chỉ kiểm tra bốn ensemble sau:

| Candidate | Công thức |
|---|---|
| E1 | mean của 5 normalized ranks |
| E2 | median của 5 normalized ranks |
| E3 | weighted mean, weight theo validation Mean IC không âm |
| E4 | weighted mean, weight theo validation top-minus-bottom không âm |

Không dùng best seed làm final strategy.

### 7.4 Rule chọn ensemble

Candidate được chấm trên validation theo equal-country average. E3/E4 chỉ có thể thay E1 nếu đồng thời:

```text
Mean IC(candidate) >= Mean IC(E1) + 0.005
top-minus-bottom > 0
IC hit rate > 50%
max seed weight <= 50%
```

Nếu không candidate nào thỏa, giữ E1 vì đơn giản hơn. RC3 chọn **E3 IC-weighted**. Weight và configuration được ghi vào `ensemble_final_config.yaml`.

---

## 8. Calibration: score sang expected return

Raw ensemble score không được đưa trực tiếp vào MVO vì scale forecast không cùng đơn vị với covariance. Sau khi freeze ensemble, calibration được fit **chỉ trên validation** cho từng country.

Tại mỗi date, score được z-score theo cross section:

```text
z(i,t) = [S(i,t) - mean(S_t)] / SD(S_t)
```

Sau đó fit slope không intercept:

```text
y(i,t) = beta_c × z(i,t) + error
```

với `y` chuyển từ bps về decimal return. Expected alpha cho optimizer là:

```text
mu(i,t) = max(0, beta_c) × z(i,t)
```

Nếu `beta_c <= 0`, rule khóa trước yêu cầu `mu = 0`, không flip dấu. Trường hợp đó được diễn giải là market đó không có bằng chứng validation về positive usable forecast alpha. Danh mục PTCST ở market này trùng risk-only, đây là một kết quả hợp lệ chứ không phải lỗi code.

---

## 9. Risk model và cost-aware MVO

### 9.1 Ledoit–Wolf covariance

Tại mỗi signal date và mỗi quốc gia, covariance được fit bằng daily return **strictly before signal date**, trên 126 phiên lịch sử tối thiểu. Một asset chỉ có covariance hợp lệ khi đủ dữ liệu trong common history window.

Ledoit–Wolf shrinkage được dùng để covariance ổn định hơn sample covariance ở universe có số asset lớn so với số quan sát. Nếu số asset complete dưới 20, engine ghi risk fallback thay vì âm thầm dùng covariance không hợp lệ.

### 9.2 Objective function

Với weight mục tiêu `w`, weight trước trade `w_pre`, expected alpha `mu`, covariance `Sigma`, vector cost một chiều `c`, objective là:

```text
maximize:
    mu' w
    - (lambda / 2) w' Sigma w
    - c' |w - w_pre|
    - exited_cost
```

RC3 dùng:

```text
lambda = 50
turnover cap = 0.40 (full L1)
```

Lambda được chọn trên validation trước development evaluation.

### 9.3 Constraints

```text
sum(w) = 1
w_i >= 0
w_i <= max(5%, 1/N_t)
sum_i |w_i - w_pre,i| + exited_turnover <= 0.40
```

Asset không có covariance hợp lệ bị giữ tại previous weight trong optimizer. Initial allocation được gắn nhãn `initial_deployment`, không chịu turnover cap dành cho các rebalance sau; từ rebalance thứ hai cap 40% mới áp dụng.

Solver ưu tiên CLARABEL, sau đó OSQP. Nếu solver không tìm được nghiệm thì engine ghi fallback; RC3 báo solver fallback rate riêng thay vì che giấu.

### 9.4 Turnover và universe change

Turnover được báo theo full L1 trên union của holdings cũ và mới:

```text
turnover_t = continuing-name turnover
             + forced-entry turnover
             + forced-exit turnover
```

Các thành phần được lưu trong `turnover_decomposition.csv`, giúp phân biệt turnover do model đổi tỷ trọng với turnover do cổ phiếu vào/ra universe.

---

## 10. Transaction cost scenarios

Mọi cost đều là decimal return cost nhân với absolute trade. Vì vậy 10 bps = `0.001`.

| Scenario | Cost used at rebalance | Mục đích |
|---|---|---|
| C0 | fixed 10 bps mỗi đơn vị full-L1 trade | Benchmark đơn giản |
| C1 | country median historical half-spread, chỉ dùng history trước signal | Sensitivity theo thị trường |
| C2 | half-spread riêng từng stock từ quote gần nhất trước signal; thiếu quote thì dùng country median | Sensitivity thực tế hơn |

Historical BID/ASK là quote cuối ngày. C1/C2 không được diễn giải là tick-level implementation shortfall.

Chi phí portfolio mỗi ngày rebalance:

```text
cost_t = sum_i c(i,t) × |trade(i,t)| + exited_cost_t
net_return_t = gross_return_t - cost_t
```

Portfolio lưu chi tiết trong `portfolio_weights`, `portfolio_trades`, `portfolio_costs`, `daily_portfolio_returns` và `rebalance_log`.

---

## 11. Daily-state backtest

Backtest không coi mỗi forecast là một independent 5-day bet. Nó duy trì holdings liên tục:

```text
previous holdings
-> daily return và drift weights
-> rebalance ở close execution date
-> pay transaction cost
-> new holdings
-> return từ phiên kế tiếp
```

Nếu một position thiếu daily valuation, return của position đó được carry bằng 0 cho ngày đó và event được log; không xóa cả ngày portfolio. Điều này giúp evaluation coverage không bị làm đẹp bằng cách bỏ các ngày khó.

---

## 12. Benchmark risk-only và kiểm định incremental value

RC3 chạy hai strategies qua **cùng một engine** cho C0/C1/C2:

| Strategy | Alpha đưa vào optimizer |
|---|---|
| PTCST | `mu(i,t)` từ ensemble + calibration |
| Risk-only | `mu(i,t) = 0` |

Do đó:

```text
Delta Sharpe_c = Sharpe(PTCST,c) - Sharpe(risk-only,c)
Delta Return_c = mean(net return_PTCST - net return_risk-only)
```

Confidence interval của chênh lệch return và Sharpe được tính bằng paired block bootstrap theo thời gian:

- block size: 5 daily observations;
- draws: 2.000;
- resample trên paired daily PTCST/risk-only returns, không resample stock độc lập.

Nếu CI của delta Sharpe chứa 0, không khẳng định PTCST có incremental performance. Nếu beta đã bị clip về 0, PTCST và risk-only phải giống nhau; đó là kiểm tra logic của protocol.

---

## 13. Metrics đã khóa

### 13.1 Forecast metrics

Primary metrics, tính theo từng country-date rồi trung bình theo country:

1. Mean Spearman IC;
2. Median Spearman IC;
3. ICIR = Mean IC / SD(IC);
4. IC hit rate = tỷ lệ IC > 0;
5. Top-minus-bottom 5-day return.

Secondary metrics:

- Pearson IC;
- MAE bps;
- RMSE bps;
- directional accuracy.

Diagnostics:

- cross-sectional forecast dispersion;
- dispersion ratio = SD(prediction) / SD(target);
- calibration slope;
- validation decile profile.

Zero/constant forecast có ranking metrics là N/A, vì mọi asset bị tie.

### 13.2 Portfolio metrics

Với daily net return `r_t`:

```text
annualized net excess return = mean(r_t) × 252
annualized volatility        = SD(r_t) × sqrt(252)
annualized Sharpe            = annualized return / annualized volatility
```

Các metrics còn lại:

- maximum drawdown từ cumulative wealth;
- mean turnover per rebalance;
- annualized turnover;
- cumulative cost drag = tổng gross return trừ tổng net return;
- return per unit turnover;
- evaluation coverage;
- mean eligible N và mean valid-risk N;
- covariance fallback rate;
- solver fallback rate;
- missing valuation event rate per held asset-day.

### 13.3 Nguyên tắc tổng hợp ASEAN

Kết quả phải báo cáo theo từng country trước. Nếu cần ASEAN summary, dùng equal-country average, không mechanically pool mọi daily observation giữa các nước vì lịch giao dịch, liquidity, sample size và risk coverage khác nhau.

---

## 14. Audit trail và file output

RC3 sinh các artifacts chính:

```text
ensemble_input_reconciliation.csv
ensemble_universe_audit.csv
single_seed_recomputed_metrics.csv
seed_pairwise_rank_corr.csv
seed_consensus_diagnostics.csv
ensemble_validation_candidates.csv
ensemble_validation_selection.csv
ensemble_final_config.yaml
ensemble_calibration_summary.csv
ensemble_decile_validation.csv

portfolio/ptcst/{C0,C1,C2}/...
portfolio/risk_only/{C0,C1,C2}/...
portfolio_summary.csv
ensemble_portfolio_summary.csv
risk_only_summary.csv
incremental_value_summary.csv
incremental_bootstrap_ci.csv
audit_manifest.json
```

Mỗi portfolio scenario còn có ledger weights, trades, costs, daily returns, risk coverage, solver log, fallback log, turnover decomposition, split audit, universe coverage và reliability metrics.

---

## 15. Quy tắc diễn giải và giới hạn claim

RC3 chỉ cho phép các claim sau:

- Có thể nói protocol có kiểm soát timing, split purge, alignment, risk coverage, solver logging và transaction cost scenarios.
- Có thể nói risk/cost-aware engine tạo portfolio result tốt ở một số quốc gia trong sample development.
- Chỉ được nói PTCST có incremental value ở một country nếu `Delta Sharpe`, `Delta Return` và bootstrap evidence hỗ trợ so với risk-only.

Không được:

- gọi 2024–2025 là final untouched test;
- gọi Sharpe cao của risk-only là forecast success;
- chọn best seed sau khi đã thấy development result;
- flip beta âm để cứu performance;
- coi EOD BID/ASK là tick-level execution cost;
- so Sharpe V1 với RC3 như apples-to-apples khi hai engine khác nhau.

---

## 16. Trạng thái phương pháp hiện tại

`V2.1-RC3` là một **audited development protocol**. Audit kỹ thuật đã hoàn tất; kết quả economic incremental alpha có thể âm và vẫn là một finding hợp lệ. Bước hợp lệ tiếp theo, nếu muốn nâng độ mạnh nghiên cứu, là lấy một holdout 2026 chưa nhìn thấy hoặc replay V1 trên cùng RC3 engine; không tiếp tục tune theo kết quả 2024–2025.

