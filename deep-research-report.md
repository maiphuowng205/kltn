# Tối ưu hóa danh mục có xét chi phí bằng dự báo lợi suất dựa trên Transformer

## Tóm tắt điều hành và diễn giải vấn đề nghiên cứu

Đề tài **“Cost-Aware Portfolio Optimization with Transformer-Based Return Forecasting”** nghiên cứu một chuỗi quyết định gồm ba khâu khác nhau nhưng liên kết chặt chẽ: dự báo lợi suất tài sản, chuyển dự báo thành trọng số danh mục, và xác định lượng giao dịch cùng chi phí phát sinh khi chuyển từ danh mục hiện tại sang danh mục mới. Sự phân biệt này rất quan trọng: mô hình dự báo tốt chưa chắc tạo ra danh mục tốt; danh mục tối ưu trên giấy chưa chắc tốt sau chi phí; và mô hình chi phí giao dịch không phải là mô hình khớp lệnh hay mô phỏng vi cấu trúc thị trường.

Khung nghiên cứu phù hợp nhất cho một nghiên cứu sinh viên là **mô hình hai giai đoạn có kiểm soát chi phí**. Ở giai đoạn đầu, Transformer dự báo lợi suất vượt trội trong tuần kế tiếp cho nhiều cổ phiếu đồng thời. Ở giai đoạn hai, các dự báo này được đưa vào một bài toán mean–variance dài hạn, chỉ mua, có giới hạn tỷ trọng và có phạt giao dịch theo độ thay đổi trọng số. Thiết kế này cho phép đánh giá độc lập chất lượng dự báo, giá trị của bộ tối ưu hóa, và giá trị của việc xét chi phí. Đây cũng là cách khắc phục một nhược điểm phổ biến của nghiên cứu dự báo tài chính: cải thiện sai số thống kê nhưng không chứng minh được lợi ích kinh tế sau chi phí.

**Khuyến nghị cốt lõi** là sử dụng dữ liệu cổ phiếu Hoa Kỳ CRSP theo ngày, xây dựng một tập cổ phiếu luân chuyển gồm 100 cổ phiếu vốn hóa lớn và thanh khoản cao dựa trên thông tin có sẵn tại từng thời điểm, dự báo lợi suất vượt lãi suất phi rủi ro trong năm ngày giao dịch kế tiếp, tái cân bằng hàng tuần, và ước lượng hiệp phương sai bằng Ledoit–Wolf shrinkage. CRSP bao gồm dữ liệu theo ngày và tháng, corporate actions, định danh vĩnh viễn, và hơn 36.000 chứng khoán đang hoặc từng giao dịch, nên phù hợp hơn các nguồn công khai trong việc xử lý đổi mã, hủy niêm yết và survivorship bias. citeturn17search0turn17search20turn17search28

Kiến trúc đề xuất là một **Transformer hai trục gọn nhẹ**: bộ mã hóa theo thời gian lấy cảm hứng từ PatchTST để xử lý chuỗi dữ liệu của từng cổ phiếu, sau đó một lớp attention theo chiều tài sản học quan hệ chéo giữa các cổ phiếu tại cùng một ngày dự báo. PatchTST sử dụng các đoạn thời gian, hay patches, để tạo token và đạt hiệu quả tính toán tốt hơn khi chuỗi đầu vào dài; MASTER cho thấy việc luân phiên tổng hợp thông tin nội cổ phiếu và liên cổ phiếu có ý nghĩa đối với dự báo cổ phiếu. citeturn19search0turn16search24turn16search8

Bài toán danh mục chính nên là:

\[
\max_{w_t}
\left[
w_t^\top \widehat{\mu}_t
-\frac{\lambda}{2}w_t^\top \widehat{\Sigma}_t w_t
-\sum_{i=1}^{N}c_{i,t}\left|w_{i,t}-w^-_{i,t}\right|
\right],
\]

trong đó \(w^-_t\) là trọng số thực tế ngay trước khi giao dịch, sau khi danh mục cũ đã trôi theo biến động giá, chứ không đơn giản là trọng số mục tiêu tại lần tái cân bằng trước. Việc dùng \(w^-_t\) giúp tính đúng turnover và chi phí. Chi phí tỷ lệ theo giá trị giao dịch là một xấp xỉ lồi, dễ giải và phù hợp với spread cùng commission khi quy mô giao dịch không quá lớn; với giao dịch lớn, tác động thị trường thường đòi hỏi thành phần phi tuyến hoặc bậc hai. Các nghiên cứu kinh điển và hiện đại đều cho thấy khi có chi phí, chiến lược tối ưu cần điều chỉnh từ từ hoặc tạo vùng không giao dịch thay vì tái cân bằng hoàn toàn sau mọi thay đổi nhỏ của tín hiệu. citeturn20search0turn20search1turn20search2

**Câu hỏi nghiên cứu chính** nên được phát biểu như sau:

> Việc kết hợp dự báo lợi suất đa cổ phiếu bằng Transformer với tối ưu hóa mean–variance có xét chi phí giao dịch có cải thiện hiệu quả rủi ro–lợi suất thuần ngoài mẫu so với các mô hình dự báo, danh mục và cơ chế tái cân bằng tiêu chuẩn hay không?

Các câu hỏi hỗ trợ gồm:

- Transformer có tạo ra thông tin dự báo ngoài mẫu tốt hơn Ridge, XGBoost, LSTM, TCN và Transformer encoder cơ bản xét theo rank IC, Pearson IC và sai số dự báo hay không?
- Cải thiện dự báo có chuyển thành Sharpe ratio, certainty-equivalent return và alpha danh mục cao hơn hay không?
- Phạt turnover có làm giảm giao dịch và tăng lợi suất thuần, hay làm mất quá nhiều tín hiệu dự báo?
- Kết quả có bền vững trước mức chi phí, tần suất tái cân bằng, ước lượng hiệp phương sai, giới hạn tỷ trọng và chế độ thị trường khác nhau hay không?
- Mô hình hai giai đoạn có ổn định và dễ tái lập hơn mô hình end-to-end tối ưu trực tiếp mục tiêu danh mục hay không?

Ba lớp bài toán cần được định nghĩa rõ:

| Lớp | Đầu vào | Đầu ra | Tiêu chí đánh giá |
|---|---|---|---|
| Dự báo lợi suất | Lịch sử giá, lợi suất, khối lượng, thanh khoản, quy mô, biến thị trường và đặc trưng thời gian | \(\widehat{\mu}_{i,t}\), dự báo lợi suất vượt trội của từng tài sản | MAE, RMSE, IC, rank IC, directional accuracy |
| Tối ưu hóa danh mục | Dự báo lợi suất, ma trận hiệp phương sai, danh mục đang nắm giữ, ràng buộc | Trọng số mục tiêu \(w_t\) | Gross/net Sharpe, CE return, drawdown, concentration |
| Mô hình giao dịch và chi phí | Thay đổi trọng số, spread hoặc giả định chi phí, quy mô danh mục | Chi phí và lợi suất thuần | Turnover, chi phí trung bình, số giao dịch, net–gross gap |

Transformer nhận một tensor có thể biểu diễn dưới dạng \(X_t\in\mathbb{R}^{N_t\times L\times F}\), với \(N_t\) là số cổ phiếu đủ điều kiện tại thời điểm \(t\), \(L\) là số ngày lookback, và \(F\) là số đặc trưng. Mô hình tạo ra một vector dự báo \(\widehat{\mu}_t\in\mathbb{R}^{N_t}\). Bộ tối ưu hóa kết hợp vector này với \(\widehat{\Sigma}_t\), trọng số trước giao dịch và các giới hạn đầu tư để xác định trọng số mới. Framework được gọi là **cost-aware** vì chi phí hoặc giới hạn turnover xuất hiện trực tiếp trong hàm mục tiêu hoặc tập ràng buộc trước khi giao dịch được quyết định, thay vì chỉ được trừ khỏi kết quả sau khi chiến lược đã được xây dựng.

## Dữ liệu, nguồn thay thế và thiết kế mẫu thực nghiệm

### Đánh giá các nguồn dữ liệu

Không có một nguồn công khai miễn phí nào đồng thời cung cấp lịch sử cổ phiếu toàn diện, thành viên thị trường theo từng thời điểm, lợi suất hủy niêm yết, corporate actions, fundamentals point-in-time và dữ liệu spread chất lượng cao. Vì vậy, lựa chọn dữ liệu quyết định mức độ mạnh của tuyên bố nghiên cứu.

| Nguồn | Truy cập | Phạm vi, thời gian và tần suất | Biến và quy mô gần đúng |
|---|---|---|---|
| **CRSP US Stock Databases** | Tổ chức, thường qua WRDS | Cổ phiếu Hoa Kỳ; dữ liệu theo ngày và tháng; lịch sử trên 100 năm | Giá, return, volume, shares outstanding, distributions, corporate actions, identifiers; hơn 36.000 chứng khoán active và inactive. citeturn17search0turn17search24 |
| **CRSP/Compustat Merged** | Tổ chức, subscription | Doanh nghiệp Hoa Kỳ; tần suất thị trường theo ngày/tháng và kế toán theo quý/năm | Liên kết PERMNO/PERMCO với fundamentals Compustat; thích hợp cho market cap, sector và đặc trưng cơ bản point-in-time nếu xử lý ngày công bố đúng |
| **Yahoo Finance** | Công khai qua web; thư viện như `yfinance` không phải giao diện dữ liệu nghiên cứu chính thức | Nhiều quốc gia và loại tài sản; chủ yếu ngày, tuần, tháng | OHLCV, adjusted close, dividends, splits; số tài sản lớn nhưng lịch sử mã hủy niêm yết và thành viên chỉ số không đầy đủ |
| **Alpha Vantage** | API miễn phí giới hạn và gói trả phí | Cổ phiếu, ETF, FX, crypto và macro toàn cầu; ngày, tuần, tháng, intraday; nhiều endpoint có hơn 20 năm | OHLCV, adjusted close, dividends, splits, fundamentals và technical indicators. citeturn17search1turn17search25 |
| **Massive, trước đây là Polygon.io** | API/flat files; miễn phí giới hạn và subscription | Cổ phiếu Hoa Kỳ, options, FX, crypto; dữ liệu cổ phiếu từ khoảng 2003; tick, quote, minute và daily | Trades, NBBO quotes, aggregates, ticker reference, splits và dividends; hơn 32.000 ticker được nhà cung cấp nêu trên trang sản phẩm. citeturn17search2 |
| **Nasdaq Data Link WIKI Prices** | Công khai/lịch sử | Khoảng 3.000 công ty Hoa Kỳ; daily; kết thúc ngày 11-4-2018 | Giá, dividends và splits; không còn cập nhật, do đó chỉ phù hợp cho replication lịch sử. citeturn18search3 |
| **Kenneth French Data Library** | Công khai | Hoa Kỳ và quốc tế; daily, weekly, monthly; dữ liệu Hoa Kỳ từ 1926 | Market, SMB, HML, RMW, CMA, momentum, RF và các danh mục nghiên cứu; không phải dữ liệu từng cổ phiếu. citeturn18search0turn18search4turn18search22 |
| **FRED/ALFRED** | Công khai qua web/API | Hàng trăm nghìn chuỗi kinh tế Hoa Kỳ và quốc tế; daily đến annual | Lãi suất, credit spreads, sản lượng, lạm phát, lao động, liquidity; ALFRED/vintage dates hỗ trợ tránh dùng dữ liệu đã sửa đổi sau này. citeturn18search1turn18search5turn18search11 |
| **LSEG Tick History/Refinitiv** | Subscription tổ chức | Hơn 30 năm dữ liệu intraday, khoảng 580+ venues, nhiều asset classes | Trades, quotes và market depth; rất mạnh cho spread và market-impact calibration nhưng chi phí và dung lượng cao. citeturn18search2turn18search18 |
| **Bloomberg** | Terminal hoặc enterprise subscription | Thị trường toàn cầu, đa tài sản; intraday đến annual | Giá, fundamentals, corporate actions, estimates, classifications và analytics; chất lượng cao nhưng khó tái lập nếu người đọc không có giấy phép |
| **Kaggle** | Công khai hoặc theo điều kiện của người đăng | Phụ thuộc từng bộ dữ liệu | Thường là snapshot từ Yahoo, exchanges hoặc vendor; provenance, cập nhật, survivorship và điều chỉnh corporate actions không đồng nhất |
| **Dữ liệu sở giao dịch chính thức** | Công khai một phần hoặc trả phí | Phụ thuộc sở và quốc gia | Danh sách niêm yết, giá, announcements, sometimes order book; thường phải ghép nhiều bảng và có quy định cấp phép riêng |

Đặc điểm điều chỉnh, chi phí và thiên lệch của các nguồn quan trọng được tóm tắt như sau:

| Nguồn | Corporate actions và delisting | Dữ liệu chi phí | Điểm mạnh | Hạn chế chính |
|---|---|---|---|---|
| CRSP | Corporate actions, distributions, lịch sử tên/mã và chứng khoán inactive; có thể kết hợp return với delisting return | Có một số trường quote/bid–ask tùy giai đoạn và phiên bản; có thể tạo proxy từ giá, volume và liquidity | Point-in-time tốt, định danh vĩnh viễn, giảm survivorship bias, chuẩn phổ biến trong nghiên cứu tài chính | Subscription; cần hiểu share codes, exchange codes, delisting và thay đổi phiên bản dữ liệu |
| CRSP/Compustat | Tốt nếu liên kết bằng CCM và lag theo ngày công bố | Không trực tiếp cung cấp execution cost đầy đủ | Cho phép kết hợp kỹ thuật, quy mô và fundamentals | Fundamentals dễ gây look-ahead nếu dùng fiscal date thay vì filing/publication date |
| Yahoo Finance | Adjusted close thường phản ánh splits/dividends đối với mã đang có | Không có spread lịch sử đáng tin cậy trong gói cơ bản | Dễ sử dụng và tái lập về mặt mã nguồn | Hủy niêm yết, đổi mã, lịch sử constituents và revisions khó kiểm soát |
| Alpha Vantage | Endpoint adjusted có adjusted close, dividends và split events | Không có market-impact chi tiết; intraday/quote tùy gói | API chính thức, tài liệu rõ, phù hợp đồ án quy mô nhỏ | Rate limit, chi phí nếu tải nhiều mã, point-in-time universe và delisting chưa tương đương CRSP |
| Massive | Có reference endpoints, splits và dividends | Trades và historical quotes cho phép tính spread ở gói phù hợp | Tốt nhất trong nhóm API bán lẻ cho nghiên cứu spread và intraday | Subscription và khối lượng dữ liệu; cần tự xây dựng universe và kiểm tra adjustment |
| Nasdaq WIKI | Có splits/dividends trong lịch sử | Không có quote/spread | Dễ tái lập các nghiên cứu trước 2018 | Đã ngừng cập nhật và không phù hợp cho kiểm định giai đoạn gần đây |
| French Library | Total-return factor portfolios; có archives theo data vintage | Không có chi phí từng cổ phiếu | RF, benchmark và factor alpha công khai | Không dùng để huấn luyện mô hình cổ phiếu riêng lẻ |
| FRED/ALFRED | Revisions được quản lý qua vintages | Không áp dụng | Macro và lãi suất miễn phí; ALFRED hỗ trợ point-in-time | Tần suất thấp, release lag và revision phải được xử lý rõ |
| LSEG | Điều chỉnh và reference data cấp tổ chức | Quotes, depth và trades phù hợp transaction-cost analysis | Phạm vi và độ sâu rất cao | Chi phí, licensing, engineering và reproducibility thấp đối với sinh viên |
| Bloomberg | Corporate actions và reference data mạnh | Spread, volume và liquidity tùy sản phẩm | Tích hợp đa nguồn | Không công khai; khó chia sẻ dữ liệu và pipeline hoàn chỉnh |

### Dữ liệu chính và phương án công khai

**Dữ liệu chính được khuyến nghị:** CRSP US Stock Database theo ngày qua WRDS, kết hợp Kenneth French daily factors và RF. Nếu giấy phép cho phép, có thể bổ sung Compustat/CCM cho sector, market cap và fundamentals, nhưng fundamentals không nên là điều kiện bắt buộc của mô hình chính. CRSP đặc biệt phù hợp vì định danh PERMNO/PERMCO tồn tại xuyên qua đổi ticker, có chứng khoán inactive, lịch sử corporate actions và độ dài mẫu vượt xa thời gian cần thiết cho walk-forward learning. citeturn17search0turn17search28

**Phương án công khai khả thi:** Alpha Vantage adjusted daily cho một tập cố định gồm khoảng 50–100 cổ phiếu thanh khoản cao đã được xác định bằng thông tin tại ngày bắt đầu nghiên cứu, kết hợp French factors và FRED/ALFRED. Phương án này tái lập được nhưng không thể được mô tả là hoàn toàn không có survivorship bias nếu tập mã được chọn từ danh sách hiện tại. Cần công bố danh sách mã, ngày tải dữ liệu, dữ liệu thô được lưu lại, tiêu chí chọn tại đầu mẫu và mọi mã không thể truy xuất. Alpha Vantage cung cấp OHLCV điều chỉnh, dividends và split events với lịch sử dài, nhưng việc tái tạo universe lịch sử và delisted stocks vẫn kém CRSP. citeturn17search1turn17search13

Nếu quyền truy cập Massive/Polygon cho phép tải historical quotes, phương án công khai/trả phí thấp có thể được nâng cấp bằng cách tính half-spread theo từng tài sản và thời điểm. Điều này cải thiện mô hình chi phí nhưng không tự động giải quyết survivorship bias; universe point-in-time vẫn phải được xây dựng riêng. citeturn17search2

### Thiết kế mẫu thực nghiệm được đề xuất

| Thành phần | Khuyến nghị |
|---|---|
| Thị trường | Cổ phiếu phổ thông Hoa Kỳ trên NYSE, NYSE American/AMEX và Nasdaq |
| Khoảng dữ liệu thô | 03-01-2000 đến 31-12-2025 |
| Universe | 100 cổ phiếu đủ điều kiện có vốn hóa thị trường trễ lớn nhất tại đầu mỗi tháng |
| Loại chứng khoán | CRSP share codes 10 và 11; loại ADR, closed-end fund, ETF, preferred stock, warrants và units |
| Tần suất dữ liệu | Đặc trưng theo ngày |
| Chân trời dự báo | Năm phiên giao dịch, close-to-close sau một ngày trễ triển khai |
| Tái cân bằng | Hàng tuần |
| Lookback của Transformer | 60 ngày; robustness 20 và 120 ngày |
| Cửa sổ rủi ro | 252 ngày giao dịch để ước lượng covariance; robustness 126 và 504 ngày |
| Train ban đầu | 2000–2014 |
| Validation phát triển | 2015–2018 |
| Test hoàn toàn ngoài mẫu | 2019–2025 |
| Cập nhật mô hình | Retrain theo quý; tạo forecast hàng tuần |
| Thanh khoản tối thiểu | Median dollar volume trong 60 ngày trước ít nhất 10 triệu USD/ngày |
| Giá tối thiểu | Giá đóng cửa trễ ít nhất 5 USD |
| Lịch sử tối thiểu | Ít nhất 252 quan sát trước ngày đủ điều kiện |
| Benchmark | CRSP value-weighted market index; S&P 500 total return là benchmark bổ sung |
| Lãi suất phi rủi ro | Daily RF từ Kenneth French, compounded theo tuần |
| Constraints chính | Long-only, fully invested, tối đa 5% mỗi cổ phiếu, không leverage |
| Chi phí cơ sở khi thiếu spread | 10 basis points cho mỗi USD được giao dịch; robustness 5–50 bps |
| Covariance | Rolling Ledoit–Wolf shrinkage trên 252 daily returns |
| Mô hình tối ưu | Cost-aware mean–variance |

Khoảng test 2019–2025 chứa các chế độ rất khác nhau, gồm cú sốc COVID-19, giai đoạn phục hồi nhanh, lạm phát và thắt chặt tiền tệ. Điều này không làm test trở thành bằng chứng đầy đủ cho mọi regime, nhưng giúp tránh một test period chỉ gồm một thị trường tăng ổn định. Việc dừng cuối năm 2025 thay vì dùng tám tháng đầu năm 2026 tránh một năm quan sát chưa hoàn chỉnh và cho phép dùng các block theo năm trong kiểm định robustness.

Universe nên được cập nhật mỗi tháng bằng market capitalization tính từ dữ liệu có sẵn ở cuối tháng trước. Tại mỗi ngày tái cân bằng, mô hình chỉ có thể giao dịch các cổ phiếu đã vượt điều kiện niêm yết, giá, lịch sử và thanh khoản trước đó. Cổ phiếu mới niêm yết chỉ được nhận vào sau khi có đủ 252 phiên. Cổ phiếu bị hủy niêm yết phải giữ trong dữ liệu đến ngày cuối cùng và lợi suất cuối cùng phải phản ánh delisting return nếu có. Việc loại chúng khỏi lịch sử sau khi biết kết cục tạo survivorship bias; nghiên cứu về CRSP đã chỉ ra rằng bỏ qua hủy niêm yết có thể làm sai lệch kết quả chiến lược. citeturn17search0turn17search16

Trong bốn cách xây dựng universe, thứ tự ưu tiên là:

| Phương án | Đánh giá |
|---|---|
| Rolling universe dựa trên thông tin trễ | **Tốt nhất:** phản ánh investable universe, cho phép entries và exits, kiểm soát survivorship tốt |
| Historical constituents tại từng ngày | Tốt nếu lịch sử constituents đáng tin cậy và có effective dates |
| Fixed universe được chọn tại đầu kỳ bằng thông tin lúc đó | Khả thi nhưng bỏ qua công ty xuất hiện sau này và có thể tạo selection bias |
| Current constituents áp dụng ngược về quá khứ | **Không nên dùng:** loại các doanh nghiệp thất bại hoặc rời chỉ số và tạo survivorship/look-ahead bias |

Missing values ngắn trong đặc trưng có thể được forward-fill tối đa một hoặc hai ngày đối với biến chậm thay đổi, nhưng không được forward-fill returns hay volume qua thời gian ngừng giao dịch dài. Quan sát không có giá hợp lệ vào ngày thực thi nên không được giao dịch; nếu đang nắm giữ một cổ phiếu bị đình chỉ, trọng số của nó phải được giữ hoặc xử lý theo quy tắc liquidation thực tế thay vì giả định bán không chi phí.

Lợi suất nên được tính từ total-return series:

\[
r_{i,t}=\frac{P^{adj}_{i,t}}{P^{adj}_{i,t-1}}-1,
\]

hoặc dùng trực tiếp CRSP return đã phản ánh distributions. Target năm ngày là:

\[
y_{i,t}^{(5)}
=
\prod_{h=1}^{5}(1+r_{i,t+h})-
\prod_{h=1}^{5}(1+r_{f,t+h}).
\]

Không nên trừ đơn giản \(5r_f\) khi tần suất và mức lãi suất thay đổi; compounded risk-free return nhất quán hơn. French Library công bố daily RF dựa trên one-month Treasury bill rate và cung cấp total-return market factors, phù hợp cho bước này. citeturn18search7turn18search4

Các biến macro chỉ được bổ sung nếu pipeline dùng ngày công bố thực tế hoặc ALFRED vintage. Ví dụ, giá trị CPI được công bố vào giữa tháng không thể được gán cho toàn bộ tháng đó; dữ liệu FRED hiện tại đã qua revision cũng không được xem như dữ liệu nhà đầu tư biết trong quá khứ. FRED API cho phép truy xuất vintage dates và observations như tồn tại tại ngày lịch sử. citeturn18search1turn18search5turn18search8

## Kiến trúc và phương pháp dự báo Transformer

### So sánh các kiến trúc phù hợp

Transformer ban đầu thay thế recurrence bằng self-attention và feed-forward layers, cho phép mỗi token tham chiếu trực tiếp đến các token khác trong chuỗi. Tuy nhiên, attention chuẩn có chi phí bộ nhớ và tính toán bậc hai theo số token, trong khi lợi suất tài chính có signal-to-noise thấp, non-stationarity cao và ít quan sát độc lập hơn dữ liệu ngôn ngữ. Vì vậy, việc dùng Transformer lớn nguyên bản thường không hợp lý cho một nghiên cứu sinh viên. citeturn11search10

| Kiến trúc | Cách biểu diễn và quan hệ thời gian | Quan hệ giữa tài sản | Ưu điểm cho tài chính | Hạn chế và tính toán |
|---|---|---|---|---|
| Original Transformer | Mỗi thời điểm là token; positional encoding; full self-attention | Chỉ có nếu đưa các tài sản vào chung một sequence hoặc feature vector | Baseline chuẩn, dễ giải thích | \(O(L^2)\); có thể overfit; không thiết kế riêng cho time series. citeturn11search10 |
| Transformer encoder | Chỉ dùng encoder và prediction head | Có thể áp dụng riêng từng tài sản hoặc trên panel | Đơn giản, dễ triển khai, baseline cần thiết | Quan hệ chéo không tự động xuất hiện nếu train từng mã |
| Temporal Fusion Transformer | Kết hợp recurrent local processing, gated residual networks, variable selection và interpretable attention; hỗ trợ multi-horizon | Biến tĩnh và time-varying covariates được tích hợp nhưng không chuyên biệt cho hàng trăm cổ phiếu | Tốt khi có nhiều covariates, missingness và nhu cầu diễn giải | Phức tạp, nhiều thành phần, nặng hơn nhu cầu dự báo một horizon. citeturn10search3turn10search21 |
| Informer | ProbSparse attention và cơ chế decoder cho long-sequence forecasting | Giống Transformer đa biến thông thường | Hiệu quả với chuỗi rất dài | Lợi thế hạn chế khi lookback chỉ 60–120 ngày; thiết kế thiên về long-horizon. citeturn11search0 |
| Autoformer | Decomposition trend/seasonal và Auto-Correlation ở mức sub-series | Có thể xử lý multivariate series | Phù hợp chuỗi có chu kỳ và mùa vụ rõ | Lợi suất cổ phiếu có periodicity yếu; decomposition có thể không mang lại lợi ích. citeturn19search3 |
| FEDformer | Series decomposition kết hợp biểu diễn miền tần số Fourier/wavelet | Multivariate forecasting | Giảm tính toán và lọc nhiễu tần số | Thêm hyperparameters; bằng chứng gốc chủ yếu trên energy, traffic, weather và benchmark phi tài chính. citeturn19search1 |
| PatchTST | Chia từng chuỗi thành patches; channel-independent temporal encoder | Bản gốc không chủ động attention giữa channels | Hiệu quả, mã nguồn công khai, giảm số token và hỗ trợ lookback dài | Channel independence có thể bỏ lỡ quan hệ giữa cổ phiếu. citeturn19search0 |
| iTransformer | Đảo cách token hóa: mỗi biến là token, lịch sử của biến tạo embedding | Mạnh trong học quan hệ giữa variables | Hấp dẫn cho panel đa biến | Nếu mỗi cổ phiếu và mỗi feature là một biến, số token và universe thay đổi trở nên khó quản lý. citeturn19search2 |
| ALSP-TF | Attention thích ứng với short/long patterns; graph self-supervised regularization | Học quan hệ tập thể giữa cổ phiếu | Thiết kế trực tiếp cho stock ranking | Pipeline và graph regularization phức tạp; thí nghiệm gốc dùng top-\(k\) trading, không phải cost-aware MVO. citeturn14view1turn15view0 |
| MASTER | Luân phiên intra-stock và inter-stock aggregation; market-guided feature selection | Mô hình hóa momentary và cross-time stock correlations | Phù hợp trực tiếp với multi-asset return forecasting; có code chính thức | Phức tạp hơn PatchTST; bằng chứng gốc chủ yếu trên CSI300/CSI800. citeturn16search4turn16search8 |

ALSP-TF được kiểm định trên 1.026 cổ phiếu Nasdaq, 1.737 cổ phiếu NYSE và 95 cổ phiếu TSE, sử dụng dữ liệu ngày và lookback 16 ngày; nghiên cứu đánh giá chiến lược chọn cổ phiếu bằng Sharpe ratio, cumulative investment return và nDCG. Đây là bằng chứng đáng chú ý rằng attention có thể học đồng thời short–long patterns và quan hệ chéo, nhưng chiến lược thử nghiệm không mô hình hóa một bộ tối ưu mean–variance có chi phí. citeturn15view0turn15view1

MASTER sử dụng CSI300 và CSI800, với dữ liệu ngày từ 2008 đến 2022, và luân phiên aggregation nội cổ phiếu và liên cổ phiếu. Mô hình này liên quan trực tiếp hơn nhiều Transformer forecasting papers chỉ dự báo một chỉ số đơn lẻ, nhưng vẫn tập trung chủ yếu vào hiệu năng dự báo và stock selection thay vì kiểm định cost-aware portfolio optimization. citeturn16search0turn16search4turn16search24

### Kiến trúc được khuyến nghị

Mô hình chính nên là **Patch-based Temporal and Cross-Sectional Transformer**, viết tắt có thể là PTCST, với bốn khối:

1. Mỗi cổ phiếu có chuỗi \(L=60\) ngày và \(F\) đặc trưng.
2. Chuỗi được chia thành patches năm ngày, stride năm ngày; mỗi patch được chiếu sang embedding 64 chiều.
3. Một temporal Transformer encoder dùng chung tham số xử lý từng cổ phiếu.
4. Embedding cuối của 100 cổ phiếu được đưa qua một hoặc hai lớp cross-sectional attention, sau đó prediction head tạo một lợi suất dự báo cho từng cổ phiếu.

Đây là **suy luận thiết kế** từ PatchTST và MASTER, không phải tuyên bố rằng kiến trúc kết hợp này đã được chứng minh vượt trội. Patching giảm số temporal tokens; parameter sharing giúp tăng số mẫu hiệu dụng; lớp cross-sectional attention khắc phục hạn chế channel-independent; và số lớp nhỏ giữ chi phí tính toán trong phạm vi một GPU phổ thông. citeturn19search0turn16search24

Tensor đầu vào có dạng:

\[
X_t =
\left\{x_{i,t-L+1:t}\right\}_{i=1}^{N_t},
\qquad
X_t\in\mathbb{R}^{N_t\times L\times F}.
\]

Để xử lý universe thay đổi, mỗi batch được tạo theo ngày dự báo, dùng mask cho vị trí không đủ điều kiện. Không nên gán một embedding học được duy nhất cho mỗi ticker rồi kỳ vọng mô hình hoạt động với cổ phiếu mới; nên ưu tiên sector embedding, exchange embedding, size bucket và feature-based representation. Stock-ID embedding có thể được thử trong robustness nhưng có nguy cơ ghi nhớ doanh nghiệp.

### Đặc trưng đầu vào

Tập đặc trưng chính nên giới hạn khoảng 20–40 biến, tránh hàng trăm chỉ báo kỹ thuật tương quan cao:

| Nhóm | Biến đề xuất |
|---|---|
| Lợi suất | Daily total return; cumulative returns 5, 10, 20 và 60 ngày; overnight return nếu có open |
| Rủi ro | Rolling volatility 5, 20 và 60 ngày; downside volatility; high–low range |
| Khối lượng và thanh khoản | Log volume, log dollar volume, turnover ratio, volume change, zero-volume flag, Amihud-style price-impact proxy |
| Quy mô và giá | Log market cap, log price, share turnover |
| Thị trường | Market return, market volatility, cross-sectional dispersion, advance–decline proxy |
| Tương đối | Return trừ market, return trừ sector, rolling beta, residual momentum |
| Lịch | Day-of-week, month-end, quarter-end và holiday proximity |
| Macro tùy chọn | Treasury rates, term spread, credit spread, VIX-like volatility proxy; chỉ dùng dữ liệu point-in-time |

Fundamentals nên được để cho một extension. Nếu đưa vào, mọi biến phải có effective date bằng filing/publication date cộng một lag bảo thủ, không phải fiscal quarter end. Một nghiên cứu chính dựa trên giá, volume, market cap và market variables sẽ dễ kiểm toán hơn và giảm nguy cơ leakage.

### Target dự báo

| Target | Ưu điểm | Hạn chế đối với danh mục |
|---|---|---|
| Raw future return | Trực tiếp, có đơn vị phù hợp cho MVO | Bao gồm risk-free component và regime market chung |
| Excess return trên RF | Phù hợp quyết định phân bổ giữa risky assets và cash; giữ magnitude | Vẫn nhiễu và nhạy với outliers |
| Market-adjusted return | Tập trung vào stock-selection alpha | MVO cần cộng lại market component nếu mục tiêu là total return |
| Cross-sectional rank | Ổn định hơn outliers; phù hợp top-\(k\) | Không cung cấp magnitude để tối ưu mean–variance |
| Direction | Dễ đánh giá classification | Bỏ qua độ lớn lợi suất; xác suất tăng không đủ xác định expected return |
| Volatility-scaled return | Cân bằng tài sản có volatility khác nhau | Cần chuyển đổi ngược về đơn vị lợi suất |
| Quantile/distribution forecast | Phản ánh uncertainty và downside | Kiến trúc và optimizer phức tạp hơn; cần nhiều dữ liệu |

**Target chính được khuyến nghị** là lợi suất total return vượt RF trong năm ngày giao dịch kế tiếp, đo bằng basis points. Target được winsorize theo ngưỡng chỉ ước lượng trên training data, chẳng hạn phần vị 0,5% và 99,5%, hoặc dùng Huber loss để giảm ảnh hưởng extreme observations mà không trực tiếp cắt dữ liệu test.

Pure rank target không nên là target chính vì mean–variance optimization cần một vector kỳ vọng có ý nghĩa về độ lớn. Tuy nhiên, rank IC nên là forecast metric chính bổ sung, và một mô hình rank-only nên xuất hiện trong robustness. Một phương án nâng cao là huấn luyện bằng loss kết hợp:

\[
\mathcal{L}
=
\mathcal{L}_{Huber}(y,\widehat{y})
+
\eta\,\mathcal{L}_{rank}(y,\widehat{y}),
\]

với \(\eta\) nhỏ và được chọn hoàn toàn trên validation period. Nếu rank loss làm pipeline quá phức tạp, Huber loss đơn thuần là lựa chọn chính dễ bảo vệ hơn.

### Huấn luyện và tuning

| Thành phần | Thiết lập ban đầu |
|---|---|
| Lookback | 60 ngày |
| Patch length/stride | 5/5 |
| \(d_{\text{model}}\) | 64 |
| Temporal layers | 2 |
| Cross-sectional layers | 1 |
| Attention heads | 4 |
| Feed-forward width | 128 hoặc 256 |
| Dropout | 0,10–0,20 |
| Optimizer | AdamW |
| Learning rate | \(3\times10^{-4}\) hoặc \(10^{-3}\), tuned |
| Weight decay | \(10^{-5}\) đến \(10^{-3}\) |
| Batch | 16–32 forecast dates |
| Loss | Huber |
| Max epochs | 100 |
| Early stopping | Validation rank IC hoặc validation Huber loss, patience 10 |
| Gradient clipping | 1,0 |
| Retraining | Hàng quý |
| Random seeds | Ít nhất 5 cho mô hình deep learning |

Normalization phải được fit chỉ bằng dữ liệu quá khứ. Đối với return và volume features, robust rolling normalization bằng median và interquartile range thường an toàn hơn global mean–standard deviation. Cross-sectional z-score tại mỗi ngày có thể được dùng cho market cap, liquidity và relative momentum, nhưng mọi statistic phải được tính từ các cổ phiếu đủ điều kiện tại ngày đó.

Hyperparameter tuning nên dùng rolling validation hoặc blocked time-series validation, không dùng random \(k\)-fold. Ngân sách tuning cần tương đương tương đối giữa các mô hình: cùng số trials, cùng thời gian lịch sử và cùng thông tin đầu vào. Chọn mô hình theo một tiêu chí được xác định trước, chẳng hạn validation rank IC trung bình với điều kiện MAE không suy giảm nghiêm trọng, thay vì thử nhiều metric rồi chọn metric làm mô hình đề xuất trông tốt nhất.

## Mô hình tối ưu hóa danh mục có xét chi phí

### Mean–variance không có chi phí

Bài toán cơ sở là:

\[
\max_{w_t}
\quad
w_t^\top \widehat{\mu}_t
-\frac{\lambda}{2}w_t^\top\widehat{\Sigma}_t w_t,
\]

với:

\[
\mathbf{1}^\top w_t=1,\qquad
0\leq w_{i,t}\leq w_{\max}.
\]

Markowitz cung cấp nền tảng cho việc cân bằng expected return và variance, nhưng nghiệm mean–variance rất nhạy với sai số expected return và covariance. Bằng chứng ngoài mẫu của DeMiguel, Garlappi và Uppal cho thấy nhiều chiến lược tối ưu hóa phức tạp không nhất quán vượt qua \(1/N\) về Sharpe, certainty-equivalent return và turnover, nhấn mạnh việc bắt buộc phải có equal-weight benchmark mạnh. citeturn21search3

### Mean–variance có chi phí

Mô hình chính là:

\[
\max_{w_t}
\quad
w_t^\top\widehat{\mu}_t
-\frac{\lambda}{2}w_t^\top\widehat{\Sigma}_t w_t
-\gamma
\sum_{i=1}^{N_t}
c_{i,t}\left|w_{i,t}-w^-_{i,t}\right|.
\]

Ở đây:

- \(w^-_{i,t}\) là trọng số trước giao dịch;
- \(\Delta w_{i,t}=w_{i,t}-w^-_{i,t}\);
- \(c_{i,t}\) là chi phí cho một đơn vị giá trị được mua hoặc bán;
- \(\gamma=1\) nếu \(c_{i,t}\) đã được đo trực tiếp bằng đơn vị lợi suất;
- \(\lambda\) là mức e ngại rủi ro.

Nếu \(\gamma\) được tune tự do trong khi \(c\) đã là estimate chi phí kinh tế, mô hình có thể “giảm nhẹ” chi phí để tối ưu validation Sharpe. Thiết kế trong sạch hơn là đặt \(\gamma=1\), tune \(\lambda\), rồi kiểm định robustness bằng cách nhân \(c\) với các hệ số bảo thủ khác nhau.

Trọng số trước giao dịch được tính từ trọng số mục tiêu trước đó và lợi suất tài sản:

\[
w^-_{i,t}
=
\frac{w_{i,t-1}(1+r_{i,t-1\rightarrow t})}
{\sum_jw_{j,t-1}(1+r_{j,t-1\rightarrow t})}.
\]

Bỏ qua bước drift này thường đánh giá sai lượng giao dịch.

### So sánh các mô hình chi phí

| Mô hình | Giả định | Ưu điểm | Hạn chế | Vai trò đề xuất |
|---|---|---|---|---|
| \(c\sum_i|\Delta w_i|\) | Mọi tài sản có cùng proportional cost | Lồi, đơn giản, dễ tái lập; cần ít dữ liệu | Không phản ánh khác biệt thanh khoản hoặc regime | Mô hình chính khi không có spread |
| \(\sum_i c_{i,t}|\Delta w_i|\) | Mỗi tài sản/thời điểm có cost khác nhau | Phản ánh spread và liquidity; vẫn lồi | Proxy cost có thể nhiễu; cần point-in-time quote | Mô hình chính nếu CRSP quote hoặc Massive data đủ tốt |
| \(\sum_i q_{i,t}(\Delta w_i)^2\) | Market impact tăng theo kích thước giao dịch | Trơn, dễ differentiation; hợp với large orders | Cần AUM, ADV và calibration; under/overstate small-trade spread | Robustness hoặc extension |
| \(\sum_i|\Delta w_i|\leq\tau\) | Ngân sách turnover cố định | Minh bạch; ngăn optimizer phản ứng quá mức | Không quy đổi thành chi phí tiền; cùng \(\tau\) có ý nghĩa khác nhau theo liquidity | Ràng buộc bổ sung |
| No-trade region | Không giao dịch khi lợi ích kỳ vọng chưa vượt chi phí | Nền tảng kinh tế mạnh; giảm churn | Khó hiệu chỉnh nhiều tài sản; phụ thuộc đường đi và thresholds | Robustness |
| Threshold rebalancing | Chỉ giao dịch khi \(|w_i^*-w_i^-|>\delta\) | Dễ triển khai thực tế | Không nhất thiết globally optimal; threshold tùy ý | Rule-based comparator |

Với proportional transaction costs, các mô hình continuous-time kinh điển tạo ra một vùng không giao dịch, trong đó nhà đầu tư giữ nguyên vị thế cho tới khi danh mục chạm biên. Gârleanu và Pedersen cho thấy với predictable returns và trading costs, vị thế tối ưu điều chỉnh dần theo tín hiệu thay vì nhảy ngay đến frictionless target. Những kết quả này hỗ trợ việc dùng turnover penalty hoặc no-trade robustness, dù nghiên cứu sinh viên không cần giải một bài toán stochastic control liên tục. citeturn20search0turn20search1turn20search12

**Khuyến nghị khi không có spread chi tiết:** proportional cost không đổi \(c=10\) bps trên mỗi USD giao dịch, cộng turnover cap. Mức 10 bps không được trình bày như một estimate “thật” cho toàn thị trường; đó là giả định cơ sở phải đi cùng sensitivity 5, 10, 20, 30 và 50 bps.

**Khuyến nghị khi có quote:** dùng rolling median của half quoted spread trong 20 phiên:

\[
c_{i,t}
=
\operatorname{median}_{s=t-20}^{t-1}
\left[
\frac{Ask_{i,s}-Bid_{i,s}}
{Ask_{i,s}+Bid_{i,s}}
\right]
+c^{commission}.
\]

Spread phải bị lag và winsorize, vì stale hoặc erroneous quotes có thể tạo estimate cực đoan. Market impact không nên được khẳng định là đã mô hình hóa nếu chỉ có spread.

Hai quy ước turnover cần được báo cáo song song:

\[
TO^{L1}_t=\sum_i|\Delta w_{i,t}|,
\qquad
TO^{one-way}_t=\frac{1}{2}\sum_i|\Delta w_{i,t}|.
\]

Trong danh mục fully invested không có cash flow, tổng lượng mua bằng tổng lượng bán, nên one-way turnover bằng nửa \(L_1\). Công thức chi phí phải nêu rõ \(c\) áp dụng cho tổng absolute traded value hay one-way turnover để tránh hệ số hai.

### Ràng buộc danh mục

| Ràng buộc | Khuyến nghị và lý do |
|---|---|
| Fully invested | \(\sum_i w_i=1\); giữ cash ngoài danh mục chính chỉ khi nghiên cứu có quyết định phân bổ cash rõ |
| Long-only | \(w_i\geq0\); giảm nhu cầu borrow cost và phù hợp nghiên cứu sinh viên |
| Maximum weight | \(w_i\leq5\%\) với 100 cổ phiếu; robustness 2% và 10% |
| Leverage | Không leverage trong mô hình chính |
| Sector | Không quá trọng số sector benchmark cộng 10 điểm phần trăm, hoặc hard cap 30%; dùng sector data point-in-time |
| Turnover | \(TO^{L1}\leq40\%\) mỗi tuần làm giới hạn bảo vệ; tune trên validation |
| Minimum holding | Không bắt buộc vì tạo mixed-integer problem; có thể dùng threshold rule |
| Short selling | Chỉ trong robustness nếu có borrow-cost assumption |
| Cash | 0% trong mô hình chính; 0–5% trong robustness |
| Number of holdings | Không ép cardinality trong mô hình chính; concentration được kiểm soát bằng max-weight |

Ràng buộc sector giúp ngăn mô hình biến thành một bet ngành tập trung do một regime ngắn. Tuy nhiên, benchmark-relative sector constraints đòi hỏi historical sector mapping; nếu dữ liệu này không point-in-time, hard cap sector đơn giản và công bố rõ sẽ an toàn hơn.

### Ước lượng covariance

| Phương pháp | Ưu điểm | Hạn chế |
|---|---|---|
| Sample covariance | Minh bạch, không tham số | Nhiễu, ill-conditioned khi \(N\) lớn so với cửa sổ |
| Rolling sample covariance | Thích ứng theo thời gian | Vẫn nhiễu; thay đổi mạnh khi cửa sổ dịch chuyển |
| EWMA | Trọng số lớn cho quan sát gần; phản ứng regime | Decay parameter nhạy; vẫn có thể không ổn định |
| Ledoit–Wolf shrinkage | Co covariance mẫu về một target có cấu trúc; cải thiện conditioning | Có thể làm mất một số structure động |
| Factor covariance | Giảm chiều, dễ giải thích exposure | Phụ thuộc factor specification và estimation |
| Deep covariance forecast | Có thể học nonlinear dynamics | Tăng mạnh degrees of freedom, leakage và overfitting |

**Lựa chọn chính:** Ledoit–Wolf shrinkage trên 252 daily returns. Công trình Ledoit–Wolf đề xuất shrinkage để cải thiện ma trận covariance mẫu trong bối cảnh nhiều tài sản; điều này đặc biệt phù hợp khi optimizer cần nghịch đảo hoặc khai thác các eigenvalues nhỏ. citeturn20search3

**Robustness:** EWMA covariance với half-life 60 ngày; sample covariance; và factor covariance đơn giản gồm market plus sector factors. Deep covariance forecasting không nên nằm trong mô hình chính vì sẽ làm khó xác định cải thiện đến từ return Transformer hay risk model.

Risk aversion \(\lambda\) nên được chọn trên validation để đạt target volatility, chẳng hạn 10–15% annualized, hoặc tối đa hóa net certainty-equivalent return:

\[
CE =
\overline{r}^{net}
-\frac{a}{2}\operatorname{Var}(r^{net}),
\]

với \(a\) được xác định trước. Chọn \(\lambda\) bằng test Sharpe sẽ tạo hyperparameter leakage.

### Hai giai đoạn so với end-to-end

| Tiêu chí | Hai giai đoạn | End-to-end |
|---|---|---|
| Interpretability | Cao: tách forecast, risk và optimization | Thấp hơn; khó biết mô hình học alpha hay trực tiếp học exposure |
| Forecast evaluation | Có MAE, IC và calibration riêng | Có thể không tạo forecast có ý nghĩa |
| Implementation | CVXPY hoặc solver lồi độc lập với PyTorch | Cần differentiable optimizer hoặc parameterization trọng số |
| Overfitting | Thấp hơn, dù vẫn đáng kể | Cao hơn do mục tiêu Sharpe/utility nhiễu và path-dependent |
| Transaction costs | Dễ đưa vào optimizer và backtest | Cần differentiable approximation cho absolute turnover hoặc solver layer |
| Reproducibility | Tốt hơn | Nhạy với initialization, batch path và solver |
| Academic contribution | Rõ ràng qua ablations | Có thể mới hơn nhưng khó chẩn đoán |
| Computation | Trung bình | Cao |
| Constraints | Giải chính xác bằng convex optimization | Có thể cần projection hoặc optimization layer |

Nghiên cứu end-to-end cho risk budgeting của Uysal, Li và Mulvey kết hợp dự báo và tối ưu trong một mạng thống nhất và báo cáo kết quả ngoài mẫu trên các ETF lớn giai đoạn 2017–2021; phiên bản model-based nhúng risk-budgeting layer vào mạng. Nghiên cứu này chứng minh end-to-end có thể hoạt động, nhưng không cho thấy mọi bài toán cổ phiếu với expected-return forecasting và turnover đều nên được huấn luyện chung. citeturn14view2

**Khuyến nghị:** dùng thiết kế hai giai đoạn cho nghiên cứu chính; end-to-end cost-adjusted utility chỉ là extension. Thiết kế chính nên đủ mạnh để trả lời liệu Transformer có thêm thông tin dự báo và liệu cost-aware optimizer có biến thông tin đó thành lợi ích kinh tế. Nếu gộp hai khâu ngay từ đầu, một kết quả tốt hoặc xấu rất khó quy cho forecast, covariance, constraints hay portfolio loss.

## Tổng quan nghiên cứu liên quan và khoảng trống

### Transformer cho dự báo tài chính

| Nghiên cứu và trạng thái | Dữ liệu, bài toán | Mô hình, danh mục và chi phí | Baselines, metrics và phát hiện | Hạn chế và liên quan | DOI/nguồn |
|---|---|---|---|---|---|
| Wang et al. (2022), **peer-reviewed, IJCAI**, “Adaptive Long-Short Pattern Transformer for Stock Investment Selection” | Daily OHLCV; Nasdaq 1.026 cổ phiếu, NYSE 1.737, TSE 95; train/validation/test khoảng 2013–2020 tùy thị trường; stock ranking | ALSP-TF với pattern distiller, time-adaptive attention và graph regularization; top-\(k\) buy-hold-sell; không có cost-aware MVO | So sánh ARIMA, RNN, graph, RL, ranking và Transformer; SR, cumulative return, nDCG; paper báo cáo ALSP-TF tốt hơn baselines | Không tính realistic costs; sample và quy tắc portfolio khác đề tài này; rất liên quan đến cross-sectional attention | 10.24963/ijcai.2022/551. citeturn14view1turn15view0 |
| Li et al. (2024), **peer-reviewed, AAAI**, “MASTER: Market-Guided Stock Transformer for Stock Price Forecasting” | CSI300 và CSI800; daily 2008–2022; multi-stock forecasting | Market-guided feature selection; intra/inter-stock aggregation; không phải mean–variance, không cost-aware | So với temporal, relational và Transformer stock models; paper báo cáo ưu thế forecast và learned correlations | China-only; economic evaluation chưa tách forecast/optimizer/cost | 10.1609/aaai.v38i1.27767. citeturn13search0turn16search4 |
| Xiao, Hua, và Qin (2024), **peer-reviewed, Finance Research Letters**, “A Self-Attention Based Cross-Sectional Return Forecasting Model with Evidence from the Chinese Market” | Cross-sectional Chinese stock returns; out-of-sample forecasting | Self-attention học nonlinearities, heterogeneity và stock interactions; long–short application | Out-of-sample \(R^2\) và long–short profitability được báo cáo tốt hơn benchmarks | Publisher abstract không cho thấy transaction-cost-aware optimization; cần kiểm tra full text về constituents và point-in-time controls | 10.1016/j.frl.2024.105144. citeturn16search1turn16search5 |
| Ma, Wang, và Chen (2023), **peer-reviewed, International Review of Financial Analysis**, “Attention Is All You Need: An Interpretable Transformer-Based Asset Allocation Approach” | Multi-asset allocation; paper tập trung trực tiếp vào phân bổ | Interpretable Transformer-based allocation | So sánh các phương pháp asset allocation và nghiên cứu attention interpretation | Gần đề tài về Transformer allocation, nhưng không thay thế kiểm định forecast-plus-MVO có chi phí; chi tiết dataset cần xác nhận từ full text | 10.1016/j.irfa.2023.102876. citeturn16search6turn16search25 |
| Fan et al. (2024), **peer-reviewed, AAAI**, “StockMixer” | Multi-stock price/return forecasting | MLP mixer, không phải Transformer | Được đề xuất như kiến trúc đơn giản nhưng mạnh | Quan trọng như strong non-Transformer baseline, tránh chỉ so Transformer với ARIMA/LSTM yếu | Official AAAI page. citeturn13search8turn13search14 |

Các nghiên cứu gần đây về cross-sectional forecasting thường cho thấy attention có thể mô hình hóa tương tác giữa cổ phiếu, nhưng evaluation hay dựa trên forecast metrics, top-\(k\) long–short hoặc simplified profitability. Việc thiếu mean–variance optimizer không phải là khuyết điểm đối với mục tiêu gốc của các paper đó, nhưng tạo không gian cho nghiên cứu kiểm tra liệu tín hiệu attention có còn giá trị khi đi qua covariance estimation, long-only constraints, turnover penalty và costs.

Các preprint 2025–2026 như *Asset Pricing in Transformer*, các nghiên cứu stock-ranking loss và các khung predict-then-optimize cho thấy xu hướng chuyển từ price forecasting sang cross-sectional expected returns và decision-focused objectives. Tuy nhiên, chúng cần được ghi rõ là preprint hoặc conference version, và trạng thái peer review phải được kiểm tra lại tại thời điểm nộp luận văn. citeturn16search6turn6search11turn1search8turn1search22

### Deep learning và dự báo kết hợp tối ưu hóa

| Nghiên cứu và trạng thái | Bài toán và dữ liệu | Forecast/allocation/cost | Kết quả và giới hạn | Liên quan |
|---|---|---|---|---|
| Ma, Han, và Wang (2021), **peer-reviewed, Expert Systems with Applications** | Dự báo return để preselect stocks, sau đó portfolio optimization | RF, SVR, LSTM, DMLP, CNN; mean–variance và Omega portfolios | Paper báo cáo prediction-based portfolios có thể cải thiện allocation; không phải Transformer và không tập trung cost-aware turnover | Tiền lệ trực tiếp cho forecast-then-optimize; baseline quan trọng. citeturn16search3turn16search7 |
| Uysal, Li, và Mulvey (2024), **peer-reviewed, Annals of Operations Research** | ETF indices; out-of-sample 2017–2021 | Model-free và model-based end-to-end risk budgeting; Sharpe training objective | Paper báo cáo model-based approach ổn định hơn pure risk budgeting trong thí nghiệm; dataset công khai nhưng code chỉ theo yêu cầu | Bằng chứng mạnh nhất cho end-to-end differentiable allocation, nhưng khác expected-return MVO cổ phiếu. citeturn14view2 |
| Li và Mulvey (2021), **peer-reviewed, INFORMS Journal on Optimization** | Portfolio optimization dưới regime switching | Neural networks kết hợp dynamic programming; transaction costs | Kết nối neural learning, regime và chi phí trong dynamic allocation | Gần Group B/C nhưng không phải Transformer cross-sectional forecasting. citeturn14view2 |
| Zhang, Zohren, và Roberts (2021), **preprint**, “A Universal End-to-End Approach to Portfolio Optimization via Deep Learning” | General portfolio learning | Direct portfolio weights hoặc distribution; portfolio-level losses | Cung cấp framework linh hoạt nhưng kết quả nhạy với objective, constraints và data-generating process | Phù hợp làm extension, không nên là main baseline duy nhất. citeturn7search0 |
| Martínez-Barbero et al. (2025), **peer-reviewed, Computational Economics**, “Portfolio Optimization with Prediction-Based Return Using Machine Learning” | Prediction-based return và portfolio optimization | Machine-learning forecasts đưa vào optimizer | Củng cố nhu cầu đánh giá dự báo ở cấp danh mục | Cần so sánh với Transformer và thêm cost/turnover ablation. citeturn16search27 |

Các nghiên cứu forecast-then-optimize cho thấy kết quả tốt nhất không nhất thiết đến từ mô hình deep nhất. Ma, Han và Wang kết hợp nhiều mô hình ML/DL với mean–variance và Omega optimization, trong khi bằng chứng benchmark rộng hơn cho thấy estimation error có thể làm \(1/N\) khó bị vượt ngoài mẫu. Do đó, XGBoost, Ridge, minimum variance và equal weight là các đối thủ thực chất, không phải baselines mang tính hình thức. citeturn16search3turn21search3

### Tối ưu hóa có transaction costs

| Nghiên cứu và trạng thái | Mô hình | Chi phí và phát hiện lý thuyết/thực nghiệm | Liên quan |
|---|---|---|---|
| Davis và Norman (1990), **peer-reviewed, Mathematics of Operations Research** | Dynamic consumption–investment với một risky asset và bank account | Proportional costs dẫn đến wedge-shaped no-transaction region | Nền tảng lý thuyết cho threshold/no-trade robustness. citeturn20search0 |
| Gârleanu và Pedersen (2013), **peer-reviewed, Journal of Finance** | Dynamic trading với predictable returns | Quadratic trading costs; optimal positions điều chỉnh dần theo predictors và decay | Biện minh cho turnover penalty và gradual rebalancing. citeturn20search1 |
| Li et al. (2018), **peer-reviewed, Quantitative Finance** | Online portfolio selection | Tích hợp transaction costs vào online allocation | Cho thấy costs nên xuất hiện trong decision rule, không chỉ post-hoc evaluation. citeturn8search10turn8search20 |
| Qiao et al. (2023), **peer-reviewed, Journal of Empirical Finance** | Time-varying volatility và portfolio selection | Kết hợp volatility động với transaction costs | Liên quan đến robustness dùng EWMA/time-varying covariance. citeturn8search6 |
| Ledoit và Wolf (2025), **peer-reviewed, Quarterly Review of Economics and Finance** | Markowitz portfolios under transaction costs | Nghiên cứu trực tiếp tác động của costs lên Markowitz allocation | Bằng chứng gần nhất cho cost-aware MVO, nhưng không có Transformer forecasts. citeturn20search2turn20search14 |

### Khoảng trống nghiên cứu

Tổng hợp tài liệu cho thấy các khoảng trống sau có mức độ hỗ trợ khác nhau:

| Khoảng trống đề xuất | Đánh giá |
|---|---|
| Transformer papers chỉ đánh giá forecast accuracy | **Có một phần.** ALSP-TF, MASTER và Xiao et al. có economic hoặc ranking evaluation, nên không đúng nếu nói toàn bộ chỉ báo cáo MSE. Tuy nhiên, ít nghiên cứu đưa forecasts vào constrained cost-aware MVO. |
| Portfolio-learning studies bỏ transaction costs | **Khá rõ.** Một số có costs, nhưng nhiều paper allocation báo cáo gross performance hoặc simplified rebalancing. |
| Báo cáo gross thay vì net returns | **Phổ biến nhưng cần kiểm tra từng paper.** Không nên gán thiếu sót này cho paper nếu full text chưa xác nhận. |
| Transformer chỉ được so với baseline yếu | **Có rủi ro rõ.** Một nghiên cứu mới nên có XGBoost, Ridge, LSTM/TCN và PatchTST, không chỉ ARIMA và vanilla RNN. |
| Bỏ survivorship/look-ahead bias | **Khó xác nhận từ abstract.** Đây là khoảng trống về reporting và empirical discipline hơn là tuyên bố mọi paper đều sai. |
| Forecast improvements không chuyển thành economic gains | **Có cơ sở mạnh.** Mean forecasts rất nhiễu và optimization khuếch đại estimation error; \(1/N\) thường cạnh tranh ngoài mẫu. citeturn21search3 |
| Costs được giả định thay vì ước lượng | **Khá rõ trong nghiên cứu dùng daily public data.** Quote/depth data thường không có hoặc đắt. |
| Temporal và cross-sectional relations không được mô hình hóa đồng thời | **Đang được khắc phục** bởi MASTER, ALSP-TF và self-attention cross-sectional models; không còn là khoảng trống hoàn toàn. |
| Tập trung Hoa Kỳ | **Không đúng đối với Transformer stock forecasting**, vì nhiều paper quan trọng dùng Trung Quốc, NYSE/Nasdaq và Nhật Bản. |
| Thiếu regime robustness | **Có khả năng cao.** Nhiều test periods ngắn hoặc một thị trường; cần kiểm định theo volatility và drawdown regimes. |

**Đóng góp mạnh và dễ bảo vệ nhất** không phải là “Transformer đầu tiên cho portfolio optimization.” Tài liệu đã có Transformer allocation, end-to-end portfolio learning và transaction-cost portfolio models. Đóng góp nên được phát biểu thận trọng:

> Tài liệu được xác định dường như cung cấp còn hạn chế bằng chứng về việc kết hợp đồng thời dự báo lợi suất chéo và theo thời gian bằng Transformer, một optimizer mean–variance minh bạch, transaction costs được tính tại từng lần tái cân bằng, và một backtest point-in-time kiểm soát survivorship, leakage và turnover. Một đóng góp tiềm năng là xác định liệu lợi ích dự báo của Transformer có tồn tại sau covariance estimation, portfolio constraints và realistic cost sensitivity hay không.

Đóng góp này dựa trên **tính tích hợp và chất lượng nhận dạng thực nghiệm**, không nhất thiết dựa trên một kiến trúc neural hoàn toàn mới.

## Baselines, backtest, đánh giá và kiểm định thống kê

### Forecasting baselines

Một danh sách quá dài làm tăng multiple testing và giảm khả năng tuning công bằng. Nên triển khai các sanity checks đơn giản trước, sau đó giữ tám mô hình chính.

| Mô hình | Vai trò | Hyperparameters chính | Target |
|---|---|---|---|
| Historical mean | Null forecast và sanity check | Cửa sổ 20, 60, 252 ngày | Weekly excess return |
| AR/Ridge pooled | Baseline tuyến tính mạnh, kiểm soát shrinkage | Lags, \(L_2\) penalty | Weekly excess return |
| XGBoost | Strong nonlinear tabular baseline | Depth 2–6, learning rate, estimators, subsample, column sampling | Weekly excess return |
| MLP | Kiểm tra giá trị của sequence modeling | 1–3 layers, width, dropout, weight decay | Weekly excess return |
| LSTM | Baseline recurrent chuẩn | Hidden size, layers, dropout, lookback | Weekly excess return |
| TCN | Baseline convolutional sequence mạnh | Channels, kernel, dilation, dropout | Weekly excess return |
| Vanilla Transformer encoder | Tách giá trị Transformer chuẩn khỏi thiết kế mới | Layers, heads, \(d_{\text{model}}\), dropout | Weekly excess return |
| PatchTST | Modern time-series Transformer baseline | Patch length, stride, layers, heads | Weekly excess return |
| PTCST đề xuất | Patch temporal encoder và cross-sectional attention | Các tham số trên cộng cross-sectional layers | Weekly excess return |

Để giữ nghiên cứu khả thi, **tập cuối cùng gồm tám mô hình** có thể bỏ MLP hoặc gộp historical mean và AR/Ridge thành nhóm statistical. Khuyến nghị cuối là historical mean, Ridge, XGBoost, LSTM, TCN, vanilla Transformer, PatchTST và PTCST.

Previous-period return, moving average, ARIMA và exponential smoothing nên được chạy như diagnostic baselines, nhưng không nhất thiết xuất hiện trong mọi bảng chính. Exponential smoothing phù hợp hơn với level/trend series; đối với gần-white-noise return, historical mean hoặc zero forecast thường là null mạnh hơn. ARIMA được phép chọn order nhỏ trên validation, không dùng automated search trên test.

Mọi mô hình phải:

- sử dụng cùng feature availability và target definition;
- nhận cùng train/validation/test dates;
- không dùng random shuffling qua thời gian;
- được tune với ngân sách gần tương đương;
- được retrain theo cùng lịch hàng quý;
- có ít nhất năm seeds nếu stochastic;
- báo cáo mean và dispersion giữa seeds.

### Portfolio baselines và ablation

| Danh mục | Forecast Transformer | Optimizer | Cost-aware decision | Mục đích |
|---|---:|---:|---:|---|
| Equal weight rebalanced | Không | Không | Không | Benchmark \(1/N\) mạnh |
| Equal-weight buy-and-hold | Không | Không | Không | Tách tác động của rebalancing |
| Value-weighted | Không | Không | Không | Market-like benchmark |
| Minimum variance, LW | Không | Có | Không | Kiểm tra giá trị covariance không cần mean forecast |
| Historical-mean MVO | Không | Có | Không | Classical MVO |
| XGBoost MVO | Không | Có | Không | Strong ML forecast plus optimizer |
| LSTM MVO | Không | Có | Không | Deep sequential comparison |
| Transformer top-\(k\) | Có | Không | Không | Tách optimizer khỏi forecast |
| Transformer MVO | Có | Có | Không | Giá trị forecast cộng optimizer, gross/cost-unaware |
| Cost-aware XGBoost MVO | Không | Có | Có | Tách giá trị Transformer khỏi cost model |
| Cost-aware Transformer MVO | Có | Có | Có | Mô hình đề xuất |
| Risk parity | Không | Có | Có hoặc không | Supplemental risk-based benchmark |
| Maximum diversification | Không | Có | Có hoặc không | Supplemental |
| Black–Litterman | Tùy | Có | Tùy | Chỉ thêm nếu có cách chuyển forecast thành views đáng tin cậy |

Bộ core nên gồm equal weight, value weight, minimum variance, historical-mean MVO, XGBoost MVO, Transformer top-\(k\), Transformer MVO, cost-aware XGBoost MVO và cost-aware Transformer MVO. Risk parity, maximum diversification và Black–Litterman có thể đưa vào phụ lục để tránh làm bảng chính quá rộng.

Các so sánh nhận dạng giả thuyết:

| Giả thuyết | So sánh |
|---|---|
| Transformer có forecast tốt hơn | PTCST so với Ridge, XGBoost, LSTM, TCN, vanilla Transformer và PatchTST |
| Transformer tạo economic value | Transformer MVO so với XGBoost MVO và LSTM MVO với cùng covariance/constraints |
| Optimizer tạo giá trị | Transformer MVO so với Transformer top-\(k\) |
| Cost awareness tạo giá trị | Cost-aware Transformer MVO so với Transformer MVO sau khi cả hai bị trừ cùng realized costs |
| Kết quả không chỉ do cost penalty | Cost-aware Transformer MVO so với cost-aware XGBoost MVO |
| Complexity đáng giá | Cost-aware Transformer MVO so với equal weight và minimum variance |

Một nguyên tắc quan trọng là **cost-unaware strategy vẫn phải bị trừ chi phí thực tế trong evaluation**. “Cost-unaware” chỉ có nghĩa optimizer không xét chi phí khi chọn trọng số, không có nghĩa backtest được miễn chi phí. Nếu chỉ trừ costs cho mô hình cost-aware, phép so sánh bị sai.

### Timeline walk-forward

Quy trình tuần \(k\) nên được cố định:

1. **Sau close ngày \(t\):** thu thập tất cả dữ liệu được công bố không muộn hơn \(t\).
2. **Sau close \(t\):** chuẩn hóa bằng statistics quá khứ và tạo dự báo cho lợi suất từ close \(t+1\) đến close \(t+6\).
3. **Sau close \(t\):** ước lượng covariance và giải optimizer.
4. **Tại close \(t+1\):** thực hiện trọng số mục tiêu bằng dữ liệu giá \(t+1\), không dùng lợi suất ngày \(t+1\) để tạo signal.
5. **Tại \(t+1\):** trừ transaction cost theo thay đổi từ pre-trade weights sang target weights.
6. **Từ close \(t+1\) đến close \(t+6\):** ghi nhận holding-period return.
7. **Lặp lại:** cập nhật pre-trade weights sau drift.

Việc đặt một ngày trễ triển khai loại bỏ giả định phi thực tế rằng mô hình có thể dùng close \(t\), huấn luyện, tối ưu và giao dịch chính xác tại cùng close đó. Nếu nghiên cứu có daily open data đáng tin cậy, có thể thực thi tại open \(t+1\); quy tắc phải được thống nhất giữa label và backtest.

Chronological split chính là train 2000–2014, validation 2015–2018, test 2019–2025. Trong test, mô hình được phép retrain theo quý bằng tất cả dữ liệu quá khứ, nhưng kiến trúc, feature list, cost rule, hyperparameter ranges và model-selection criterion phải bị khóa sau validation.

Một phương án mạnh hơn là nested walk-forward:

- training window mười năm hoặc expanding;
- validation hai năm;
- test một năm;
- cuộn theo năm;
- gộp các out-of-sample test blocks sau khi mọi lựa chọn được xác định.

Phương án này cho nhiều forecast origins và giảm phụ thuộc vào một split, nhưng tốn tính toán hơn. Có thể dùng split cố định làm main result và rolling annual folds làm robustness.

### Ngăn thiên lệch và leakage

| Rủi ro | Biện pháp |
|---|---|
| Look-ahead bias | Lag mọi feature; thực thi một ngày sau signal; dùng publication/effective dates |
| Survivorship bias | Rolling CRSP universe có active/inactive securities; không dùng current index members ngược thời gian |
| Delisting bias | Giữ cổ phiếu đến ngày hủy niêm yết và dùng delisting return |
| Normalization leakage | Fit scalers trên training history hoặc rolling past-only window |
| Overlapping-label leakage | Dùng weekly non-overlapping forecast origins; nếu daily origins thì purge ít nhất năm ngày quanh split |
| Fundamental leakage | Dùng filing date và reporting lag, không dùng fiscal period end |
| Macro revision leakage | Dùng ALFRED vintages |
| Hyperparameter overfitting | Chỉ tune trên validation; khóa protocol trước final test |
| Selection bias | Định nghĩa universe và liquidity thresholds trước khi xem test results |
| Execution bias | Dùng next-day execution, không same-close; trừ costs tại mọi rebalance |
| Multiple-testing bias | Hạn chế model count; Reality Check/SPA và report toàn bộ variants |
| Seed selection | Báo cáo trung bình và độ phân tán, không chọn seed tốt nhất |

### Forecast metrics

| Metric | Vai trò |
|---|---|
| MAE | Dễ diễn giải, ít nhạy outliers hơn MSE |
| RMSE | Phạt mạnh sai số lớn |
| Huber loss | Phù hợp training khi returns có tails |
| Directional accuracy | Đánh giá đúng dấu nhưng bỏ qua magnitude |
| Pearson IC | Correlation tuyến tính giữa forecast và realized returns theo cross-section |
| Spearman rank IC | Đánh giá ranking, ít nhạy scale và outliers |
| IC information ratio | Mean IC chia standard deviation theo thời gian |
| Top-minus-bottom spread | Đo giá trị kinh tế thô của ranking |
| Calibration slope | Hồi quy realized return lên predicted return để kiểm tra magnitude |

**Forecast metrics chính:** mean weekly Spearman rank IC và MAE/Huber loss. **Metrics phụ:** Pearson IC, directional accuracy, ICIR và top-minus-bottom spread.

Prediction error thấp không đảm bảo portfolio tốt vì mô hình dự báo gần bằng zero có thể đạt RMSE thấp nhưng không xếp hạng được cổ phiếu. Ngược lại, một mô hình có rank IC tốt nhưng magnitude calibration kém có thể làm MVO tạo weights quá cực đoan. Portfolio performance còn phụ thuộc covariance, constraints, turnover và correlation của errors giữa tài sản.

### Portfolio metrics

| Nhóm | Metrics |
|---|---|
| Lợi suất | Annualized gross return, annualized net return, cumulative wealth |
| Rủi ro | Annualized volatility, downside volatility, maximum drawdown, VaR, CVaR |
| Risk-adjusted | Net Sharpe, Sortino, Calmar, certainty-equivalent return |
| Giao dịch | \(L_1\) turnover, one-way turnover, average trades, cost drag, holding duration |
| Cấu trúc | Number of active holdings, HHI, maximum weight, effective number of assets |
| Exposure | Sector weights, market beta, size/value/momentum exposures |
| Benchmark-relative | Tracking error, information ratio, alpha và factor-adjusted alpha |

**Primary portfolio metrics:** net Sharpe ratio, net certainty-equivalent return và turnover. Ba metric này trực tiếp phản ánh mục tiêu return–risk–cost.

**Secondary metrics:** net annualized return, maximum drawdown, Sortino, Calmar, CVaR, cost drag, concentration và factor alpha. Gross results phải được báo cáo cạnh net results để cho thấy phần hiệu năng bị mất do trading.

Sharpe phải được tính từ weekly excess returns và annualize nhất quán:

\[
SR_{ann}=
\sqrt{52}\frac{\overline{r_p-r_f}}
{s(r_p-r_f)}.
\]

Không annualize trung bình và volatility từ tần suất khác nhau, và không dùng daily observations được nội suy từ weekly portfolio decisions như các quan sát độc lập.

### Kiểm định thống kê

Diebold–Mariano kiểm định null rằng hai forecast có expected loss bằng nhau. Trong panel cổ phiếu, nên tính average loss differential theo từng forecast date, tạo một time series tuần, rồi dùng DM/HAC hoặc Harvey–Leybourne–Newbold small-sample adjustment. Không nên coi hàng trăm cổ phiếu cùng ngày là hàng trăm quan sát độc lập. citeturn21search4turn21search24

Đối với rank IC, Sharpe, CE return và drawdown, block bootstrap hoặc stationary bootstrap phù hợp hơn bootstrap từng tuần độc lập vì portfolio returns và forecast performance có serial dependence. Block length có thể được chọn trong khoảng 4–12 tuần và kiểm tra độ nhạy.

Kiểm định đề xuất gồm:

| Kiểm định | Ứng dụng |
|---|---|
| Diebold–Mariano/HLN | MAE hoặc squared-error difference giữa forecast models |
| Paired block bootstrap | Chênh lệch rank IC, net return, Sharpe, CE và turnover |
| Jobson–Korkie hoặc Ledoit–Wolf Sharpe comparison | Kết quả phụ; cần nêu assumptions |
| White Reality Check | Kiểm soát data snooping khi chọn strategy tốt nhất trong nhiều variants. citeturn21search1turn21search21 |
| Hansen Superior Predictive Ability | Multiple-model predictive-performance test, thường mạnh hơn Reality Check với poor alternatives |
| Factor alpha regression | Hồi quy net excess return lên market, SMB, HML, RMW, CMA và momentum, dùng Newey–West errors |
| Subperiod interaction regression | Kiểm tra performance khác nhau theo high-volatility, bear-market và rate regimes |

P-values không nên là tiêu chí duy nhất. Một chênh lệch Sharpe có ý nghĩa thống kê nhưng cost savings rất nhỏ hoặc danh mục quá tập trung có thể không có ý nghĩa kinh tế; ngược lại, mẫu test bảy năm có thể thiếu power đối với một chiến lược có effect size vừa phải.

### Robustness checks

| Chiều robustness | Thiết lập |
|---|---|
| Transaction costs | 0, 5, 10, 20, 30, 50 bps; constant và asset-specific |
| Rebalancing | Hàng tuần, hai tuần, hàng tháng |
| Lookback | 20, 60, 120 ngày |
| Universe size | 50, 100, 200 cổ phiếu |
| Universe construction | Rolling universe, fixed-at-start, historical index constituents nếu có |
| Liquidity | Dollar-volume thresholds 5, 10, 25 triệu USD |
| Covariance | Sample, EWMA, Ledoit–Wolf, factor covariance |
| Risk aversion | Grid tạo target volatility khoảng 8%, 12%, 16% |
| Maximum weight | 2%, 5%, 10% |
| Turnover cap | Không cap, 20%, 40%, 80% \(L_1\) mỗi tuần |
| Market regimes | Bull, bear, high-volatility, low-volatility, pre/post-COVID |
| Training window | Expanding và rolling mười năm |
| Target | Excess return, market-adjusted return, volatility-scaled return, rank |
| Execution | Next close và next open nếu dữ liệu có |
| Seeds | Ít nhất năm |
| Gross/net | Báo cáo cả hai, cùng cost accounting |
| Cost model | Linear, linear plus turnover cap, quadratic robustness, no-trade threshold |

Robustness không nên được dùng để chọn một cấu hình tốt nhất sau khi xem test. Kết quả chính phải dựa trên cấu hình được pre-specified; robustness cho biết dấu và mức độ của hiệu ứng có ổn định hay không.

## Thiết kế nghiên cứu cuối cùng, lộ trình và rủi ro

### Thiết kế cuối cùng được khuyến nghị

| Thành phần | Quyết định cuối cùng |
|---|---|
| Dataset | CRSP US Stock Database daily qua WRDS; Kenneth French daily factors/RF; optional Compustat sector data |
| Asset universe | Rolling top 100 common stocks theo lagged market cap, sau price/liquidity/history filters |
| Study period | 03-01-2000 đến 31-12-2025 |
| Train/validation/test | 2000–2014 / 2015–2018 / 2019–2025 |
| Frequency | Daily features, weekly forecasts và rebalancing |
| Forecast target | Năm ngày total excess return trên RF |
| Lookback | 60 ngày |
| Transformer | Hai temporal PatchTST-style encoder layers và một cross-sectional attention layer |
| Loss | Huber; auxiliary rank loss chỉ trong extension |
| Forecast baselines | Historical mean, Ridge, XGBoost, LSTM, TCN, vanilla Transformer, PatchTST |
| Portfolio optimizer | Long-only cost-aware mean–variance |
| Covariance | Rolling 252-day Ledoit–Wolf shrinkage |
| Cost function | \(\sum_i c_{i,t}|\Delta w_i|\); asset-specific half-spread nếu đủ dữ liệu, nếu không constant 10 bps |
| Constraints | Fully invested, long-only, max 5%, sector cap, no leverage, \(L_1\) turnover cap 40% |
| Execution | Signal sau close \(t\), trade tại close \(t+1\), hold năm phiên |
| Retraining | Hàng quý |
| Validation | Blocked chronological validation; no random shuffle |
| Primary forecast metrics | Spearman rank IC và MAE |
| Primary portfolio metrics | Net Sharpe, net CE return và turnover |
| Main statistical tests | DM/HLN, paired block bootstrap, SPA/Reality Check, factor alpha |
| Main robustness | Cost, frequency, lookback, universe size, covariance, risk aversion, max weight, regimes |

Mô hình hai giai đoạn là lựa chọn tốt nhất cho nghiên cứu đầu tiên. Forecast quality có thể được kiểm tra trước khi portfolio optimization; optimizer có nghiệm lồi và constraints rõ; transaction cost accounting có thể được kiểm toán; ablations trực tiếp nhận dạng ba nguồn giá trị: forecast, optimization và cost awareness.

End-to-end model chỉ nên được thêm sau khi pipeline chính hoàn thành. Extension hợp lý là huấn luyện prediction head thông qua một differentiable mean–variance layer bằng net utility:

\[
\mathcal{L}_{E2E}
=
-\left[
r_{p,t}^{net}
-\frac{a}{2}(r_{p,t}^{net})^2
\right],
\]

hoặc negative net Sharpe trên mini-batches theo thời gian. Tuy nhiên, batch Sharpe là một objective nhiễu, non-additive và phụ thuộc sampling; absolute turnover cũng không trơn tại zero. Vì vậy, kết quả end-to-end nên được xem như exploratory, không thay thế main two-stage evidence.

### Lộ trình triển khai

| Giai đoạn | Đầu việc và deliverable |
|---|---|
| Data audit | Tải dữ liệu thô; kiểm tra identifiers, corporate actions, delistings, missingness; lưu data vintage và data dictionary |
| Universe engine | Viết point-in-time screening; xuất danh sách eligible assets cho từng tháng; kiểm tra entries/exits |
| Backtest kernel | Xây dựng pre-trade drift, transaction-cost accounting, next-day execution và benchmark portfolios trước khi dùng ML |
| Statistical baselines | Historical mean, Ridge, AR diagnostic và XGBoost; xác nhận labels, metrics và walk-forward split |
| Risk engine | Ledoit–Wolf covariance, minimum variance và historical-mean MVO; kiểm tra solver failures |
| Deep baselines | LSTM, TCN, vanilla Transformer và PatchTST với cùng feature pipeline |
| Proposed model | Thêm cross-sectional attention, masking và sector embeddings; ablation temporal-only/cross-sectional-only |
| Cost-aware optimizer | Constant cost, asset-specific cost nếu có, turnover cap và drift-aware trading |
| Final walk-forward | Khóa protocol; chạy test một lần; lưu mọi forecasts, weights, trades và costs |
| Statistical analysis | Bootstrap, DM, alpha regressions, regime splits và multiple-testing controls |
| Reproducibility package | Configuration files, seeds, environment, data-access instructions, pseudocode và tables generated from stored outputs |

Mỗi forecast, target weight, executed weight, pre-trade weight, realized return và transaction cost nên được lưu theo ngày và PERMNO. Nếu chỉ lưu cumulative wealth, không thể kiểm toán turnover, execution lag hoặc exposure.

### Rủi ro và hạn chế

**Expected-return noise** là rủi ro lớn nhất. Mean–variance weights nhạy hơn nhiều với mean forecasts so với covariance estimates. Max-weight constraints, shrinkage covariance, validation-based risk aversion và forecast calibration là các biện pháp giảm rủi ro, nhưng không loại bỏ nó. Bằng chứng \(1/N\) khó bị đánh bại ngoài mẫu là lý do equal weight phải được xem là benchmark chính. citeturn21search3turn20search3

**Chi phí được giả định** không tương đương transaction-cost estimate. Constant 10 bps chỉ cung cấp stress scenario. Ngay cả quoted spread cũng không phản ánh hoàn toàn slippage, market impact, queue position và trade timing. Kết luận nên dùng cụm “net of assumed proportional costs” nếu không có actual execution data.

**Daily data không đủ để mô hình hóa market impact chính xác.** Quadratic cost yêu cầu portfolio AUM, participation rate, intraday volume profile và order execution rule. Không nên tuyên bố framework mô phỏng trade execution nếu chỉ trừ một cost scalar.

**Universe top 100 làm giảm microcap frictions nhưng giới hạn external validity.** Kết quả không thể tự động áp dụng cho toàn bộ CRSP universe, small caps hoặc emerging markets. Robustness với 50/200 cổ phiếu và liquidity thresholds khác nhau giúp đánh giá phạm vi nhưng không loại bỏ hạn chế.

**Cross-sectional attention có thể học market và sector exposures thay vì stock-specific alpha.** Cần báo cáo beta, sector weights và factor regressions. Nếu portfolio vượt benchmark chỉ nhờ beta cao hơn trong bull market, đó không phải bằng chứng mạnh cho stock-selection alpha.

**Non-stationarity và regime dependence** khiến một mô hình tốt trong 2019–2021 có thể không tốt trong 2022–2025. Regime breakdown và rolling performance phải được báo cáo, không chỉ một Sharpe toàn kỳ.

**Compute và tuning bias** có thể ưu ái Transformer. Một mô hình được thử hàng trăm cấu hình không thể so công bằng với XGBoost được thử năm cấu hình. Tuning budget và search space phải được ghi trước.

**Data-vintage dependence** là rủi ro tái lập. Dữ liệu tài chính có thể được sửa, reconstituted hoặc thay đổi cách tính. Nên lưu extract date, query code, variable definitions và checksums; French Library cũng cung cấp historical archives, cho thấy benchmark data có thể thay đổi theo data cut. citeturn18search13

**Statistical power** bị giới hạn bởi số tuần test, không phải số hàng stock-date. Hàng trăm forecasts tại cùng ngày chịu chung market shock và không độc lập. Statistical inference nên dựa trên date-level losses và block bootstrap, tránh p-values quá lạc quan.

Kết luận phù hợp không phải “Transformer đánh bại thị trường,” mà là một trong các kết luận có điều kiện sau:

- Transformer cải thiện rank IC nhưng không cải thiện net portfolio performance;
- Transformer tạo economic gains gross nhưng gains biến mất sau costs;
- cost-aware optimization giảm turnover đủ để cải thiện net Sharpe;
- XGBoost hoặc đơn giản hơn đạt hiệu năng tương đương, hàm ý complexity của Transformer chưa được biện minh;
- kết quả chỉ tồn tại ở một số regime hoặc cost assumptions.

Tất cả các kết quả này đều có giá trị học thuật nếu pipeline được xác định đúng, không thiên lệch và có ablation rõ.

## Tài liệu tham khảo

Davis, M. H. A., & Norman, A. R. (1990). Portfolio selection with transaction costs. *Mathematics of Operations Research, 15*(4), 676–713. https://doi.org/10.1287/moor.15.4.676 citeturn20search0

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the \(1/N\) portfolio strategy? *The Review of Financial Studies, 22*(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075 citeturn21search3

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599 citeturn21search4

Gârleanu, N., & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *The Journal of Finance, 68*(6), 2309–2340. https://doi.org/10.1111/jofi.12080 citeturn20search1

Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics, 23*(4), 365–380. https://doi.org/10.1198/073500105000000063

Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix: Problems in mean-variance optimization. *The Journal of Portfolio Management, 30*(4), 110–119. https://doi.org/10.3905/jpm.2004.110 citeturn20search3

Ledoit, O., & Wolf, M. (2025). Markowitz portfolios under transaction costs. *The Quarterly Review of Economics and Finance, 100*, 101962. https://doi.org/10.1016/j.qref.2025.101962 citeturn20search2turn20search14

Li, T., Liu, Z., Shen, Y., Wang, X., Chen, H., & Huang, S. (2024). MASTER: Market-guided stock transformer for stock price forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence, 38*(1), 162–170. https://doi.org/10.1609/aaai.v38i1.27767 citeturn13search0turn13search11

Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting, 37*(4), 1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012 citeturn10search3turn10search21

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted transformers are effective for time series forecasting. *International Conference on Learning Representations*. citeturn19search2

Ma, T., Wang, W., & Chen, Y. (2023). Attention is all you need: An interpretable transformer-based asset allocation approach. *International Review of Financial Analysis, 90*, 102876. https://doi.org/10.1016/j.irfa.2023.102876 citeturn16search6

Ma, Y., Han, R., & Wang, W. (2021). Portfolio optimization with return prediction using deep learning and machine learning. *Expert Systems with Applications, 165*, 113973. https://doi.org/10.1016/j.eswa.2020.113973 citeturn16search3

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1), 77–91. https://doi.org/10.2307/2975974

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *International Conference on Learning Representations*. citeturn19search0

Shumway, T. (1997). The delisting bias in CRSP data. *The Journal of Finance, 52*(1), 327–340. https://doi.org/10.1111/j.1540-6261.1997.tb03818.x

Uysal, A. S., Li, X., & Mulvey, J. M. (2024). End-to-end risk budgeting portfolio optimization with neural networks. *Annals of Operations Research, 339*, 397–426. https://doi.org/10.1007/s10479-023-05539-4 citeturn14view2

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*. citeturn11search10

Wang, H., Wang, T., Li, S., Zheng, J., Guan, S., & Chen, W. (2022). Adaptive long-short pattern transformer for stock investment selection. *Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence*, 3970–3977. https://doi.org/10.24963/ijcai.2022/551 citeturn14view1

White, H. (2000). A reality check for data snooping. *Econometrica, 68*(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152 citeturn21search1

Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems, 34*. citeturn19search3

Xiao, X., Hua, X., & Qin, K. (2024). A self-attention based cross-sectional return forecasting model with evidence from the Chinese market. *Finance Research Letters, 62*, 105144. https://doi.org/10.1016/j.frl.2024.105144 citeturn16search5

Zhang, Z., Zohren, S., & Roberts, S. (2021). A universal end-to-end approach to portfolio optimization via deep learning. *arXiv*. https://arxiv.org/abs/2111.09170 citeturn7search0

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence, 35*(12), 11106–11115. https://doi.org/10.1609/aaai.v35i12.17325 citeturn11search0

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. *Proceedings of the 39th International Conference on Machine Learning, 162*, 27268–27286. citeturn19search1

Center for Research in Security Prices. (2026). *CRSP US Stock Databases*. Morningstar Indexes. citeturn17search0

Federal Reserve Bank of St. Louis. (2026). *FRED API documentation*. citeturn18search1turn18search5

French, K. R. (2026). *Data Library*. Tuck School of Business, Dartmouth College. citeturn18search0turn18search4