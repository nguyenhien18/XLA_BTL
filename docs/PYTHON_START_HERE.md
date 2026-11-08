# Phần Python của bài tập lớn

## Chạy ở đâu?

Mở PowerShell trong `D:\BTL_XLA`. Các lệnh bên dưới dùng trực tiếp Python trong `.venv`, không yêu cầu activate môi trường.

```powershell
.\.venv\Scripts\python.exe src\otsu.py --split demo
.\.venv\Scripts\python.exe src\watershed.py --split demo
.\.venv\Scripts\python.exe src\stardist_predict.py --split demo
```

Hoặc chạy tất cả và đánh giá:

```powershell
.\scripts\run_python_demo.ps1
```

Lần chạy lại sẽ bỏ qua ảnh có đúng provenance và cấu hình. Nếu muốn chủ động thay kết quả bằng cấu hình hiện tại, dùng `--overwrite` ở từng chương trình hoặc `-Overwrite` ở script PowerShell. Dữ liệu gốc và nhãn chuẩn không bị ghi đè.

## Từng thuật toán làm gì?

- `segmentation.py::otsu_threshold`: tự cài đặt tìm ngưỡng Otsu bằng histogram và cực đại phương sai giữa hai lớp.
- `otsu.py`: tìm ngưỡng, lấy vùng sáng hơn ngưỡng, gán nhãn các thành phần liên thông 8 hướng.
- `watershed.py`: dùng foreground Otsu, tính distance transform, chọn hạt giống ở cực đại cục bộ, chạy marker-controlled watershed. Có đảm bảo mỗi thành phần rời rạc có hạt giống. Giữ các vùng sát mép ảnh.
- `stardist_predict.py`: dùng thư viện StarDist chính thức, trọng số pretrained đã xác minh. Đây là triển khai suy luận, không phải tự huấn luyện hoặc tự viết toàn bộ mạng/NMS.
- `evaluate.py`: ghép đối tượng theo IoU một-một; tính TP/FP/FN, Precision, Recall, F1, mean matched IoU, PQ; chỉ số nền/đối tượng bổ sung được ghi rõ binary IoU/Dice.
- `compare_methods.py`: bảng/biểu đồ chung và đối chiếu trực tiếp hai đầu ra Fiji/Python; nhãn chuẩn vẫn là căn cứ đánh giá chất lượng.

Các thuật toán distance transform, connected components và watershed dùng SciPy/scikit-image. Báo cáo phải ghi rõ phần tự cài đặt và phần dùng thư viện, không mô tả mọi thành phần đều do nhóm viết từ đầu.

## Cấu hình ban đầu

`configs/python_baseline.json` lưu tham số trước khảo sát validation:

- Otsu: chưa làm mượt, chưa lọc theo diện tích.
- Watershed: cùng tiền xử lý Otsu, khoảng cách tối thiểu giữa các đỉnh là 5 pixel.
- StarDist: `2D_versatile_fluo`, normalize percentile 1/99.8, probability 0.48, NMS 0.30, boundary exclusion 0.
- StarDist chạy CPU để đối chiếu cách chạy Fiji hiện tại. Thời gian trong log chưa phải benchmark kiểm soát vì lần đầu bao gồm TensorFlow khởi tạo.

Hai bản có cùng tên mô hình/cấu hình không đảm bảo ảnh nhãn giống từng pixel, do phiên bản TensorFlow và quy trình NMS/raster hóa khác nhau. Cần dựa vào bảng đối chiếu thực đo.

## Dữ liệu và cách chia tập

Demo gồm 10 ảnh nằm trong test, dùng để kiểm tra quy trình và minh họa; không dùng điểm demo để chọn tham số. Khảo sát/thay tham số phải làm trên 82 ảnh validation. Tập test cuối gồm 82 ảnh. Dữ liệu huấn luyện gốc của pretrained có thể trùng DSB2018, nên chưa được tuyên bố test là ảnh hoàn toàn chưa thấy đối với model pretrained.

## Môi trường

Python 3.13, môi trường riêng tại `.venv`. `requirements-lock.txt` ghi chính xác các phiên bản sau khi đã cài và kiểm tra. `docs/python_environment.json` ghi nền tảng và thông tin thiết bị.

Tạo lại môi trường từ bản cài Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Trọng số ở `models/2D_versatile_fluo`, nguồn và checksum trong `models/2D_versatile_fluo_source.json`.

## Kiểm tra chương trình

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các kiểm tra bao gồm ngưỡng Otsu đối chiếu scikit-image trên ảnh demo thật, ảnh cường độ đồng nhất và tách hai hình tròn tiếp xúc bằng Watershed.

## File kết quả

- Ảnh nhãn: `results/otsu`, `results/watershed`, `results/stardist_python`.
- Provenance ảnh đầu vào/cấu hình/kết quả: `provenance.json` trong từng thư mục phương pháp.
- Chỉ số: `results/metrics/*_demo_per_image.csv`, `*_demo_summary.csv`.
- So các phương pháp: `docs/methods_demo_comparison.md`.
- Hình so sánh: `results/figures/methods_demo`.
- Fiji/Python: `results/metrics/fiji_python_demo_comparison.csv`.

## Validation, test và benchmark đã hoàn thành

- Khảo sát 12 cấu hình Otsu, 36 Watershed, 15 StarDist trên 82 ảnh validation.
- Đánh giá baseline và cấu hình được chọn trên đủ 82 ảnh test.
- Đo CPU từng ảnh với warm-up và 5 lần lặp, tổng 1.230 số đo.
- Dữ liệu cơ bản ở `results/<method>` được giữ; dữ liệu tối ưu ở `results/optimized/<method>`.
- Bảng tổng hợp và phân tích: `docs/TEST_RESULTS.md`.

Tái chạy từ PowerShell tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe src\tune_parameters.py
.\scripts\run_test_experiments.ps1
```

Chạy một phương pháp với tham số đã chốt:

```powershell
.\.venv\Scripts\python.exe src\stardist_predict.py --split test --profile optimized
```

`--profile baseline` và `--profile optimized` dùng cấu hình và thư mục kết quả riêng. Các script bỏ qua đầu ra đã có đúng provenance; không chọn lại cấu hình theo điểm test.
