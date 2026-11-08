# Kết quả thí nghiệm Python trên tập test

Đã đánh giá đủ 82 ảnh test. Cấu hình được chọn bằng 82 ảnh validation; không dùng điểm test để chọn lại tham số. So sánh cả baseline và cấu hình được chọn để kiểm tra thay đổi có tổng quát hóa hay không.

## Tham số chọn trên validation

- Otsu: `{"sigma": 0.0, "min_area": 32}`; objective validation = 0.700181.
- Watershed: `{"sigma": 0.0, "min_area": 32, "min_distance": 9}`; objective validation = 0.769861.
- StarDist Python: `{"probability": 0.55, "nms": 0.3}`; objective validation = 0.935528.

Objective là trung bình F1 gộp ở ba ngưỡng IoU 0.5/0.6/0.7. Đã khảo sát 12 cấu hình Otsu, 36 Watershed, 15 StarDist. Nếu đồng điểm, lấy cấu hình xuất hiện trước trong lưới đã khai báo.

## Kết quả test tại IoU ≥ 0.5

| Phương pháp | Profile | Chuẩn | Dự đoán | TP | FP | FN | Precision | Recall | F1 | PQ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Otsu | baseline | 3852 | 3275 | 2096 | 1179 | 1756 | 0.6400 | 0.5441 | 0.5882 | 0.4894 |
| Otsu | optimized | 3852 | 2420 | 2069 | 351 | 1783 | 0.8550 | 0.5371 | 0.6598 | 0.5492 |
| Watershed | baseline | 3852 | 4827 | 2795 | 2032 | 1057 | 0.5790 | 0.7256 | 0.6441 | 0.5158 |
| Watershed | optimized | 3852 | 3305 | 2613 | 692 | 1239 | 0.7906 | 0.6783 | 0.7302 | 0.5945 |
| StarDist Python | baseline | 3852 | 3906 | 3601 | 305 | 251 | 0.9219 | 0.9348 | 0.9283 | 0.7750 |
| StarDist Python | optimized | 3852 | 3816 | 3569 | 247 | 283 | 0.9353 | 0.9265 | 0.9309 | 0.7779 |

## Thay đổi trên test

- Otsu: F1 gộp 0.5882 → 0.6598; chênh lệch +0.0716. Precision 0.6400 → 0.8550; recall 0.5441 → 0.5371; macro F1 0.7675 → 0.7995.
- Watershed: F1 gộp 0.6441 → 0.7302; chênh lệch +0.0861. Precision 0.5790 → 0.7906; recall 0.7256 → 0.6783; macro F1 0.7879 → 0.8402.
- StarDist Python: F1 gộp 0.9283 → 0.9309; chênh lệch +0.0025. Precision 0.9219 → 0.9353; recall 0.9348 → 0.9265; macro F1 0.9359 → 0.9337.

Tên optimized nghĩa là được chọn trên validation, không đảm bảo cải thiện trên mọi ảnh hoặc mọi chỉ số test.

## Thời gian chạy CPU — cấu hình optimized

| Phương pháp | Mean (ms) | Median (ms) | P95 (ms) | Số ảnh | Lặp/ảnh |
|---|---:|---:|---:|---:|---:|
| Otsu | 2.291 | 0.992 | 5.998 | 82 | 5 |
| Watershed | 15.954 | 6.357 | 48.443 | 82 | 5 |
| StarDist Python | 135.623 | 88.123 | 294.065 | 82 | 5 |

Các ảnh được đọc vào RAM trước. Mỗi phương pháp được warm-up trên từng ảnh; đo lặp với thứ tự phương pháp trộn ngẫu nhiên bằng seed cố định. Thời gian bao gồm tiền xử lý và phân đoạn/NMS, loại trừ đọc/ghi file, tải mô hình, đếm và tính chỉ số. Ảnh có nhiều kích thước nên độ phân tán còn phụ thuộc kích thước và mật độ; đây là số đo trên máy hiện tại, không phải tốc độ phổ quát. Xem protocol benchmark trong results/metrics.

## Giới hạn và minh họa

- Các chỉ số tổng hợp dùng đủ 82 ảnh. Hình so sánh chi tiết chỉ xuất cho 10 ảnh demo đã chọn từ trước, không chọn theo kết quả đẹp.
- Bảng và benchmark trong tài liệu này dùng trọng số pretrained và NMS thư viện. Thí nghiệm bổ sung về fine-tune và NMS tự cài đã hoàn tất, trình bày riêng tại `docs/FINETUNE_RESULTS.md` và `docs/CUSTOM_IMPLEMENTATION.md`; không trộn thời gian thực thi của các đợt chạy.
- Bộ DSB2018 có thể trùng dữ liệu huấn luyện gốc của pretrained. Tập test giữ lại trong dự án không tự chứng minh độc lập với quá trình huấn luyện gốc.
- Fiji mới được đối chiếu trên 10 ảnh demo; bảng test này là ba phương pháp Python, không phải 82 ảnh Fiji.
- F1 gộp dùng tổng TP/FP/FN, khác trung bình F1 theo ảnh. Bảng nhiều ngưỡng IoU giúp kiểm tra chất lượng biên.

## Tái chạy

`python src/tune_parameters.py` để khảo sát lại validation; `scripts/run_test_experiments.ps1` để chạy test và benchmark. Dùng Python trong `.venv`. Hai profile lưu riêng ở `results/<method>` và `results/optimized/<method>`.
