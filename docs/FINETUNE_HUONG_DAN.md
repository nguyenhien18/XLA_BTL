# Đọc và chạy phần fine-tune

## Hiểu đơn giản

Pretrained là mô hình đã được tác giả huấn luyện sẵn. Fine-tune là cho mô hình đó học tiếp trên dữ liệu train của dự án. Đây là thay đổi trọng số mạng thật, khác với chỉ chỉnh ngưỡng score/NMS.

Đã chạy 16 epoch trong khoảng 11,8 phút trên CPU. Epoch là một đợt gồm 64 batch, mỗi batch 4 miếng ảnh (patch) 128×128 được lấy mẫu; không có nghĩa duyệt đủ toàn bộ 381 ảnh trong một epoch. Dùng xoay/lật đồng thời ảnh và mask, thay đổi cường độ chỉ trên ảnh. Learning rate bắt đầu 0,00001. Early stopping dừng khi validation loss không cải thiện đủ trong 5 epoch liên tiếp; checkpoint được chọn ở epoch 11.

## Các file cần biết

| Đường dẫn từ gốc dự án | Tác dụng |
|---|---|
| `configs/finetune.json` | Cấu hình thí nghiệm, seed, số bước, kích thước patch và ngưỡng so sánh |
| `src/finetune_stardist.py` | Đọc train/validation, học tiếp từ pretrained và lưu checkpoint |
| `models/stardist_finetuned_v1/weights_best.h5` | Trọng số epoch có validation loss thấp nhất; dùng cho đánh giá |
| `models/stardist_finetuned_v1/history.csv` và `history.json` | Loss thực tế từng epoch; CSV đánh số epoch từ 0, báo cáo đánh số từ 1 |
| `models/stardist_finetuned_v1/training_metadata.json` | Trạng thái complete, số ảnh, thời gian, checkpoint và checksum |
| `src/evaluate_finetuned.py` | Dự đoán test bằng checkpoint đã chọn và hậu xử lý tự cài |
| `results/stardist_finetuned/` | 82 ảnh nhãn dự đoán TIFF, không phải nhãn chuẩn |
| `results/metrics/finetuned_test_per_image.csv` | 328 dòng: 82 ảnh × 4 ngưỡng IoU |
| `results/metrics/finetune_before_after.csv` | Bảng so sánh F1, precision, recall trước/sau |
| `results/figures/finetune/` | Đường loss, biểu đồ trước/sau và 10 ảnh minh họa demo |
| `results/metrics/finetune_evaluation_validation.json` | Kiểm chứng chỉ số với StarDist và checksum đầu vào/đầu ra |

## Cách chạy

Dùng PowerShell tại `D:\BTL_XLA`:

```powershell
# Đánh giá lại mô hình đã có; ghi lại các đầu ra fine-tuned tương ứng.
& .\.venv\Scripts\python.exe -B .\src\evaluate_finetuned.py
```

Không cần chạy huấn luyện lại để demo. `src/finetune_stardist.py` cố ý từ chối chạy nếu thư mục mô hình đích đã tồn tại, để bảo toàn log/trọng số. Muốn tái tạo đầy đủ, dùng bản sao dự án mới chưa có thư mục `models/stardist_finetuned_v1`, giữ nguyên dữ liệu/split và môi trường, rồi chạy script huấn luyện trước script đánh giá. Seed được cố định; không tuyên bố mọi môi trường/phần cứng cho kết quả giống từng bit.

## Kết quả cần giải thích khi bảo vệ

Ở IoU 0,5, TP tăng 3569 → 3616, FN giảm 283 → 236 nhưng FP tăng 247 → 379. Vì vậy recall tăng còn precision và F1 giảm (93,09% → 92,16%). Giảm khoảng 0,93 điểm phần trăm F1, không phải tăng chất lượng. Thí nghiệm giữ nguyên ngưỡng score/NMS, chưa khảo sát lại ngưỡng cho mô hình fine-tune. Loss tính trên patch khác với F1 ghép đối tượng trên toàn ảnh; loss tốt hơn không đảm bảo F1 test cao hơn. Chưa đủ bằng chứng để khẳng định nguyên nhân duy nhất của việc giảm F1.

Train/validation/test không trùng nhau trong dự án. Tuy nhiên pretrained có thể đã học một phần DSB2018 trước đó, nên fine-tune không biến đây thành test hoàn toàn độc lập. Mô tả giới hạn này trong báo cáo.
