# Kết quả fine-tune StarDist

Đã huấn luyện tiếp từ pretrained trên 381 ảnh train, dùng 82 ảnh validation. Chạy 16 epoch; chọn epoch 11 theo val_loss thấp nhất. Thời gian 11.8 phút trên CPU.

Đánh giá đủ 82 ảnh test sau khi chốt checkpoint. So sánh giữ nguyên score 0,55 và NMS 0,30; chưa tối ưu lại ngưỡng cho mô hình fine-tune. Phần dựng đa giác/NMS tự cài đã khớp tập ứng viên của thư viện trên tất cả ảnh đánh giá. Các chỉ số được đối chiếu hàm tham chiếu StarDist.

| IoU | F1 pretrained | F1 fine-tune | Chênh lệch |
|---|---:|---:|---:|
| 0.5 | 0.9309 | 0.9216 | -0.0093 |
| 0.6 | 0.8975 | 0.8796 | -0.0179 |
| 0.7 | 0.8299 | 0.8090 | -0.0210 |
| 0.8 | 0.6547 | 0.6484 | -0.0063 |

Fine-tune không đảm bảo cải thiện mọi chỉ số. Đường loss dùng các patch và augmentation nên không đồng nhất với đánh giá toàn ảnh. Mô hình pretrained có thể đã thấy một phần DSB2018 trong huấn luyện gốc; fine-tune không loại bỏ giới hạn này.

Hồ sơ: configs/finetune.json; models/stardist_finetuned_v1/history.csv, history.json, training_metadata.json; results/metrics/finetune_before_after.csv; results/figures/finetune/.