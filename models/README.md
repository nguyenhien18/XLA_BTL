# Mô hình

Trọng số không được đưa vào Git. Chạy `python scripts/download_model.py` để tải mô hình pretrained chính thức vào `models/2D_versatile_fluo/` và kiểm tra SHA-256.

Lịch sử fine-tune đã báo cáo nằm trong `experiments/recorded_run/finetune/`. Checkpoint fine-tuned không kèm repo. Có thể tạo lại bằng `src/finetune_stardist.py` sau khi chuẩn bị đầy đủ 545 ảnh. Kết quả trên máy/môi trường khác có thể khác lần ghi nhận.

Nguồn mô hình và checksum: `2D_versatile_fluo_source.json`.
