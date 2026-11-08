# Chạy tự động StarDist trong Fiji

## Script và kết quả

- Script: `D:\BTL_XLA\fiji\macros\batch_stardist_demo.groovy`.
- Danh sách ảnh: `D:\BTL_XLA\data\splits\demo.txt`.
- Ảnh đầu vào: `D:\BTL_XLA\data\prepared\images`.
- Kết quả: `D:\BTL_XLA\results\stardist_fiji`.
- Nhật ký: `D:\BTL_XLA\results\metrics\fiji_batch_*.csv`.
- Các lần chạy mới được ghi tham số vào `D:\BTL_XLA\fiji\parameters.csv`.

Đây là script **Groovy chạy bằng Fiji**, sử dụng trực tiếp các plugin Java StarDist, CSBDeep và TensorFlow đã cài; không phải chương trình StarDist Python.

## Cách mở trong Fiji

1. Mở Fiji đã cài StarDist.
2. Chọn **File → New → Script…** để mở Script Editor.
3. Trong Script Editor chọn **File → Open…** và mở file `.groovy` ở trên.
4. Kiểm tra **Language → Groovy**, rồi bấm **Run**.
5. Theo dõi dòng `OK`, `SKIP`, `ERROR` và dòng tổng kết `DONE` trong phần output.

Không cần mở từng ảnh hay chọn các ROI bằng tay.

## Cấu hình cố định

- Model: `Versatile (fluorescent nuclei)`.
- Normalize: bật, percentile 1 / 99.8.
- Probability threshold 0.48; overlap/NMS threshold 0.30.
- Tiles 1; boundary exclusion 0.
- Không sử dụng nhãn chuẩn để chạy mô hình.

## TIFF và ROI được tạo thế nào?

Script gọi lệnh StarDist 2D với output Label Image để không thay danh sách ROI đang mở trong Fiji. Bản đồ probability và distance của cùng lần suy luận được đưa qua chính lớp NMS của plugin để xuất các đa giác ROI gốc. Không chạy CNN lần hai. Trước khi lưu, script kiểm tra các đa giác tái tạo đúng từng pixel của ảnh nhãn. Các ROI được ghi trực tiếp thành ZIP nên không bị lỗi chỉ lưu một ROI do chọn thiếu.

Mỗi ảnh tạo hai file:

```text
img_XXXX.tif
img_XXXX_rois.zip
```

## Chạy lại và bảo toàn kết quả

- Cặp TIFF/ZIP đã tồn tại, đúng kích thước và khớp số đối tượng: `SKIP`, giữ nguyên.
- Chỉ có một file hoặc cặp không hợp lệ: báo `ERROR`, không tự ghi đè.
- Khi tất cả 10 ảnh đã đủ, chạy lại sẽ báo `processed=0, skipped=10, failed=0`.
- Nếu muốn trình diễn lại toàn bộ suy luận trong một thư mục mới, đổi dòng `outputDir` trong script sang `results/stardist_fiji_rerun`. Không cần xóa kết quả đã có.
- Thời gian trong CSV bao gồm mở ảnh, suy luận, NMS và xuất ROI; không dùng trực tiếp như benchmark thời gian suy luận thuần.
- Hai kết quả chạy tay được giữ nguyên. Script không suy đoán cấu hình cũ từ file kết quả, nên chỉ ghi tham số cho những ảnh thực sự được script chạy mới.

## Kiểm tra đã thực hiện

Script đã được chạy bằng ScriptService của chính bộ Fiji đang cài trên máy: xử lý thành công 8 ảnh còn lại, bỏ qua 2 cặp đã lưu từ thao tác tay, không lỗi. Các kết quả được kiểm tra lại bằng công cụ đọc TIFF và ZIP độc lập.

Nguồn API plugin: https://github.com/stardist/stardist-imagej
