# Phần tự cài đặt và kiểm chứng Fiji

## Phạm vi tự cài

`src/custom_postprocess.py` tự cài các bước chuyển đầu ra mạng thành nhãn:

1. Lấy vị trí có score vượt ngưỡng, quy đổi gốc từ lưới mạng sang tọa độ ảnh.
2. Dựng đỉnh đa giác theo x = cx + r*cos(theta), y = cy + r*sin(theta), với 32 góc cách đều.
3. Sắp ứng viên theo score giảm dần, giữ ứng viên chưa bị loại và tính giao với các ứng viên thấp điểm hơn.
4. Loại ứng viên nếu diện tích giao / diện tích đa giác nhỏ hơn vượt ngưỡng NMS. Bỏ qua các cặp có hộp bao không giao để giảm tính toán.
5. Vẽ các đa giác còn lại thành ảnh nhãn; vùng giao thuộc ứng viên có score cao hơn.

CNN vẫn do StarDist/TensorFlow chạy với mô hình pretrained hoặc fine-tuned. NumPy xử lý mảng; pyclipper chỉ tính giao đa giác; skimage.draw.polygon tô vùng đa giác. Việc quyết định ứng viên nào được giữ/loại là code tự viết, không gọi hàm NMS StarDist. Đây là cài đặt lại phần hình học và hậu xử lý, không phải tự viết toàn bộ CNN hay thư viện hình học.

Quy ước số thực và chuyển tọa độ sang số nguyên được đối chiếu triển khai StarDist để kiểm tra tương thích. Nguồn thuật toán tham chiếu: https://github.com/stardist/stardist/blob/main/stardist/lib/stardist2d.cpp và https://github.com/stardist/stardist/blob/main/stardist/geometry/geom2d.py.

## Kiểm chứng

`src/check_custom_postprocess.py` kiểm tra hình học đơn giản, hai đa giác lồng nhau, thứ tự score và tập rỗng, sau đó so với thư viện trên ảnh thật. Hai đa giác lồng nhau giúp phát hiện nhầm lẫn giữa overlap của NMS và IoU.

Lần kiểm tra 10 ảnh demo đã khớp toàn bộ tọa độ đỉnh, chỉ số ứng viên sau NMS và từng pixel của ảnh nhãn optimized. Kết quả chi tiết tại `results/metrics/custom_postprocess_validation.json`. Kiểm chứng mở rộng đã hoàn tất trên 82/82 ảnh test, khớp toàn bộ pixel và chỉ số ứng viên; xem `custom_postprocess_validation_test.json`.

Chạy trong PowerShell tại `D:\BTL_XLA`:

```powershell
& .\.venv\Scripts\python.exe -B .\src\check_custom_postprocess.py --split demo
& .\.venv\Scripts\python.exe -B .\src\check_custom_postprocess.py --split test
```

Kết quả nhãn lưu ở `results/custom_stardist/`. Các thời gian ghi trong phép kiểm tra chỉ là nhật ký thực thi; có thể chạy đồng thời với huấn luyện, không dùng làm benchmark so sánh tốc độ với C++.

## Định lượng trực tiếp bằng Fiji

`fiji/macros/evaluate_demo.groovy` đọc ảnh nhãn chuẩn và nhãn dự đoán Fiji, đếm pixel, tạo bảng giao giữa các đối tượng, ghép cặp một-một bằng đường tăng trong đồ thị hai phía. Script tính TP/FP/FN, Precision, Recall, F1, binary IoU và binary Dice mà không gọi Python.

Đã chạy với lõi ImageJ 1.54p, Groovy 4.0.23 và Java 8 đi kèm bản Fiji của dự án ở chế độ headless. Kết quả 40 dòng (10 ảnh, bốn ngưỡng IoU 0.5/0.6/0.7/0.8) khớp Python cho cả 10 trường đã kiểm tra. Chi tiết: `results/metrics/fiji_quantitative_validation.json`.

Script Fiji tối đa hóa số cặp đạt ngưỡng, đủ để đối chiếu TP/FP/FN và F1. Nó chưa tối ưu thêm tổng IoU khi nhiều cách ghép có cùng số cặp, nên không dùng script này để kết luận PQ hoặc mean matched IoU đã được kiểm chứng độc lập.

Để chạy trong giao diện Fiji: mở script, chọn ngôn ngữ Groovy, bấm Run. Sau khi chạy, bảng `Fiji quantitative validation` xuất hiện. Chụp bảng và cửa sổ Script Editor để bổ sung minh họa thao tác cho báo cáo. Chạy headless tạo CSV và log nhưng không phải ảnh chụp giao diện.

Đầu ra:

- `results/metrics/fiji_quantitative_demo.csv`: từng ảnh và từng ngưỡng.
- `results/metrics/fiji_quantitative_demo_summary.csv`: kết quả gộp.
- `results/metrics/fiji_quantitative_run.log`: nhật ký thực thi Java/Groovy.

Các chỉ số nhị phân chỉ đo nền/đối tượng. Đánh giá instance vẫn cần ghép từng nhân; đếm đủ số lượng không chứng minh đã phân đoạn đúng.
