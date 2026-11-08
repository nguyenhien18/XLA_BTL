# Chạy trực tiếp một ảnh

Lệnh này đọc ảnh đầu vào, chạy lại ba phương pháp và mở một cửa sổ hiển thị ảnh gốc cùng các đường bao dự đoán. Không đọc kết quả phân đoạn cũ và không sử dụng nhãn chuẩn. Từng ô được cập nhật sau khi phương pháp chạy xong; đây không phải huấn luyện mô hình.

Tại PowerShell:

```powershell
Set-Location D:\BTL_XLA
& .\.venv\Scripts\python.exe .\src\demo_image.py --image .\data\prepared\images\img_0320.png
```

Chọn một ảnh khác bằng cách đổi đường dẫn sau `--image`. Dùng ảnh xám 2D 8-bit trong `data/prepared/images`. Mặc định dùng cấu hình baseline như demo trước, với score 0,48/NMS 0,30. Thêm `--profile optimized` để dùng cấu hình được chọn trên validation; số dự đoán có thể đổi.

Chương trình mở ảnh gốc, chạy Otsu, Watershed rồi tải CNN StarDist, suy luận và chạy dựng đa giác/NMS tự cài. Cửa sổ có thể tạm không phản hồi trong lúc TensorFlow khởi tạo hoặc mô hình đang tính; chờ các dòng FINISHED trong terminal. Đường xanh biểu diễn biên dự đoán, không phải đánh dấu đúng/sai vì lần demo này không đọc ground truth. Số vùng trên tiêu đề là số dự đoán, không phải độ chính xác.

Mỗi lượt lưu vào một thư mục mới trong `results/live_demo`, gồm ba TIFF nhãn, ảnh so sánh và run.json. Không thay kết quả trong các thí nghiệm báo cáo. Đóng cửa sổ kết quả để trở lại PowerShell. Muốn chỉ lưu và không mở cửa sổ, thêm `--no-show`.
