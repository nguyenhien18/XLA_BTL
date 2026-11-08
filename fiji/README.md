# Thực hành Fiji

Mở ảnh PNG trong `data/prepared/images` theo danh sách `data/splits/demo.txt`. Mở cùng tên TIFF trong `data/prepared/labels` để xem nhãn chuẩn.

Lưu kết quả StarDist dạng Label Image vào `results/stardist_fiji/<tên ảnh>.tif`; không ghi đè nhãn chuẩn. Ảnh chụp màn hình lưu trong `fiji/screenshots`.

Ghi mô hình, phiên bản plugin, chuẩn hóa và ngưỡng vào `parameters.csv` sau mỗi cấu hình chạy. Cấu hình chạy tự động đã được ghi trong `parameters.csv`; các lần chạy tay cần ghi riêng nếu muốn lưu lịch sử đầy đủ.

Hướng dẫn plugin chính thức: https://imagej.net/plugins/stardist

## Script tự động đã hoàn thành

Xem `HUONG_DAN_SCRIPT.md`. Script ở `macros/batch_stardist_demo.groovy` đã chạy thành công 8 ảnh còn lại; đủ 10 cặp TIFF/ROI trong thư mục kết quả. Những lần chạy tự động mới được ghi tham số trong `parameters.csv`.
