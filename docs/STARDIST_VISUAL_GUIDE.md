# Đọc hiểu StarDist qua kết quả của dự án

Đây là ghi chú học và bộ chú thích hình, chưa phải báo cáo nộp. Các hình nằm trong `results/explanations/stardist/`; mỗi hình có PNG để xem và SVG để chèn tài liệu. Code tạo hình: `src/visualize_stardist.py`.

## 1. Đầu vào, nhãn chuẩn và dự đoán khác nhau thế nào?

| Thành phần | Nơi lưu | Cách hiểu |
|---|---|---|
| Ảnh gốc đầu vào | `data/prepared/images/*.png` | Ảnh xám chụp nhân tế bào; điểm sáng không tự mang số thứ tự nhân. |
| Nhãn chuẩn, ground truth | `data/prepared/labels/*.tif` | Chú giải từ bộ dữ liệu; 0 là nền, các số dương phân biệt từng nhân. |
| Nhãn dự đoán | `results/optimized/stardist_python/*.tif` | Kết quả mô hình tìm ra từ ảnh đầu vào. |
| Probability/score map | File `*_network_outputs.npz`, mảng `probability` | Các số thực 0–1 do mạng dự đoán, dùng chọn vị trí sinh đa giác. |
| Distance map | Cùng file NPZ, mảng `distance` | 32 độ dài tia tại mỗi vị trí đầu ra của mạng. |

Màu trong hình nhãn chỉ giúp phân biệt các ID. ID 7 trong dự đoán không cần trùng ID 7 trong nhãn chuẩn: chương trình ghép cặp bằng độ chồng lấn hình dạng. Kết quả chính của dự án là ảnh nhãn số, không phải ảnh màu minh họa.

## 2. Xem cả quy trình trước

![Quy trình đang dùng](../results/explanations/stardist/01_stardist_workflow.png)

Ảnh → chuẩn hóa → CNN/U-Net → bản đồ score và khoảng cách → sinh đa giác ứng viên → NMS bỏ ứng viên trùng → ảnh nhãn từng nhân. Ý tưởng đa giác star-convex và dự đoán bằng CNN xuất phát từ [bài báo StarDist, Schmidt và cộng sự, 2018](https://arxiv.org/abs/1806.03535).

Cấu hình đọc trực tiếp từ mô hình dự án: `2D_versatile_fluo`, ảnh một kênh, 32 tia, grid `(2,2)`, U-Net depth 3, **1.425.185 tham số**. Đây là mô hình đã huấn luyện sẵn. Các hình trong tài liệu này minh họa mô hình pretrained. Sau đó dự án đã bổ sung fine-tune và dựng đa giác/NMS tự cài; xem `FINETUNE_RESULTS.md` và `CUSTOM_IMPLEMENTATION.md`. CNN vẫn do StarDist/TensorFlow thực hiện.

## 3. CNN/U-Net làm việc gì?

CNN dùng các bộ lọc học được để biến ảnh thành đặc trưng. U-Net có nhánh thu nhỏ để lấy ngữ cảnh, nhánh tăng kích thước và kết nối giữa hai nhánh để giữ thông tin vị trí. Xem [bài báo U-Net, Ronneberger và cộng sự, 2015](https://arxiv.org/abs/1505.04597).

Trong mô hình đang chạy, đầu ra cuối gồm một kênh score và 32 kênh độ dài tia. Đây là **33 kênh dữ liệu tại mỗi vị trí**, không phải chỉ phát hiện được 33 nhân.

Ví dụ thực đo `img_0320`: ảnh 256×256 → score 128×128 → distance 128×128×32. Grid 2 nghĩa là các gốc ứng viên cách nhau 2 pixel trên ảnh đầu vào. Với `img_0144` 360×360, hai kích thước tương ứng là 180×180 và 180×180×32. Bản đầy đủ các lớp mạng nằm trong `model_summary.txt`.

## 4. Probability map phải đọc ra sao?

Mở `img_0144_02_probability_map.png`: trái là ảnh đầu vào, giữa là score thực của mạng, phải là đường đồng mức score 0,55. Đường đồng mức này **chưa phải biên dự đoán của nhân**; biên được dựng bằng các tia ở bước sau.

Trong cách huấn luyện StarDist, score tham chiếu được tạo từ khoảng cách đến biên, chuẩn hóa trong từng đối tượng, nên vùng phía trong nhân thường có score cao. Hình `06_target_vs_prediction.png` so sánh nhãn chuẩn, score tham chiếu tính từ nhãn và score mạng tính từ ảnh. Đây chỉ là minh họa mục tiêu học; dự án không dùng hình này để huấn luyện. Công thức tạo mục tiêu có trong [mã nguồn huấn luyện chính thức](https://github.com/stardist/stardist/blob/main/stardist/models/model2d.py).

Profile optimized chọn score **>0,55** để sinh ứng viên. Không diễn giải score 0,55 là “chắc chắn đúng 55%”. Profile baseline dùng 0,48; việc đổi ngưỡng là chọn tham số trên validation, không phải học lại trọng số mạng.

## 5. 32 tia tạo ra đường biên thế nào?

![32 tia dự đoán thật](../results/explanations/stardist/img_0144_03_star_convex_rays.png)

Hình phóng to một ứng viên của `img_0144`. Gốc ở `(y=312, x=98)` có score 0,9267. Mạng dự đoán 32 độ dài theo các góc cách đều **11,25°**; nối các đầu tia thành đa giác. Gốc dự đoán không bắt buộc trùng trọng tâm hình học của nhân.

Với tia k=0,…,31, góc θk=2πk/32 và độ dài rk:

```text
xk = cx + rk × cos(θk)
yk = cy + rk × sin(θk)
```

Tọa độ mảng dùng thứ tự `(y,x)`. Độ dài tia đã tính theo pixel ảnh đầu vào; chỉ gốc trên lưới đầu ra được quy đổi theo grid. Xem [mã hình học StarDist](https://github.com/stardist/stardist/blob/main/stardist/geometry/geom2d.py). File `img_0144_rays.csv` lưu đủ góc, độ dài và tọa độ 32 đầu tia thực tế.

Star-convex nghĩa là có một gốc nhìn được mọi điểm trong hình qua các đoạn thẳng nằm trong hình. Một số hình lõm vẫn thỏa điều kiện này; biểu diễn có giới hạn với hình có lỗ hoặc không thỏa điều kiện star-convex.

## 6. Vì sao cần NMS?

![Ứng viên trước và sau NMS](../results/explanations/stardist/img_0144_04_nms_before_after.png)

Nhiều vị trí trong cùng một nhân có thể sinh các đa giác gần trùng nhau. NMS ưu tiên score cao và loại các ứng viên chồng lấn quá ngưỡng với ứng viên đã giữ. Đếm ứng viên trước NMS không phải đếm nhân.

Số đo thật trong profile optimized:

| Ảnh | Ứng viên qua score >0,55 | Giữ sau NMS | Bị loại |
|---|---:|---:|---:|
| img_0144 | 2.563 | 29 | 2.534 |
| img_0320 | 632 | 42 | 590 |

Hình NMS chỉ phóng to một vùng. Với img_0144, vùng này có 144 gốc ứng viên thuộc một nhãn chuẩn, giữ lại 1; chỉ vẽ 28 đường để dễ nhìn. NMS vẫn chạy trên toàn ảnh. Nhãn chuẩn chỉ giúp chọn vùng minh họa, không tham gia dự đoán hoặc quyết định loại ứng viên.

Trong triển khai Python đang dùng, NMS overlap = diện tích giao / diện tích đa giác nhỏ hơn. **IoU đánh giá** = diện tích giao / diện tích hợp. Hai công thức khác nhau; ngưỡng NMS 0,30 và IoU đánh giá 0,50 không thay thế nhau. Công thức NMS được đối chiếu [mã C++ chính thức](https://github.com/stardist/stardist/blob/main/stardist/lib/stardist2d.cpp).

Hình `07_nms_overlap_vs_evaluation_iou.png` dùng ví dụ hình học tự tạo: A diện tích 100 chứa B diện tích 25 thì NMS overlap=1,00, còn IoU=0,25. Đây không phải số đo từ ảnh tế bào.

## 7. Đọc kết quả mà không nhầm phiên bản

Các hình `*_05_final_labels.png` hiển thị ảnh đầu vào, nhãn kết quả và biên đa giác. Với cả hai ảnh minh họa, code kiểm tra nhãn tạo lại **giống từng pixel** với file optimized đã lưu. Metadata: `explanation_metadata.json`.

img_0320 có 42 dự đoán ở profile optimized (score 0,55), trong khi lần Fiji/baseline trước đó có 43 (score 0,48). Cần ghi rõ profile khi trình bày. Số dự đoán cũng không đồng nghĩa số dự đoán đúng; phải ghép với nhãn chuẩn và tính TP/FP/FN.

Trên **đủ 82 ảnh test**, F1 gộp tại IoU 0,50 của optimized: Otsu **0,6598**, Watershed **0,7302**, StarDist Python **0,9309**. Chi tiết tại `docs/TEST_RESULTS.md`. Đổi tham số tăng precision nhưng giảm recall; riêng StarDist có macro F1 giảm nhẹ, nên không kết luận mọi chỉ số đều tốt hơn.

Mô hình pretrained có thể đã thấy một phần DSB2018 trong huấn luyện gốc. Các split của dự án không trùng nhau nhưng chưa chứng minh độc lập với dữ liệu huấn luyện gốc. Kết quả này mô tả thực nghiệm trên tập giữ lại của dự án.

## 8. Tạo lại bộ hình

Trong PowerShell:

```powershell
Set-Location D:\BTL_XLA
& .\.venv\Scripts\python.exe -B .\src\visualize_stardist.py
```

Script ghi lại các hình và metadata minh họa; không đổi dữ liệu, tham số đã chọn hay file dự đoán dùng đánh giá. Mặc định dùng img_0144 và img_0320. Bộ hình có 11 PNG và 11 SVG; NPZ giữ đầu ra mạng để đối chiếu. Hướng dẫn thao tác plugin xem [StarDist trên ImageJ](https://imagej.net/plugins/stardist) và `fiji/HUONG_DAN_SCRIPT.md`.

## 9. Trước khi viết báo cáo chính

Người dùng muốn cung cấp file mẫu khóa trước. Cần đọc mẫu trước khi soạn DOCX/PDF/PPTX để thống nhất bố cục, bìa và định dạng. Các ghi chú và bảng kết quả hiện tại là tài liệu đầu vào, chưa phải bản nộp. Khi viết phải phân biệt phần nhóm tự cài đặt với phần gọi thư viện/mô hình pretrained.
