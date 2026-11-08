# Dữ liệu dùng cho bài tập

## Nguồn

- BBBC038v1 / 2018 Data Science Bowl, Broad Institute.
- Trang mô tả: https://bbbc.broadinstitute.org/BBBC038
- File gốc: https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip
- Giấy phép ghi tại nguồn: CC0.
- Bài báo nguồn đề nghị trích dẫn: https://doi.org/10.1038/s41592-019-0612-7
- SHA-256 ZIP: `dcb6edc2690f137406638b2309581a71522c4dff19157d118453b448dcddcb68`.

ZIP lưu đủ 670 ảnh với mask riêng của từng nhân. Dự án sử dụng một phần theo tiêu chí bên dưới. Không lấy ảnh đã xử lý từ dự án cũ.

## Phạm vi được chốt

Ảnh hiển vi 2D có nội dung xám, nền tối và nhân sáng. Tiêu chí kỹ thuật cố định, áp dụng trước khi chạy thuật toán:

1. Ba kênh R/G/B của ảnh gốc có giá trị giống nhau; alpha phải hoàn toàn đục.
2. Trung vị độ sáng nhỏ hơn 100 trên thang 0–255.
3. Độ sáng trung bình bên trong nhãn nhân lớn hơn độ sáng trung bình nền.
4. Ảnh và nhãn khớp kích thước; mask không rỗng, không chồng lấn.
5. Không giữ ảnh có nội dung pixel trùng hoàn toàn với ảnh đã chọn.

Tiêu chí này xác định phạm vi dữ liệu của bài; không phải chứng nhận phương thức chụp hoặc chất lượng gán nhãn sinh học. Ảnh ngoài phạm vi được ghi trong `data/excluded.csv` và vẫn có trong ZIP nguồn. Đặc biệt có một ảnh nền tối nhưng phần gán nhãn tối hơn nền nên được loại theo tiêu chí 3, không sửa nhãn để ép phù hợp.

## Chuẩn hóa định dạng

- `images/*.png`: ảnh xám 8-bit, giữ nguyên kích thước và cường độ kênh xám từ nguồn, không resize/crop/tăng tương phản.
- `labels/*.tif`: uint16, nền 0; mỗi mask riêng trong nguồn được gán một ID nguyên khác nhau theo thứ tự tên file mask.
- Hai file cùng tên là một cặp.
- Màu/đường biên trong preview chỉ phục vụ hiển thị. Preview được thu nhỏ và tăng tương phản, không dùng làm đầu vào thí nghiệm.

## Chia tập

Seed: `20261104`. Trong mỗi nhóm số nhân `1–10`, `11–30`, `31–60`, `61–100`, `>100`, trộn ngẫu nhiên ID rồi lấy khoảng 15% test, 15% val, phần còn lại train. Làm tròn trong từng nhóm nên tỷ lệ tổng có thể lệch nhẹ 70/15/15.

Demo: seed `20261105`, chọn tối đa 2 ảnh trong test của mỗi nhóm số nhân, tổng dự kiến 10 ảnh. Không dùng điểm của StarDist/Otsu/Watershed để chọn ảnh demo.

Danh sách đã ghi cố định trong `data/splits`. Augmentation, nếu có, chỉ thực hiện sau khi chia tập và chỉ trên train. Chưa xác minh được các nhóm theo người hiến mẫu hoặc thí nghiệm vì metadata này không có trong gói đang sử dụng; các tập hiện phân chia theo ID ảnh.

## Giới hạn khi dùng pretrained

Theo tài liệu StarDist, `2D_versatile_fluo` và `2D_paper_dsb2018` được huấn luyện trên các phần của DSB2018:
https://github.com/stardist/stardist#pretrained-models-for-2d

Vì vậy, test ở đây là tập giữ lại của DỰ ÁN, chưa chứng minh độc lập với dữ liệu huấn luyện của mô hình pretrained. Có thể dùng cho demo và so sánh tham chiếu; không tuyên bố là đánh giá trên ảnh mô hình chưa từng thấy nếu chưa kiểm tra provenance. Khi tự huấn luyện, không dùng val/test để huấn luyện.

Số lượng chính xác và thống kê các tập nằm trong `dataset_summary.json`.
