# Code Python hiện có

- `utils.py`: đọc dữ liệu, nhãn và các tập.
- `segmentation.py`: Otsu tự cài đặt bước tìm ngưỡng; quy trình Watershed; StarDist pretrained.
- `pipeline.py`: chạy theo split, ghi cấu hình/hash để tránh dùng lẫn kết quả khác tham số.
- `otsu.py`, `watershed.py`, `stardist_predict.py`: ba lệnh chạy chính.
- `evaluate.py`: ghép đối tượng IoU một-một, chỉ số và hình so sánh.
- `compare_methods.py`: tổng hợp ba phương pháp, đối chiếu Fiji/Python.

Hướng dẫn: `docs/PYTHON_START_HERE.md`.

Đã chạy và đánh giá cả ba phương pháp Python trên 10 ảnh demo. Otsu đã đối chiếu ngưỡng với scikit-image; Watershed đã kiểm tra trên hai đĩa tiếp xúc; 16 bộ chỉ số khớp hàm `stardist.matching.matching_dataset`.

Đã có `tune_parameters.py`, `benchmark.py`, `summarize_test.py`: khảo sát validation, đánh giá đủ 82 ảnh test ở hai profile và benchmark CPU có warm-up, 5 lần đo/ảnh. Xem `docs/TEST_RESULTS.md`. Đã bổ sung fine-tune và hậu xử lý tự cài; CNN vẫn dùng StarDist/TensorFlow.

- `custom_postprocess.py`: tự dựng đa giác, greedy NMS và gán nhãn; dùng pyclipper cho phép giao đa giác, skimage cho tô vùng.
- `check_custom_postprocess.py`: kiểm chứng hình học và so với thư viện trên 82 ảnh test.
- `finetune_stardist.py`: fine-tune trọng số trên train, chọn checkpoint theo validation; từ chối ghi đè một lần huấn luyện đã có.
- `evaluate_finetuned.py`: đánh giá checkpoint được chọn trên 82 test, đối chiếu chỉ số, xuất biểu đồ và bảng trước/sau.

Xem `docs/FINETUNE_RESULTS.md` để đọc cả kết quả giảm F1 và giới hạn thí nghiệm.
