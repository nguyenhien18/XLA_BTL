"""Collect the frozen test results, before/after validation tuning, and timings."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate import save_csv

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    metrics = ROOT / "results/metrics"
    figures = ROOT / "results/figures/final_test"
    figures.mkdir(parents=True, exist_ok=True)
    methods = ["otsu", "watershed", "stardist_python"]
    titles = {"otsu": "Otsu", "watershed": "Watershed", "stardist_python": "StarDist Python"}
    timings = {r["method"]: r for r in read_csv(metrics / "benchmark_test_optimized_summary.csv")}
    selection = json.loads((metrics / "validation_selection.json").read_text(encoding="utf-8"))
    rows = []
    all_thresholds = []
    for method in methods:
        for profile, suffix in [("baseline", ""), ("optimized", "_optimized")]:
            summary = read_csv(metrics / f"{method}_test{suffix}_summary.csv")
            all_thresholds.extend(summary)
            row = dict(next(r for r in summary if float(r["iou_threshold"]) == 0.5))
            row["profile"] = profile
            row["mean_ms"] = timings[method]["mean_ms"] if profile == "optimized" else ""
            row["median_ms"] = timings[method]["median_ms"] if profile == "optimized" else ""
            rows.append(row)
    save_csv(metrics / "final_test_summary.csv", rows)
    save_csv(metrics / "final_test_all_thresholds.csv", all_thresholds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    x = np.arange(len(methods))
    for profile, shift, color in [("baseline", -0.18, "#8da8b7"), ("optimized", 0.18, "#177fa3")]:
        values = [float(next(r for r in rows if r["method"] == m and r["profile"] == profile)["F1"]) for m in methods]
        axes[0].bar(x + shift, values, 0.36, label=profile, color=color)
        for i, value in enumerate(values):
            axes[0].text(i + shift, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x, [titles[m] for m in methods])
    axes[0].set_ylabel("Pooled instance F1 at IoU >= 0.5")
    axes[0].set_ylim(0, 1.08)
    axes[0].legend()
    times = [float(timings[m]["median_ms"]) for m in methods]
    axes[1].bar(x, times, color="#bf7834")
    axes[1].set_xticks(x, [titles[m] for m in methods])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median CPU pipeline latency (ms, log scale)")
    for i, value in enumerate(times):
        axes[1].text(i, value * 1.12, f"{value:.2f} ms", ha="center", fontsize=9)
    axes[1].set_ylim(max(min(times) / 2, 0.01), max(times) * 2.5)
    fig.suptitle("82 test images | parameters chosen only on validation | optimized CPU timing")
    fig.savefig(figures / "overview.png", dpi=160)
    plt.close(fig)
    search = read_csv(metrics / "stardist_python_validation_search.csv")
    prob = sorted({json.loads(r["parameters"])["probability"] for r in search})
    nms = sorted({json.loads(r["parameters"])["nms"] for r in search})
    matrix = np.zeros((len(prob), len(nms)))
    for row in search:
        config = json.loads(row["parameters"])
        matrix[prob.index(config["probability"]), nms.index(config["nms"])] = float(row["objective"])
    fig, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    plot = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(nms)), nms)
    axis.set_yticks(range(len(prob)), prob)
    axis.set_xlabel("NMS threshold")
    axis.set_ylabel("Probability threshold")
    axis.set_title("StarDist validation: mean pooled F1 at IoU 0.5/0.6/0.7")
    for i in range(len(prob)):
        for j in range(len(nms)):
            axis.text(j, i, f"{matrix[i,j]:.6f}", ha="center", va="center", color="white" if matrix[i,j] < matrix.mean() else "black")
    fig.colorbar(plot, ax=axis)
    fig.savefig(figures / "stardist_validation_heatmap.png", dpi=160)
    plt.close(fig)
    text = ["# Kết quả thí nghiệm Python trên tập test", "",
            "Đã đánh giá đủ 82 ảnh test. Cấu hình được chọn bằng 82 ảnh validation; không dùng điểm test để chọn lại tham số. So sánh cả baseline và cấu hình được chọn để kiểm tra thay đổi có tổng quát hóa hay không.", "",
            "## Tham số chọn trên validation", ""]
    for method in methods:
        chosen = selection["methods"][method]
        text.append(f"- {titles[method]}: `{json.dumps(chosen['parameters'])}`; objective validation = {chosen['objective']:.6f}.")
    text += ["", "Objective là trung bình F1 gộp ở ba ngưỡng IoU 0.5/0.6/0.7. Đã khảo sát 12 cấu hình Otsu, 36 Watershed, 15 StarDist. Nếu đồng điểm, lấy cấu hình xuất hiện trước trong lưới đã khai báo.", "",
             "## Kết quả test tại IoU ≥ 0.5", "", "| Phương pháp | Profile | Chuẩn | Dự đoán | TP | FP | FN | Precision | Recall | F1 | PQ |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {titles[row['method']]} | {row['profile']} | {row['ground_truth_count']} | {row['predicted_count']} | {row['TP']} | {row['FP']} | {row['FN']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['F1']):.4f} | {float(row['panoptic_quality']):.4f} |")
    text += ["", "## Thay đổi trên test", ""]
    for method in methods:
        before = float(next(r for r in rows if r["method"] == method and r["profile"] == "baseline")["F1"])
        after = float(next(r for r in rows if r["method"] == method and r["profile"] == "optimized")["F1"])
        b = next(r for r in rows if r["method"] == method and r["profile"] == "baseline")
        o = next(r for r in rows if r["method"] == method and r["profile"] == "optimized")
        text.append(f"- {titles[method]}: F1 gộp {before:.4f} → {after:.4f}; chênh lệch {after-before:+.4f}. Precision {float(b['precision']):.4f} → {float(o['precision']):.4f}; recall {float(b['recall']):.4f} → {float(o['recall']):.4f}; macro F1 {float(b['macro_F1']):.4f} → {float(o['macro_F1']):.4f}.")
    text += ["", "Tên optimized nghĩa là được chọn trên validation, không đảm bảo cải thiện trên mọi ảnh hoặc mọi chỉ số test.", "",
             "## Thời gian chạy CPU — cấu hình optimized", "", "| Phương pháp | Mean (ms) | Median (ms) | P95 (ms) | Số ảnh | Lặp/ảnh |", "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        row = timings[method]
        text.append(f"| {titles[method]} | {float(row['mean_ms']):.3f} | {float(row['median_ms']):.3f} | {float(row['p95_ms']):.3f} | {row['images']} | {row['repeats_per_image']} |")
    text += ["", "Các ảnh được đọc vào RAM trước. Mỗi phương pháp được warm-up trên từng ảnh; đo lặp với thứ tự phương pháp trộn ngẫu nhiên bằng seed cố định. Thời gian bao gồm tiền xử lý và phân đoạn/NMS, loại trừ đọc/ghi file, tải mô hình, đếm và tính chỉ số. Ảnh có nhiều kích thước nên độ phân tán còn phụ thuộc kích thước và mật độ; đây là số đo trên máy hiện tại, không phải tốc độ phổ quát. Xem protocol benchmark trong results/metrics.", "",
             "## Giới hạn và minh họa", "",
             "- Các chỉ số tổng hợp dùng đủ 82 ảnh. Hình so sánh chi tiết chỉ xuất cho 10 ảnh demo đã chọn từ trước, không chọn theo kết quả đẹp.",
             "- StarDist sử dụng trọng số pretrained; nhóm chưa huấn luyện lại hoặc tự cài toàn bộ CNN/NMS.",
             "- Bộ DSB2018 có thể trùng dữ liệu huấn luyện gốc của pretrained. Tập test giữ lại trong dự án không tự chứng minh độc lập với quá trình huấn luyện gốc.",
             "- Fiji mới được đối chiếu trên 10 ảnh demo; bảng test này là ba phương pháp Python, không phải 82 ảnh Fiji.",
             "- F1 gộp dùng tổng TP/FP/FN, khác trung bình F1 theo ảnh. Bảng nhiều ngưỡng IoU giúp kiểm tra chất lượng biên.", "",
             "## Tái chạy", "", "`python src/tune_parameters.py` để khảo sát lại validation; `scripts/run_test_experiments.ps1` để chạy test và benchmark. Dùng Python trong `.venv`. Hai profile lưu riêng ở `results/<method>` và `results/optimized/<method>`.", ""]
    (ROOT / "docs/TEST_RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
