"""Compare evaluated methods and independently match Fiji/Python StarDist outputs."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate import read_array, overlap_matrix, match_instances, scores, save_csv, boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", default="demo", choices=["demo", "val", "test"])
    parser.add_argument("--profile", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--figures", choices=["all", "demo"], default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    suffix = "_optimized" if args.profile == "optimized" else ""
    prediction_root = root / "results" if args.profile == "baseline" else root / "results/optimized"
    methods = ["otsu", "watershed", "stardist_python", "stardist_fiji"]
    titles = {"otsu": "Otsu", "watershed": "Watershed", "stardist_python": "StarDist Python", "stardist_fiji": "StarDist Fiji"}
    available = []
    summaries, per_image = [], {}
    for method in methods:
        summary_path = root / "results/metrics" / f"{method}_{args.split}{suffix}_summary.csv"
        per_path = root / "results/metrics" / f"{method}_{args.split}{suffix}_per_image.csv"
        if not summary_path.is_file() or not per_path.is_file():
            continue
        available.append(method)
        with summary_path.open(encoding="utf-8-sig") as stream:
            summaries.extend(dict(row) for row in csv.DictReader(stream))
        with per_path.open(encoding="utf-8-sig") as stream:
            per_image[method] = {row["image"]: row for row in csv.DictReader(stream) if float(row["iou_threshold"]) == 0.5}
    if not available:
        raise ValueError("No evaluation outputs found")
    names = (root / "data/splits" / f"{args.split}.txt").read_text().split()
    demo_names = set((root / "data/splits/demo.txt").read_text().split())
    plot_names = names if args.figures == "all" else [name for name in names if name in demo_names]
    for method in available:
        if set(per_image[method]) != set(names):
            raise ValueError(f"Different image coverage for {method}")
    fig_dir = root / "results/figures" / f"methods_{args.split}{suffix}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_csv(root / "results/metrics" / f"methods_{args.split}{suffix}_summary.csv", summaries)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), constrained_layout=True)
    x = np.arange(len(plot_names))
    width = 0.8 / len(available)
    for i, method in enumerate(available):
        axes[0].bar(x - 0.4 + width / 2 + i * width, [float(per_image[method][name]["F1"]) for name in plot_names], width, label=titles[method])
        selected = sorted([row for row in summaries if row["method"] == method], key=lambda r: float(r["iou_threshold"]))
        axes[1].plot([float(r["iou_threshold"]) for r in selected], [float(r["F1"]) for r in selected], marker="o", label=titles[method])
    axes[0].set_xticks(x, [name.replace("img_", "") for name in plot_names])
    axes[0].set_ylim(0, 1.06)
    axes[0].set_ylabel("Instance F1 at IoU >= 0.5")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("IoU matching threshold")
    axes[1].set_ylabel("Pooled instance F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{args.profile}: bars show {len(plot_names)} images; pooled curves use all {len(names)} images")
    fig.savefig(fig_dir / "overview.png", dpi=150)
    plt.close(fig)
    primary = [r for r in summaries if float(r["iou_threshold"]) == 0.5]
    report = [f"# So sánh các phương pháp — {args.split}, profile {args.profile}", "", "Cùng ảnh, cùng nhãn chuẩn và cùng phép ghép IoU một-một. " + ("Cấu hình được chọn chỉ trên validation." if args.profile == "optimized" else "Cấu hình baseline cố định trước khảo sát validation."), "", "| Phương pháp | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary:
        report.append(f"| {titles[row['method']]} | {row['TP']} | {row['FP']} | {row['FN']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['F1']):.4f} | {float(row['mean_matched_iou']):.4f} |")
    for name in plot_names:
        with Image.open(root / "data/prepared/images" / f"{name}.png") as image:
            raw = np.array(image)
        low, high = np.percentile(raw, [1, 99.8])
        display = np.clip((raw.astype(float) - low) / max(high - low, 1), 0, 1)
        gt = read_array(root / "data/prepared/labels" / f"{name}.tif")
        fig, axes = plt.subplots(1, len(available) + 1, figsize=(3.3 * (len(available) + 1), 4), constrained_layout=True)
        gt_rgb = np.repeat(display[..., None], 3, axis=2)
        gt_rgb[boundary(gt)] = (0.2, 1, 0.3)
        axes[0].imshow(gt_rgb)
        axes[0].set_title(f"Ground truth: {len(np.unique(gt[gt > 0]))}")
        for i, method in enumerate(available, 1):
            pred = read_array(prediction_root / method / f"{name}.tif")
            true_ids, pred_ids, matrix = overlap_matrix(gt, pred)
            matched = match_instances(true_ids, pred_ids, matrix, 0.5)
            rgb = np.repeat(display[..., None], 3, axis=2)
            edges = boundary(pred)
            rgb[edges] = (0.05, 0.8, 1)
            rgb[edges & np.isin(pred, matched["false_positive_ids"])] = (1, 0.15, 0.15)
            axes[i].imshow(rgb)
            row = per_image[method][name]
            axes[i].set_title(f"{titles[method]}: {row['predicted_count']}\nF1={float(row['F1']):.3f}")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"{name} | cyan: matched prediction; red: unmatched prediction | IoU >= 0.5")
        fig.savefig(fig_dir / f"{name}_comparison.png", dpi=140)
        plt.close(fig)
    if "stardist_python" in available and "stardist_fiji" in available:
        direct = []
        for name in names:
            fiji = read_array(prediction_root / "stardist_fiji" / f"{name}.tif")
            python = read_array(prediction_root / "stardist_python" / f"{name}.tif")
            a, b, matrix = overlap_matrix(fiji, python)
            result = match_instances(a, b, matrix, 0.5)
            direct.append({
                "image": name, "fiji_count": len(a), "python_count": len(b),
                "matched_pairs": result["TP"],
                "fiji_vs_python_F1": scores(result["TP"], result["FP"], result["FN"], result["matched_iou_sum"])["F1"],
                "mean_matched_iou": result["matched_iou_sum"] / result["TP"] if result["TP"] else 0,
                "F1_vs_GT_fiji": float(per_image["stardist_fiji"][name]["F1"]),
                "F1_vs_GT_python": float(per_image["stardist_python"][name]["F1"]),
            })
        save_csv(root / "results/metrics" / f"fiji_python_{args.split}{suffix}_comparison.csv", direct)
        report += ["", "## Đối chiếu trực tiếp Fiji và Python", "", "Fiji được dùng làm tham chiếu trong phép đối chiếu này, không phải nhãn chuẩn sinh học. F1 gần 1 giữa hai chương trình không tự chứng minh cả hai phân đoạn đúng.", "", "| Ảnh | Fiji | Python | F1 giữa hai đầu ra | Mean matched IoU |", "|---|---:|---:|---:|---:|"]
        for row in direct:
            report.append(f"| {row['image']} | {row['fiji_count']} | {row['python_count']} | {row['fiji_vs_python_F1']:.4f} | {row['mean_matched_iou']:.4f} |")
    report += ["", "Lưu ý: pretrained StarDist có sử dụng dữ liệu DSB2018, nên chưa xác minh tính độc lập với tập huấn luyện gốc. Không dùng kết quả demo để chọn tham số. Thời gian trong log chạy ban đầu chưa phải benchmark có warm-up/lặp đo được kiểm soát.", ""]
    (root / "docs" / f"methods_{args.split}{suffix}_comparison.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
