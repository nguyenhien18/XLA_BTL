"""Evaluate instance masks against ground truth using one-to-one IoU matching.

Example: python src/evaluate.py --method stardist_fiji --split demo
"""
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_array(path):
    with Image.open(path) as image:
        result = np.array(image)
    if result.ndim != 2 or result.dtype.kind not in "ui":
        raise ValueError(f"Expected a 2D nonnegative integer label image: {path}")
    return result


def overlap_matrix(gt, pred):
    true_ids, true_inverse = np.unique(gt, return_inverse=True)
    pred_ids, pred_inverse = np.unique(pred, return_inverse=True)
    all_counts = np.bincount(
        true_inverse.ravel() * len(pred_ids) + pred_inverse.ravel(),
        minlength=len(true_ids) * len(pred_ids),
    ).reshape(len(true_ids), len(pred_ids))
    true_area = all_counts.sum(axis=1)
    pred_area = all_counts.sum(axis=0)
    true_keep, pred_keep = true_ids != 0, pred_ids != 0
    intersection = all_counts[np.ix_(true_keep, pred_keep)]
    union = true_area[true_keep, None] + pred_area[None, pred_keep] - intersection
    iou = np.divide(intersection, union, out=np.zeros(intersection.shape, float), where=union > 0)
    return true_ids[true_keep], pred_ids[pred_keep], iou


def match_instances(true_ids, pred_ids, iou, threshold):
    n_true, n_pred = len(true_ids), len(pred_ids)
    pairs = []
    if min(n_true, n_pred):
        # Prioritize the number of eligible one-to-one matches, then total IoU.
        # The tie-break term cannot outweigh an additional eligible match.
        cost = -(iou >= threshold).astype(float) - iou / (2 * min(n_true, n_pred))
        true_idx, pred_idx = linear_sum_assignment(cost)
        for i, j in zip(true_idx, pred_idx):
            if iou[i, j] >= threshold:
                pairs.append({"true_id": int(true_ids[i]), "pred_id": int(pred_ids[j]), "iou": float(iou[i, j])})
    matched_true = {p["true_id"] for p in pairs}
    matched_pred = {p["pred_id"] for p in pairs}
    assert len(matched_true) == len(matched_pred) == len(pairs)
    return {
        "TP": len(pairs), "FP": n_pred - len(pairs), "FN": n_true - len(pairs),
        "matched_iou_sum": sum(p["iou"] for p in pairs),
        "pairs": pairs,
        "missed_ids": [int(x) for x in true_ids if x not in matched_true],
        "false_positive_ids": [int(x) for x in pred_ids if x not in matched_pred],
    }


def scores(tp, fp, fn, iou_sum):
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "mean_matched_iou": iou_sum / tp if tp else 0.0,
        "panoptic_quality": iou_sum / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0,
    }


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boundary(mask):
    result = np.zeros(mask.shape, bool)
    result[:-1, :] |= (mask[:-1, :] != mask[1:, :]) & (mask[:-1, :] != 0)
    result[1:, :] |= (mask[1:, :] != mask[:-1, :]) & (mask[1:, :] != 0)
    result[:, :-1] |= (mask[:, :-1] != mask[:, 1:]) & (mask[:, :-1] != 0)
    result[:, 1:] |= (mask[:, 1:] != mask[:, :-1]) & (mask[:, 1:] != 0)
    return result


def plot_comparison(raw, gt, pred, matched, row, path):
    low, high = np.percentile(raw, (1, 99.8))
    display = np.clip((raw.astype(float) - low) / max(high - low, 1), 0, 1)
    ground = np.repeat(display[..., None], 3, axis=2)
    prediction = ground.copy()
    ground[boundary(gt)] = (0.15, 1.0, 0.35)
    prediction[boundary(pred)] = (0.1, 0.8, 1.0)
    ground[boundary(gt) & np.isin(gt, matched["missed_ids"])] = (1, 0.55, 0.05)
    prediction[boundary(pred) & np.isin(pred, matched["false_positive_ids"])] = (1, 0.15, 0.15)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    axes[0].imshow(display, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input image")
    axes[1].imshow(ground)
    axes[1].set_title(f"Ground truth: {row['ground_truth_count']}\nGreen: matched; orange: missed")
    axes[2].imshow(prediction)
    axes[2].set_title(f"Prediction: {row['predicted_count']}\nCyan: matched; red: false positive")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{row['image']} | IoU >= {row['iou_threshold']:.2f} | TP={row['TP']} FP={row['FP']} FN={row['FN']} | F1={row['F1']:.3f}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--method", default="stardist_fiji", choices=["stardist_fiji", "stardist_python", "otsu", "watershed"])
    parser.add_argument("--split", default="demo", choices=["demo", "train", "val", "test"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--profile", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--figures", choices=["all", "demo"], default="all")
    args = parser.parse_args()
    if any(not 0 < x <= 1 for x in args.thresholds) or len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError("IoU thresholds must be unique values in (0,1]")
    root = args.root.resolve()
    names = (root / "data/splits" / f"{args.split}.txt").read_text().split()
    if not names or len(set(names)) != len(names):
        raise ValueError("Empty or duplicate image list")
    prefix = f"{args.method}_{args.split}" + ("_optimized" if args.profile == "optimized" else "")
    prediction_root = root / "results" if args.profile == "baseline" else root / "results/optimized"
    metrics = root / "results/metrics"
    figures = root / "results/figures" / prefix
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    primary_threshold = 0.5 if 0.5 in args.thresholds else args.thresholds[0]
    demo_names = set((root / "data/splits/demo.txt").read_text().split())
    rows, details = [], []
    for name in names:
        gt = read_array(root / "data/prepared/labels" / f"{name}.tif")
        pred = read_array(prediction_root / args.method / f"{name}.tif")
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: {name}")
        true_ids, pred_ids, iou = overlap_matrix(gt, pred)
        gt_binary, pred_binary = gt > 0, pred > 0
        intersection = int((gt_binary & pred_binary).sum())
        union = int((gt_binary | pred_binary).sum())
        denominator = int(gt_binary.sum() + pred_binary.sum())
        for threshold in args.thresholds:
            matched = match_instances(true_ids, pred_ids, iou, threshold)
            tp, fp, fn = matched["TP"], matched["FP"], matched["FN"]
            row = {
                "image": name, "method": args.method, "split": args.split, "profile": args.profile,
                "iou_threshold": threshold, "ground_truth_count": len(true_ids),
                "predicted_count": len(pred_ids), "TP": tp, "FP": fp, "FN": fn,
                **scores(tp, fp, fn, matched["matched_iou_sum"]),
                "matched_iou_sum": matched["matched_iou_sum"],
                "count_absolute_error": abs(len(pred_ids) - len(true_ids)),
                "binary_iou": intersection / union if union else 1.0,
                "binary_dice": 2 * intersection / denominator if denominator else 1.0,
            }
            assert row["TP"] + row["FN"] == row["ground_truth_count"]
            assert row["TP"] + row["FP"] == row["predicted_count"]
            rows.append(row)
            details.append({"image": name, "iou_threshold": threshold, **matched})
            if threshold == primary_threshold and (args.figures == "all" or args.figures == "demo" and name in demo_names):
                with Image.open(root / "data/prepared/images" / f"{name}.png") as image:
                    raw = np.array(image.convert("L"))
                plot_comparison(raw, gt, pred, matched, row, figures / f"{name}_comparison.png")
        print(f"Evaluated {name}: GT={len(true_ids)}, prediction={len(pred_ids)}", flush=True)
    summaries = []
    for threshold in args.thresholds:
        selected = [r for r in rows if r["iou_threshold"] == threshold]
        tp, fp, fn = (sum(r[key] for r in selected) for key in ["TP", "FP", "FN"])
        iou_sum = sum(r["matched_iou_sum"] for r in selected)
        summaries.append({
            "method": args.method, "split": args.split, "profile": args.profile, "images": len(names),
            "iou_threshold": threshold, "ground_truth_count": sum(r["ground_truth_count"] for r in selected),
            "predicted_count": sum(r["predicted_count"] for r in selected),
            "TP": tp, "FP": fp, "FN": fn, **scores(tp, fp, fn, iou_sum),
            "macro_F1": float(np.mean([r["F1"] for r in selected])),
            "mean_absolute_count_error": float(np.mean([r["count_absolute_error"] for r in selected])),
        })
    primary = [r for r in rows if r["iou_threshold"] == primary_threshold]
    plot_rows = primary if args.figures == "all" else [r for r in primary if r["image"] in demo_names]
    save_csv(metrics / f"{prefix}_per_image.csv", rows)
    save_csv(metrics / f"{prefix}_summary.csv", summaries)
    (metrics / f"{prefix}_matches.json").write_text(json.dumps(details, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    x = np.arange(len(plot_rows))
    axes[0].bar(x - 0.2, [r["ground_truth_count"] for r in plot_rows], 0.4, label="Ground truth")
    axes[0].bar(x + 0.2, [r["predicted_count"] for r in plot_rows], 0.4, label="Prediction")
    axes[0].set_xticks(x, [r["image"].replace("img_", "") for r in plot_rows])
    axes[0].set_ylabel("Nuclei count")
    axes[0].set_title("Counts alone do not measure segmentation accuracy")
    axes[0].legend()
    axes[1].bar(x, [r["F1"] for r in plot_rows], color="#2489b6")
    axes[1].set_xticks(x, [r["image"].replace("img_", "") for r in plot_rows])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Instance F1")
    axes[1].set_title(f"One-to-one matching at IoU >= {primary_threshold}")
    for i, row in enumerate(plot_rows):
        axes[1].text(i, row["F1"] + 0.015, f"{row['F1']:.3f}", ha="center", fontsize=8)
    fig.suptitle(f"{args.method} / {args.profile}: showing {len(plot_rows)} images; aggregate metrics use all {len(names)}")
    fig.savefig(figures / "overview.png", dpi=150)
    plt.close(fig)

    report = [f"# So sánh {args.method} với nhãn chuẩn — tập {args.split}, profile {args.profile}", "",
              f"Đánh giá {len(names)} ảnh. Ghép một-một theo IoU, ưu tiên số cặp đạt ngưỡng rồi tổng IoU. Các hình minh họa dùng ngưỡng {primary_threshold}.", "",
              "TP: đối tượng ghép đúng. FP: dự đoán không ghép được. FN: đối tượng thật không ghép được. FP/FN có thể phản ánh lỗi biên, gộp/tách sai hoặc bỏ sót; không chỉ là lỗi đếm.", "",
              "## Kết quả từng ảnh", "", "| Ảnh | Chuẩn | Dự đoán | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary:
        report.append(f"| {row['image']} | {row['ground_truth_count']} | {row['predicted_count']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['F1']:.4f} |")
    report += ["", "## Tổng hợp theo ngưỡng IoU", "", "| IoU | TP | FP | FN | Precision | Recall | F1 gộp | Mean matched IoU | PQ |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        report.append(f"| {row['iou_threshold']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['F1']:.4f} | {row['mean_matched_iou']:.4f} | {row['panoptic_quality']:.4f} |")
    report += ["", "F1 gộp được tính từ tổng TP/FP/FN; macro_F1 là trung bình F1 theo ảnh. Hai đại lượng khác nhau. Mean matched IoU chỉ xét các cặp đạt ngưỡng nên cần đọc cùng recall/F1, không dùng riêng để kết luận.", "",
               "Binary IoU và Dice trong CSV là chỉ số phụ cho nền/đối tượng, không thể thay thế đánh giá từng cá thể.", "",
               f"Kết quả dùng đủ {len(names)} ảnh trong tập {args.split}. Hình minh họa có thể chỉ dùng tập con đã chọn trước; số liệu tổng hợp vẫn dùng toàn bộ danh sách. Tính độc lập với dữ liệu huấn luyện gốc của pretrained chưa được xác minh.", "",
               "Hình so sánh chỉ tăng tương phản khi hiển thị, không thay đổi dữ liệu đánh giá.", "",
               f"Tái chạy: `python src/evaluate.py --method {args.method} --split {args.split} --profile {args.profile}`.", ""]
    (root / "docs" / f"{prefix}_comparison.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"summary": summaries, "per_image_at_primary_threshold": primary}, indent=2))


if __name__ == "__main__":
    main()
