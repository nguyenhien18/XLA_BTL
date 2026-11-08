"""Export real StarDist network maps, 32-ray polygons and before/after NMS figures.

This script explains the frozen optimized model; it never tunes or changes labels.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from PIL import Image
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, hsv_to_rgb
from segmentation import StarDistPredictor
from pipeline import load_image


def save_figure(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def display_image(image):
    low, high = np.percentile(image, [1, 99.8])
    return np.clip((image.astype(float) - low) / max(high - low, 1), 0, 1)


def plot_polygon(axis, vertices, **kwargs):
    axis.add_patch(Polygon(np.column_stack([vertices[1], vertices[0]]), closed=True, fill=False, **kwargs))


def label_palette(count):
    # Black background; deterministic distinct colors for all displayed IDs.
    count = max(int(count), 1)
    hues = np.arange(count) / count
    hues = hues[np.random.default_rng(20261104).permutation(count)]
    hsv = np.column_stack([hues, np.full(count, 0.65), np.full(count, 0.95)])
    return ListedColormap(np.vstack([np.zeros((1, 3)), hsv_to_rgb(hsv)]))


def overlap_illustration(folder):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis in axes:
        axis.axis("off")
    axes[0].add_patch(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], fc="#6bb6de", alpha=0.7))
    axes[0].add_patch(Polygon([(2, 2), (7, 2), (7, 7), (2, 7)], fc="#ef9a56", alpha=0.85))
    axes[0].text(8, 8, "A", ha="center", fontsize=20)
    axes[0].text(4.5, 4.5, "B", ha="center", fontsize=20)
    axes[0].set(xlim=(-1, 11), ylim=(-1, 11), aspect="equal")
    axes[0].set_title("Synthetic geometry example\narea(A)=100; area(B)=25")
    axes[1].text(0.5, 0.55, "StarDist NMS overlap\nintersection / min(area A, area B)\n25 / 25 = 1.00", ha="center", va="center", fontsize=13, linespacing=1.8)
    axes[2].text(0.5, 0.55, "Evaluation IoU\nintersection / union\n25 / 100 = 0.25", ha="center", va="center", fontsize=13, linespacing=1.8)
    fig.suptitle("NMS threshold and evaluation IoU threshold use different overlap definitions", fontsize=14)
    save_figure(fig, folder, "07_nms_overlap_vs_evaluation_iou")


def workflow(folder, shape, grid_shape, rays, parameters):
    fig, axis = plt.subplots(figsize=(14, 5.5))
    axis.set(xlim=(0, 14), ylim=(0, 5.5))
    axis.axis("off")
    boxes = [
        (0.25, 3.15, 2.6, 1.45, f"1. Input image\n{shape[1]} x {shape[0]} x 1\npercentile 1 / 99.8"),
        (3.55, 3.15, 2.6, 1.45, "2. CNN / U-Net\n3 encoder-decoder levels\npretrained weights"),
        (7.1, 3.15, 3.35, 1.45, f"3. Network outputs\nscore: {grid_shape[1]} x {grid_shape[0]} x 1\ndistance: {grid_shape[1]} x {grid_shape[0]} x {rays}"),
        (7.1, 0.5, 3.35, 1.45, f"4. Candidate polygons\nscore > {parameters['probability']:.2f}\n{rays} radial distances per origin"),
        (3.55, 0.5, 2.6, 1.45, f"5. NMS\nremove duplicates\noverlap threshold {parameters['nms']:.2f}"),
        (0.25, 0.5, 2.6, 1.45, "6. Instance labels\n0 = background\n1, 2, ... = separate nuclei"),
    ]
    for x, y, width, height, text in boxes:
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08", fc="#edf4f8", ec="#267b9b", lw=1.5))
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11, linespacing=1.6)
    for start, end in [((2.95, 3.9), (3.45, 3.9)), ((6.25, 3.9), (7.0, 3.9)), ((8.8, 3.0), (8.8, 2.1)), ((7.0, 1.2), (6.25, 1.2)), ((3.45, 1.2), (2.95, 1.2))]:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17, lw=1.5, color="#405b6b"))
    axis.text(11.05, 2.7, "Inference uses only\nthe input image.\n\nGround truth is used\nfor evaluation,\nnot prediction.", va="center", fontsize=11, color="#334d5b", linespacing=1.6)
    fig.suptitle("StarDist 2D: actual model configuration used in this project", fontsize=15)
    save_figure(fig, folder, "01_stardist_workflow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--images", nargs="+", default=["img_0144", "img_0320"])
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "configs/python_optimized.json").read_text(encoding="utf-8"))
    parameters = dict(config["stardist_python"])
    model_dir = root / parameters.pop("model_dir")
    predictor = StarDistPredictor(model_dir, config["seed"])
    model = predictor.model
    from csbdeep.utils import normalize
    from stardist.nms import non_maximum_suppression_sparse
    from stardist.geometry import dist_to_coord, polygons_to_label, ray_angles
    from stardist.utils import edt_prob
    folder = root / "results/explanations/stardist"
    folder.mkdir(parents=True, exist_ok=True)
    overlap_illustration(folder)
    summary_lines = []
    model.keras_model.summary(print_fn=lambda text, **kwargs: summary_lines.append(text))
    (folder / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    metadata = {"model": "2D_versatile_fluo", "pretrained": True,
                "weights_sha256": hashlib.sha256((model_dir / "weights_best.h5").read_bytes()).hexdigest(),
                "parameters": parameters, "network_parameters": int(model.keras_model.count_params()),
                "grid": list(model.config.grid), "n_rays": int(model.config.n_rays),
                "nms_overlap_definition": "intersection_area / min(polygon_area_1, polygon_area_2)",
                "evaluation_iou_definition": "intersection_area / union_area",
                "images": []}
    for index, name in enumerate(args.images):
        image = load_image(root / "data/prepared/images" / f"{name}.png")
        gt = tifffile.imread(root / "data/prepared/labels" / f"{name}.tif")
        probability, distance = model.predict(normalize(image, parameters["percentile_low"], parameters["percentile_high"]), show_tile_progress=False)
        assert probability.ndim == 2 and distance.shape == probability.shape + (model.config.n_rays,)
        assert np.isfinite(probability).all() and np.isfinite(distance).all()
        assert probability.min() >= 0 and probability.max() <= 1
        grid = np.array(model.config.grid)
        candidate_grid = np.argwhere(probability > parameters["probability"])
        points = candidate_grid * grid
        p = probability[tuple(candidate_grid.T)]
        d = distance[tuple(candidate_grid.T)]
        kept_points, kept_p, kept_d, kept_indices = non_maximum_suppression_sparse(
            d, p, points, b=parameters["boundary"], nms_thresh=parameters["nms"],
        )
        labels = polygons_to_label(kept_d, kept_points, prob=kept_p, shape=image.shape)
        saved = tifffile.imread(root / "results/optimized/stardist_python" / f"{name}.tif")
        if not np.array_equal(labels, saved):
            raise RuntimeError(f"Explanation output differs from saved optimized prediction: {name}")
        coordinates = dist_to_coord(d, points)
        cmap = label_palette(labels.max())
        shown = display_image(image)
        gy, gx = grid
        ys = np.arange(probability.shape[0]) * gy
        xs = np.arange(probability.shape[1]) * gx
        extent = (-gx / 2, xs[-1] + gx / 2, ys[-1] + gy / 2, -gy / 2)
        if index == 0:
            workflow(folder, image.shape, probability.shape, model.config.n_rays, parameters)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
        axes[0].imshow(shown, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Input image")
        score_plot = axes[1].imshow(probability, cmap="magma", vmin=0, vmax=1, extent=extent, interpolation="nearest")
        axes[1].set_title(f"Predicted object score\nnetwork grid {probability.shape[1]} x {probability.shape[0]}")
        fig.colorbar(score_plot, ax=axes[1], fraction=0.046)
        axes[2].imshow(shown, cmap="gray", vmin=0, vmax=1)
        axes[2].contour(xs, ys, probability, levels=[parameters["probability"]], colors=["#34e2c4"], linewidths=0.7)
        axes[2].set_title(f"Score contour at {parameters['probability']:.2f}\nnot the final object boundary")
        for axis in axes:
            axis.set(xlim=(-0.5, image.shape[1] - 0.5), ylim=(image.shape[0] - 0.5, -0.5))
            axis.axis("off")
        fig.suptitle(f"{name}: score map is a network prediction, not a ground-truth mask")
        save_figure(fig, folder, f"{name}_02_probability_map")

        # Choose one unclipped predicted origin for a readable local illustration.
        # GT is used only to choose a crop/describe the illustration, never for NMS.
        choice = None
        for candidate in kept_indices:
            y, x = points[candidate]
            vertices = coordinates[candidate]
            gt_id = int(gt[int(y), int(x)])
            if gt_id and vertices[0].min() > 4 and vertices[1].min() > 4 and vertices[0].max() < image.shape[0] - 5 and vertices[1].max() < image.shape[1] - 5:
                local = np.flatnonzero(gt[points[:, 0], points[:, 1]] == gt_id)
                local_winners = np.intersect1d(local, kept_indices)
                if len(local) >= 8 and len(local_winners) == 1:
                    choice = int(candidate), gt_id, local, local_winners
                    break
        if choice is None:
            raise RuntimeError("No suitable local crop; choose a different explanation image")
        candidate, gt_id, local, local_winners = choice
        center = points[candidate]
        vertices = coordinates[candidate]
        pad = 8
        xlim = (max(-0.5, vertices[1].min() - pad), min(image.shape[1] - 0.5, vertices[1].max() + pad))
        ylim = (min(image.shape[0] - 0.5, vertices[0].max() + pad), max(-0.5, vertices[0].min() - pad))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
        for axis in axes:
            axis.imshow(shown, cmap="gray", vmin=0, vmax=1)
            axis.set(xlim=xlim, ylim=ylim)
            axis.axis("off")
        axes[0].scatter([center[1]], [center[0]], s=40, c="#ffce54", edgecolors="black")
        axes[0].set_title(f"Selected prediction origin\nscore={p[candidate]:.4f}")
        segments = [[(center[1], center[0]), (vertices[1, k], vertices[0, k])] for k in range(model.config.n_rays)]
        axes[1].add_collection(LineCollection(segments, colors="#ffcc44", linewidths=0.9))
        axes[1].scatter([center[1]], [center[0]], s=20, c="white")
        axes[1].set_title(f"{model.config.n_rays} predicted radial distances\nangles spaced by 360/{model.config.n_rays} degrees")
        plot_polygon(axes[2], vertices, edgecolor="#00f0b5", linewidth=2)
        axes[2].scatter(vertices[1], vertices[0], s=10, c="#00f0b5")
        axes[2].set_title("Connect ray endpoints\nstar-convex candidate polygon")
        fig.suptitle(f"{name}: one real candidate; the origin need not be the geometric centroid")
        save_figure(fig, folder, f"{name}_03_star_convex_rays")
        with (folder / f"{name}_rays.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["ray", "angle_radians", "distance_pixels", "origin_y", "origin_x", "endpoint_y", "endpoint_x"])
            for k, angle in enumerate(ray_angles(model.config.n_rays)):
                writer.writerow([k, float(angle), float(d[candidate, k]), int(center[0]), int(center[1]), float(vertices[0, k]), float(vertices[1, k])])

        shown_candidates = local[np.linspace(0, len(local) - 1, min(28, len(local))).astype(int)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
        for axis in axes:
            axis.imshow(shown, cmap="gray", vmin=0, vmax=1)
            axis.set(xlim=xlim, ylim=ylim)
            axis.axis("off")
        for other in shown_candidates:
            plot_polygon(axes[0], coordinates[other], edgecolor="#ffb45e", linewidth=0.65, alpha=0.5)
        axes[0].set_title(f"Before NMS: {len(local)} local candidates\nshowing {len(shown_candidates)} outlines for clarity")
        for other in shown_candidates:
            plot_polygon(axes[1], coordinates[other], edgecolor="#f19072", linewidth=0.6, alpha=0.35)
        for winner in local_winners:
            plot_polygon(axes[1], coordinates[winner], edgecolor="#06e6ab", linewidth=2.2)
            axes[1].scatter([points[winner, 1]], [points[winner, 0]], s=20, c="#06e6ab")
        axes[1].set_title(f"NMS threshold {parameters['nms']:.2f}\ngreen: candidate actually retained")
        for winner in local_winners:
            plot_polygon(axes[2], coordinates[winner], edgecolor="#06e6ab", linewidth=2.2)
        axes[2].set_title(f"After NMS: {len(local_winners)} retained locally\nduplicate candidates removed")
        fig.suptitle(f"{name}: global NMS keeps {len(kept_indices)} of {len(points)} candidates; panels show one crop")
        save_figure(fig, folder, f"{name}_04_nms_before_after")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
        axes[0].imshow(shown, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Input")
        axes[1].imshow(labels, cmap=cmap, vmin=0, vmax=max(1, labels.max()), interpolation="nearest")
        axes[1].set_title(f"Final labels: {len(kept_indices)} predictions\ncolors represent instance IDs")
        axes[2].imshow(shown, cmap="gray", vmin=0, vmax=1)
        for winner in kept_indices:
            plot_polygon(axes[2], coordinates[winner], edgecolor="#00e6ad", linewidth=0.65)
        axes[2].set_title("Retained polygons on input")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"{name}: labels reproduced pixel-for-pixel from the saved optimized prediction")
        save_figure(fig, folder, f"{name}_05_final_labels")

        if index == 0:
            target = edt_prob(gt[::gy, ::gx])
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
            axes[0].imshow(gt, cmap=label_palette(gt.max()), vmin=0, vmax=max(1, gt.max()), interpolation="nearest")
            axes[0].set_title("Provided ground-truth instance labels")
            axes[1].imshow(target, cmap="magma", vmin=0, vmax=1, extent=extent)
            axes[1].set_title("Reference score target from GT\nnormalized distance inside each object")
            axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1, extent=extent)
            axes[2].set_title("Network-predicted score\ncomputed from input image only")
            for axis in axes:
                axis.axis("off")
            fig.suptitle("Training-target illustration only: no training or fine-tuning was performed")
            save_figure(fig, folder, "06_target_vs_prediction")
        record = {"image": name, "input_shape_yx": list(image.shape), "score_shape_yx": list(probability.shape),
                  "distance_shape": list(distance.shape), "candidate_count": len(points),
                  "retained_count": len(kept_indices), "suppressed_count": len(points) - len(kept_indices),
                  "labels_identical_to_saved_optimized": True, "selected_origin_yx": center.tolist(),
                  "selected_score": float(p[candidate]), "selected_gt_id_for_display_only": gt_id,
                  "local_candidate_count": len(local), "local_retained_count": len(local_winners)}
        metadata["images"].append(record)
        np.savez_compressed(folder / f"{name}_network_outputs.npz", probability=probability, distance=distance,
                            candidate_points=points, candidate_scores=p, kept_candidate_indices=kept_indices)
        print(json.dumps(record), flush=True)
    (folder / "explanation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("All figures generated; network/NMS labels verified against saved outputs.", flush=True)


if __name__ == "__main__":
    main()
