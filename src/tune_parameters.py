"""Select parameters exclusively on validation; cache StarDist sparse predictions."""
from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import time
import numpy as np
import tifffile
from pipeline import load_image, sha256
from segmentation import segment_otsu, segment_watershed, StarDistPredictor
from evaluate import overlap_matrix, match_instances, scores, save_csv


def configurations(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def cached_labels(image_shape, probability, distance, points, prob_threshold, nms_threshold, boundary):
    from stardist.nms import non_maximum_suppression_sparse
    from stardist.geometry import polygons_to_label
    keep = probability > prob_threshold
    if not np.any(keep):
        return np.zeros(image_shape, np.uint16)
    centers, retained_prob, retained_dist, _ = non_maximum_suppression_sparse(
        distance[keep], probability[keep], points[keep], b=boundary, nms_thresh=nms_threshold,
    )
    return polygons_to_label(retained_dist, centers, prob=retained_prob, shape=image_shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = root / "configs/validation_search.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = json.loads((root / "configs/python_baseline.json").read_text(encoding="utf-8"))
    if protocol["selection_split"] != "val":
        raise ValueError("Only validation selection is supported")
    names = (root / "data/splits/val.txt").read_text().split()
    test_names = set((root / "data/splits/test.txt").read_text().split())
    demo_names = set((root / "data/splits/demo.txt").read_text().split())
    if set(names) & (test_names | demo_names) or not names or len(set(names)) != len(names):
        raise ValueError("Validation overlaps test/demo, is empty or has duplicates")
    metrics = root / "results/metrics"
    metrics.mkdir(exist_ok=True)
    image_paths = [root / "data/prepared/images" / f"{name}.png" for name in names]
    label_paths = [root / "data/prepared/labels" / f"{name}.tif" for name in names]
    dataset_hash = hashlib.sha256("".join(name + sha256(x) + sha256(y) for name, x, y in zip(names, image_paths, label_paths)).encode()).hexdigest()
    thresholds = protocol["iou_thresholds"]
    selected_config = json.loads(json.dumps(baseline))
    selection = {"split": "val", "images": len(names), "dataset_sha256": dataset_hash,
                 "protocol_sha256": sha256(protocol_path), "objective": protocol["objective"],
                 "tie_break": protocol["tie_break"], "thresholds": thresholds, "methods": {}}
    all_rows = []
    # One model object for all validation images; ground truth is never passed to it.
    predictor = None
    for method in ["otsu", "watershed", "stardist_python"]:
        candidates = configurations(protocol[method])
        totals = [{t: {"TP": 0, "FP": 0, "FN": 0, "iou_sum": 0.0} for t in thresholds} for _ in candidates]
        started = time.perf_counter()
        print(f"{method}: {len(candidates)} configurations, {len(names)} validation images", flush=True)
        if method == "stardist_python":
            fixed = dict(baseline[method])
            model_dir = root / fixed.pop("model_dir")
            predictor = StarDistPredictor(model_dir, protocol["seed"])
            cache_dir = root / "results/cache/stardist_validation"
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_hash = sha256(model_dir / "weights_best.h5")
        for image_index, (name, image_path, label_path) in enumerate(zip(names, image_paths, label_paths), 1):
            image = load_image(image_path)
            gt = tifffile.imread(label_path)
            if method == "stardist_python":
                from csbdeep.utils import normalize
                minimum_prob = min(protocol[method]["probability"])
                cache_key = hashlib.sha256(json.dumps({
                    "input": sha256(image_path), "model": model_hash,
                    "model_config": sha256(model_dir / "config.json"),
                    "low": fixed["percentile_low"], "high": fixed["percentile_high"],
                    "boundary": fixed["boundary"], "minimum_probability": minimum_prob,
                    "stardist": __import__("stardist").__version__,
                }, sort_keys=True).encode()).hexdigest()
                cache_file = cache_dir / f"{name}_{cache_key}.npz"
                if cache_file.is_file():
                    with np.load(cache_file, allow_pickle=False) as saved:
                        probability, distance, points = saved["probability"], saved["distance"], saved["points"]
                else:
                    normalized = normalize(image, fixed["percentile_low"], fixed["percentile_high"])
                    probability, distance, points = predictor.model.predict_sparse(
                        normalized, prob_thresh=minimum_prob, b=fixed["boundary"], show_tile_progress=False,
                    )
                    np.savez_compressed(cache_file, probability=probability, distance=distance, points=points)
                if image_index == 1:
                    cached_baseline = cached_labels(image.shape, probability, distance, points,
                                                    fixed["probability"], fixed["nms"], fixed["boundary"])
                    direct_baseline, _ = predictor.predict(image, **fixed)
                    if not np.array_equal(cached_baseline, direct_baseline):
                        raise RuntimeError("Cached sparse pipeline differs from standard prediction")
                    print("  Cached prediction equivalence check: PASS", flush=True)
            for candidate_index, params in enumerate(candidates):
                if method == "otsu":
                    pred, _ = segment_otsu(image, **params)
                elif method == "watershed":
                    pred, _ = segment_watershed(image, **params)
                else:
                    pred = cached_labels(image.shape, probability, distance, points,
                                         params["probability"], params["nms"], fixed["boundary"])
                true_ids, pred_ids, iou = overlap_matrix(gt, pred)
                for threshold in thresholds:
                    match = match_instances(true_ids, pred_ids, iou, threshold)
                    record = totals[candidate_index][threshold]
                    for key in ["TP", "FP", "FN"]:
                        record[key] += match[key]
                    record["iou_sum"] += match["matched_iou_sum"]
            if image_index % 10 == 0 or image_index == len(names):
                print(f"  {image_index}/{len(names)} validation images", flush=True)
        method_rows = []
        for candidate_index, params in enumerate(candidates):
            row = {"method": method, "candidate": candidate_index, "parameters": json.dumps(params, sort_keys=True)}
            f1_values = []
            for threshold in thresholds:
                result = totals[candidate_index][threshold]
                result_scores = scores(result["TP"], result["FP"], result["FN"], result["iou_sum"])
                row[f"F1_IoU_{threshold}"] = result_scores["F1"]
                f1_values.append(result_scores["F1"])
            row["objective"] = float(np.mean(f1_values))
            method_rows.append(row)
        # max returns the first candidate when objective values tie.
        best = max(method_rows, key=lambda r: r["objective"])
        chosen = candidates[best["candidate"]]
        selected_config[method].update(chosen)
        selection["methods"][method] = {"parameters": chosen, "objective": best["objective"],
                                         "candidates": len(candidates), "selected_candidate": best["candidate"],
                                         "elapsed_seconds": time.perf_counter() - started}
        save_csv(metrics / f"{method}_validation_search.csv", method_rows)
        all_rows.extend(method_rows)
        print(f"BEST {method}: {chosen}; objective={best['objective']:.6f}", flush=True)
    selected_config["purpose"] = "Selected exclusively on the frozen validation split; test scores not used for selection."
    selected_config["selection"] = selection
    (root / "configs/python_optimized.json").write_text(json.dumps(selected_config, indent=2), encoding="utf-8")
    (metrics / "validation_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    save_csv(metrics / "validation_search_all.csv", all_rows)
    print("Saved configs/python_optimized.json. Test evaluation can now start.", flush=True)


if __name__ == "__main__":
    main()
