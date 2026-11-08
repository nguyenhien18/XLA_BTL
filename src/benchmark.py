"""Controlled CPU pipeline timings: resident inputs, per-image warm-up, repeated runs."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import random
import time
import numpy as np
from pipeline import load_image
from segmentation import segment_otsu, segment_watershed, StarDistPredictor
from evaluate import save_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=["demo", "val", "test"], default="test")
    parser.add_argument("--profile", choices=["baseline", "optimized"], default="optimized")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("At least three repetitions are required")
    root = args.root.resolve()
    config_path = root / f"configs/python_{args.profile}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    names = (root / "data/splits" / f"{args.split}.txt").read_text().split()
    images = {name: load_image(root / "data/prepared/images" / f"{name}.png") for name in names}
    params = dict(config["stardist_python"])
    model_dir = root / params.pop("model_dir")
    model = StarDistPredictor(model_dir, config["seed"])
    functions = {
        "otsu": lambda image: segment_otsu(image, **config["otsu"])[0],
        "watershed": lambda image: segment_watershed(image, **config["watershed"])[0],
        "stardist_python": lambda image: model.predict(image, **params)[0],
    }
    randomizer = random.Random(config["seed"])
    rows = []
    for index, (name, image) in enumerate(images.items(), 1):
        # Untimed per-image warm-up includes each input shape and all methods.
        for function in functions.values():
            function(image)
        for repeat in range(args.repeats):
            order = list(functions)
            randomizer.shuffle(order)
            for position, method in enumerate(order):
                start = time.perf_counter_ns()
                labels = functions[method](image)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                rows.append({"image": name, "method": method, "profile": args.profile,
                             "repeat": repeat + 1, "order": position + 1,
                             "width": image.shape[1], "height": image.shape[0],
                             "elapsed_ms": elapsed_ms, "objects": int((np.unique(labels) > 0).sum())})
        if index % 10 == 0 or index == len(images):
            print(f"Benchmarked {index}/{len(images)} images", flush=True)
    summaries = []
    for method in functions:
        values = np.array([r["elapsed_ms"] for r in rows if r["method"] == method])
        per_image_means = [np.mean([r["elapsed_ms"] for r in rows if r["method"] == method and r["image"] == name]) for name in names]
        summaries.append({"method": method, "split": args.split, "profile": args.profile,
                          "images": len(images), "repeats_per_image": args.repeats,
                          "mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
                          "p95_ms": float(np.percentile(values, 95)), "std_ms": float(np.std(values)),
                          "min_ms": float(np.min(values)), "max_ms": float(np.max(values)),
                          "median_per_image_mean_ms": float(np.median(per_image_means))})
    metrics = root / "results/metrics"
    prefix = f"benchmark_{args.split}_{args.profile}"
    save_csv(metrics / f"{prefix}_detail.csv", rows)
    save_csv(metrics / f"{prefix}_summary.csv", summaries)
    metadata = {"device": "CPU", "platform": platform.platform(), "processor": platform.processor(),
                "repeats": args.repeats, "warmup": "one untimed call per image per method",
                "order": "seeded random method order within each image/repetition", "seed": config["seed"],
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "included": ["preprocessing/normalization", "segmentation", "StarDist NMS and rasterization"],
                "excluded": ["input file IO", "model construction", "writing outputs", "evaluation", "object counting"],
                "thread_configuration": "StarDist: intra=4, inter=2, OMP_NUM_THREADS=4; other library defaults",
                "project_evaluation_tasks_parallel": False,
                "external_background_load_controlled": False,
                "note": "Measured in a separate run without project evaluation jobs, on this machine and dataset; not a universal speed ranking."}
    (metrics / f"{prefix}_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
