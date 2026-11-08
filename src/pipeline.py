"""Run a method on a frozen split, recording parameters and output provenance."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime

import numpy as np
from PIL import Image
import tifffile
from segmentation import segment_otsu, segment_watershed, StarDistPredictor


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_image(path):
    with Image.open(path) as image:
        result = np.array(image)
    if result.ndim != 2 or result.dtype != np.uint8:
        raise ValueError(f"Expected the prepared grayscale uint8 input: {path}")
    return result


def run_cli(method):
    parser = argparse.ArgumentParser(description=f"Run {method} without reading ground-truth labels")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=["demo", "train", "val", "test"], default="demo")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace outputs with the selected configuration")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or root / f"configs/python_{args.profile}.json"
    full_config = json.loads(config_path.read_text(encoding="utf-8"))
    params = dict(full_config[method])
    model_dir = root / params.pop("model_dir") if method == "stardist_python" else None
    provenance_params = dict(params)
    provenance_params["implementation_sha256"] = sha256(Path(__file__).with_name("segmentation.py"))
    if model_dir:
        provenance_params["model_weights_sha256"] = sha256(model_dir / "weights_best.h5")
        provenance_params["model_config_sha256"] = sha256(model_dir / "config.json")
    fingerprint = hashlib.sha256(json.dumps({"method": method, "parameters": provenance_params}, sort_keys=True).encode()).hexdigest()
    names = (root / "data/splits" / f"{args.split}.txt").read_text().split()
    if not names or len(set(names)) != len(names):
        raise ValueError("Split is empty or contains duplicate names")
    for name in names:
        if not (root / "data/prepared/images" / f"{name}.png").is_file():
            raise FileNotFoundError(name)
    output = root / "results" / method if args.profile == "baseline" else root / "results/optimized" / method
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    model = None
    records = []
    started = datetime.now().isoformat(timespec="seconds")
    for index, name in enumerate(names, 1):
        image_path = root / "data/prepared/images" / f"{name}.png"
        destination = output / f"{name}.tif"
        image = load_image(image_path)
        input_hash = sha256(image_path)
        previous = provenance.get(name, {})
        if destination.exists() and not args.overwrite:
            if previous.get("parameter_fingerprint") != fingerprint or previous.get("input_sha256") != input_hash or previous.get("output_sha256") != sha256(destination):
                raise RuntimeError(f"{name} exists with unknown/different provenance; use --overwrite only if replacement is intended")
            labels = tifffile.imread(destination)
            if labels.shape != image.shape or labels.dtype.kind not in "ui":
                raise ValueError("Invalid existing output")
            records.append({"image": name, "status": "SKIPPED_EXISTING", "objects": len(np.unique(labels[labels > 0]))})
            print(f"[{index}/{len(names)}] SKIP {name}", flush=True)
            continue
        if method == "stardist_python" and model is None:
            model = StarDistPredictor(model_dir, full_config.get("seed", 20261104))
        start = time.perf_counter()
        if method == "otsu":
            labels, info = segment_otsu(image, **params)
        elif method == "watershed":
            labels, info = segment_watershed(image, **params)
        else:
            labels, info = model.predict(image, **params)
        elapsed = time.perf_counter() - start
        if labels.shape != image.shape or labels.dtype.kind not in "ui" or labels.min() < 0:
            raise ValueError(f"Invalid prediction: {name}")
        labels = labels.astype(np.uint16 if labels.max() <= 65535 else np.uint32)
        temporary = destination.with_suffix(".part.tif")
        tifffile.imwrite(temporary, labels, photometric="minisblack")
        if not np.array_equal(labels, tifffile.imread(temporary)):
            raise RuntimeError("Label TIFF round-trip failed")
        temporary.replace(destination)
        count = int((np.unique(labels) > 0).sum())
        provenance[name] = {
            "parameter_fingerprint": fingerprint, "parameters": provenance_params,
            "input_sha256": input_hash, "output_sha256": sha256(destination),
            "split": args.split, "created": datetime.now().isoformat(timespec="seconds"),
        }
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        records.append({"image": name, "status": "OK", "objects": count, "processing_seconds": elapsed, **info})
        print(f"[{index}/{len(names)}] {name}: {count} objects ({elapsed:.3f}s)", flush=True)
    versions = {}
    for package in ["numpy", "scipy", "scikit-image", "Pillow", "tifffile", "stardist", "csbdeep", "tensorflow", "keras"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    log = {
        "method": method, "split": args.split, "profile": args.profile, "parameters": provenance_params,
        "parameter_fingerprint": fingerprint, "started": started,
        "python": platform.python_version(), "platform": platform.platform(),
        "device": "CPU", "versions": versions, "images": records,
        "timing_note": "Model construction excluded; first TensorFlow prediction includes initialization. Not a controlled benchmark.",
    }
    log_dir = root / "results/metrics"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{method}_{args.split}_{args.profile}_run_{datetime.now():%Y%m%d_%H%M%S}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("Run log:", log_path)
