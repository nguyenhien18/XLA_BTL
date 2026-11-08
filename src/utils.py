"""Shared paths and readers; images are inputs, labels are ground truth."""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/prepared/images"
LABEL_DIR = ROOT / "data/prepared/labels"
RESULT_DIR = ROOT / "results"


def load_pair(name):
    with Image.open(IMAGE_DIR / f"{name}.png") as image:
        x = np.array(image)
    with Image.open(LABEL_DIR / f"{name}.tif") as image:
        y = np.array(image)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"Invalid image-label pair: {name}")
    return x, y


def read_split(split):
    if split not in {"train", "val", "test", "demo"}:
        raise ValueError("Unknown split")
    return (ROOT / "data/splits" / f"{split}.txt").read_text().split()
