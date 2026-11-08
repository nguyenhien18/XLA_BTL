"""Build the fixed 2D bright-nuclei cohort from the original BBBC038 ZIP."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import io
import json
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "dcb6edc2690f137406638b2309581a71522c4dff19157d118453b448dcddcb68"
SEED = 20261104


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stratum(count):
    for maximum, name in [(10, "01_1-10"), (30, "02_11-30"), (60, "03_31-60"), (100, "04_61-100")]:
        if count <= maximum:
            return name
    return "05_over100"


def prepare():
    archive_path = ROOT / "data/downloads/stage1_train.zip"
    if digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Source archive SHA-256 differs from the frozen source.")
    for folder in ["data/prepared/images", "data/prepared/labels", "data/splits", "docs", "results/figures"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if list((ROOT / "data/prepared/images").glob("*.png")):
        raise SystemExit("Prepared data already exists. Run scripts/verify_dataset.py to verify it; no files overwritten.")
    rows, excluded = [], []
    seen_pixels = {}
    with zipfile.ZipFile(archive_path) as archive:
        images, masks = {}, defaultdict(list)
        for entry in archive.namelist():
            parts = entry.split("/")
            if len(parts) != 3 or not entry.lower().endswith(".png"):
                continue
            identifier, kind, _ = parts
            if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
                raise ValueError("Unexpected identifier")
            if kind == "images":
                if identifier in images:
                    raise ValueError("Repeated source ID")
                images[identifier] = entry
            elif kind == "masks":
                masks[identifier].append(entry)
        if set(images) != set(masks):
            raise ValueError("Image and mask source IDs do not match")
        for source_index, identifier in enumerate(sorted(images), 1):
            source_bytes = archive.read(images[identifier])
            with Image.open(io.BytesIO(source_bytes)) as image:
                pixels = np.array(image)
                gray = np.array(image.convert("L"))
            reason = None
            if pixels.ndim == 3 and not (np.array_equal(pixels[..., 0], pixels[..., 1]) and np.array_equal(pixels[..., 0], pixels[..., 2])):
                reason = "color_image_outside_scope"
            elif pixels.ndim == 3 and pixels.shape[-1] == 4 and not np.all(pixels[..., 3] == 255):
                reason = "nonopaque_alpha_requires_review"
            elif float(np.median(gray)) >= 100:
                reason = "bright_background_outside_scope"
            if reason:
                excluded.append({"original_id": identifier, "reason": reason})
                continue
            # For equal RGB channels, grayscale conversion preserves the channel values.
            if pixels.ndim == 3 and not np.array_equal(gray, pixels[..., 0]):
                raise ValueError("Grayscale conversion changed intensity")
            labels = np.zeros(gray.shape, dtype=np.uint16)
            entries = sorted(masks[identifier])
            if len(entries) > 65535:
                raise ValueError("Label capacity exceeded")
            for label_id, entry in enumerate(entries, 1):
                with Image.open(io.BytesIO(archive.read(entry))) as image:
                    mask = np.array(image)
                if mask.ndim != 2 or mask.shape != gray.shape or not np.any(mask):
                    raise ValueError(f"Invalid source mask {entry}")
                foreground = mask > 0
                if np.any(labels[foreground]):
                    raise ValueError(f"Overlapping source masks {identifier}")
                labels[foreground] = label_id
            foreground_mean = float(gray[labels > 0].mean())
            background_mean = float(gray[labels == 0].mean())
            if foreground_mean <= background_mean:
                excluded.append({"original_id": identifier, "reason": "annotated_foreground_not_brighter_than_background"})
                continue
            pixel_hash = hashlib.sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()
            if pixel_hash in seen_pixels:
                excluded.append({"original_id": identifier, "reason": "duplicate_pixels_of_" + seen_pixels[pixel_hash]})
                continue
            seen_pixels[pixel_hash] = identifier
            name = f"img_{len(rows)+1:04d}"
            image_path = ROOT / "data/prepared/images" / f"{name}.png"
            label_path = ROOT / "data/prepared/labels" / f"{name}.tif"
            Image.fromarray(gray).save(image_path)
            Image.fromarray(labels).save(label_path)
            with Image.open(label_path) as saved:
                if not np.array_equal(labels, np.array(saved)):
                    raise ValueError("Label write/read mismatch")
            rows.append({
                "name": name, "original_id": identifier,
                "image": image_path.relative_to(ROOT).as_posix(),
                "label": label_path.relative_to(ROOT).as_posix(),
                "width": gray.shape[1], "height": gray.shape[0],
                "nuclei_count": len(entries), "count_stratum": stratum(len(entries)),
                "foreground_mean": round(foreground_mean, 6),
                "background_mean": round(background_mean, 6),
                "gray_median": float(np.median(gray)),
                "split": "", "demo": False,
                "source_image_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "image_sha256": digest(image_path), "label_sha256": digest(label_path),
                "pixels_sha256": pixel_hash,
            })
            if source_index % 100 == 0:
                print(f"Read {source_index}/{len(images)} source images; retained {len(rows)}", flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["count_stratum"]].append(row)
    rng = random.Random(SEED)
    demo_rng = random.Random(SEED + 1)
    for group in sorted(groups):
        members = list(groups[group])
        rng.shuffle(members)
        n_test = round(len(members)*0.15)
        n_val = round(len(members)*0.15)
        for i, row in enumerate(members):
            row["split"] = "test" if i < n_test else "val" if i < n_test+n_val else "train"
        candidates = sorted([r for r in members if r["split"] == "test"], key=lambda r:r["name"])
        for row in demo_rng.sample(candidates, min(2, len(candidates))):
            row["demo"] = True
    for split in ("train", "val", "test", "demo"):
        chosen = [r["name"] for r in rows if (r["demo"] if split == "demo" else r["split"] == split)]
        (ROOT / "data/splits" / f"{split}.txt").write_text("\n".join(chosen)+"\n", encoding="utf-8")
    write_csv(ROOT / "data/manifest.csv", rows)
    write_csv(ROOT / "data/excluded.csv", excluded)
    demo_rows = [r for r in rows if r["demo"]]
    write_csv(ROOT / "data/demo_manifest.csv", demo_rows)
    summary = {
        "source_images": len(images), "selected_images": len(rows),
        "selected_nuclei": sum(r["nuclei_count"] for r in rows),
        "excluded": dict(Counter(r["reason"] for r in excluded)),
        "seed": SEED, "demo_seed": SEED+1,
        "splits": {s: {"images": sum(r["split"]==s for r in rows), "nuclei": sum(r["nuclei_count"] for r in rows if r["split"]==s)} for s in ["train", "val", "test"]},
        "strata": {g:dict(Counter(r["split"] for r in members)) for g,members in groups.items()},
        "demo_images": [r["name"] for r in demo_rows],
        "source_sha256": ARCHIVE_SHA256,
        "geometry_changed": False, "grayscale_intensities_preserved": True,
        "pretrained_test_independence": "Not established: StarDist pretrained models used DSB2018 subsets.",
    }
    (ROOT / "docs/dataset_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    preview(demo_rows)
    print(json.dumps(summary, indent=2), flush=True)


def preview(rows):
    sheet = Image.new("RGB", (1040, ((len(rows)+1)//2)*292), "#f2f4f6")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
    except OSError:
        font = ImageFont.load_default()
    for index, row in enumerate(rows):
        with Image.open(ROOT/row["image"]) as image:
            a=np.array(image)
        with Image.open(ROOT/row["label"]) as image:
            labels=np.array(image)
        low,high=np.percentile(a,[1,99.8])
        normalized=np.clip((a.astype(float)-low)/max(high-low,1)*255,0,255).astype("uint8")
        shown=np.repeat(normalized[...,None],3,axis=-1)
        edge=np.zeros(labels.shape,bool)
        edge[:-1,:]|=(labels[:-1,:]!=labels[1:,:])&(labels[:-1,:]>0)
        edge[1:,:]|=(labels[1:,:]!=labels[:-1,:])&(labels[1:,:]>0)
        edge[:,:-1]|=(labels[:,:-1]!=labels[:,1:])&(labels[:,:-1]>0)
        edge[:,1:]|=(labels[:,1:]!=labels[:,:-1])&(labels[:,1:]>0)
        overlay=shown.copy();overlay[edge]=(0,255,110)
        x,y=(index%2)*520,(index//2)*292
        draw.text((x+5,y+4),f"{row['name']} | {row['width']}x{row['height']} | {row['nuclei_count']} nuclei",font=font,fill="black")
        for j, array in enumerate([shown,overlay]):
            image=Image.fromarray(array);image.thumbnail((250,250))
            sheet.paste(image,(x+5+j*260,y+32))
    sheet.save(ROOT/"docs/demo_preview.png")


if __name__ == "__main__":
    prepare()
