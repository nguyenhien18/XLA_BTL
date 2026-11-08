"""Verify the bundled 10 pairs and copy them to the default experiment paths."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'data/demo'
    expected = json.loads((source / 'checksums.json').read_text())
    names = (ROOT / 'data/splits/demo.txt').read_text().split()
    required = {f'images/{name}.png' for name in names} | {f'labels/{name}.tif' for name in names}
    if set(expected) != required or len(names) != 10:
        raise ValueError('Demo manifest must describe exactly 10 image/label pairs.')
    destinations = []
    for relative, digest in expected.items():
        origin = source / relative
        target = ROOT / 'data/prepared' / relative
        if hashlib.sha256(origin.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Demo source checksum differs: {relative}')
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Existing prepared data differs; no file overwritten: {target}')
        destinations.append((origin, target))
    for origin, target in destinations:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
    print('Verified 10 demo pairs. Ready in data/prepared/images and data/prepared/labels.')
    print('For all 545 images, run full preparation in a separate fresh clone; see README.md.')


if __name__ == '__main__':
    main()
