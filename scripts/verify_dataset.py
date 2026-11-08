"""Read-only checks for the delivered image/label pairs and frozen splits."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows=list(csv.DictReader((ROOT/"data/manifest.csv").open(encoding="utf-8-sig")))
    names={r["name"] for r in rows}
    assert len(names)==len(rows), "Duplicate names"
    assert len({r['original_id'] for r in rows})==len(rows), "Duplicate source IDs"
    sets={s:set((ROOT/f"data/splits/{s}.txt").read_text().split()) for s in ['train','val','test','demo']}
    assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']), "Split overlap"
    assert sets['train']|sets['val']|sets['test']==names, "Split coverage"
    assert sets['demo']<=sets['test'], "Demo must be a subset of test"
    for directory, suffix in [('images','png'),('labels','tif')]:
        assert {p.stem for p in (ROOT/f'data/prepared/{directory}').glob(f'*.{suffix}')}==names, "Unexpected/missing files"
    total=0
    seen=set()
    for row in rows:
        assert row['name'] in sets[row['split']], "Manifest/split mismatch"
        assert (row['demo']=='True')==(row['name'] in sets['demo']), "Demo flag mismatch"
        for key in ['image','label']:
            p=ROOT/row[key]
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row[f'{key}_sha256'], f"File changed: {p}"
        with Image.open(ROOT/row['image']) as im: a=np.array(im)
        with Image.open(ROOT/row['label']) as im: b=np.array(im)
        assert a.ndim==2 and a.dtype==np.uint8, "Expected uint8 grayscale input"
        assert b.shape==a.shape==(int(row['height']),int(row['width'])), "Shape mismatch"
        assert b.dtype.kind=='u' and b.dtype.itemsize==2, "Expected uint16 labels"
        ids=np.unique(b);count=int((ids>0).sum())
        assert count==int(row['nuclei_count']) and 0 in ids, "Wrong instance count/background"
        pixel_hash=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
        assert pixel_hash==row['pixels_sha256'] and pixel_hash not in seen, "Pixel mismatch/duplicate"
        seen.add(pixel_hash);total+=count
    print(json.dumps({'verified_pairs':len(rows),'nuclei':total,'split_counts':{s:len(v) for s,v in sets.items()},'status':'PASS'},indent=2))


if __name__=='__main__':main()
