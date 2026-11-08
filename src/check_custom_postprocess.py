"""Compare independently implemented postprocessing against frozen reference."""
from pathlib import Path
import json,time,hashlib,argparse
import numpy as np
import tifffile
from custom_postprocess import radial_vertices,greedy_nms,from_maps
from segmentation import StarDistPredictor
from pipeline import load_image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split', choices=['demo','test'],default='demo')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    # Behavioral cases distinguish polygon-overlap NMS from bbox/IoU NMS.
    rays=np.ones((3,32),np.float32)*10
    rays[1]=2
    origins=np.array([[20,20],[20,20],[80,80]])
    assert greedy_nms(rays,[.9,.8,.7],origins,.3).tolist()==[0,2]
    assert greedy_nms(rays,[.8,.9,.7],origins,.3).tolist()==[1,2]
    assert greedy_nms(np.empty((0,32)),[],np.empty((0,2))).size==0
    vertices=radial_vertices(np.ones((1,4))*3,[[10,20]])
    np.testing.assert_allclose(vertices,[[[10,13,10,7],[23,20,17,20]]],atol=1e-6)
    model=StarDistPredictor(root/'models/2D_versatile_fluo').model
    from csbdeep.utils import normalize
    from stardist.nms import non_maximum_suppression_sparse
    from stardist.geometry import dist_to_coord
    out=root/'results/custom_stardist';out.mkdir(exist_ok=True)
    records=[]
    for name in (root/f'data/splits/{args.split}.txt').read_text().split():
        image=load_image(root/f'data/prepared/images/{name}.png')
        prob,dist=model.predict(normalize(image,1,99.8),show_tile_progress=False)
        start=time.perf_counter(); labels,detail=from_maps(prob,dist,image.shape)
        elapsed=time.perf_counter()-start
        keep=detail['kept']
        ref=non_maximum_suppression_sparse(detail['distances'],detail['scores'],detail['origins'],nms_thresh=.3)[3]
        coords=radial_vertices(detail['distances'],detail['origins'])
        delta=float(np.max(np.abs(coords-dist_to_coord(detail['distances'],detail['origins']))))
        saved=tifffile.imread(root/f'results/optimized/stardist_python/{name}.tif')
        rec=dict(image=name,candidates=len(detail['scores']),custom_kept=len(keep),reference_kept=len(ref),same_kept_indices=bool(np.array_equal(keep,ref)),same_saved_label_pixels=bool(np.array_equal(labels,saved)),different_pixels=int(np.count_nonzero(labels!=saved)),max_vertex_error=delta,custom_postprocess_seconds=elapsed)
        tifffile.imwrite(out/f'{name}.tif',labels,photometric='minisblack')
        records.append(rec); print(json.dumps(rec),flush=True)
    result=dict(scope=f'{args.split}: {len(records)} images, optimized thresholds .55/.30; no timing claim against native code',behavioral_cases_passed=4,implementation_sha256=hashlib.sha256((root/'src/custom_postprocess.py').read_bytes()).hexdigest(),records=records)
    suffix='' if args.split=='demo' else '_test'
    (root/f'results/metrics/custom_postprocess_validation{suffix}.json').write_text(json.dumps(result,indent=2))
    assert all(r['same_kept_indices'] and r['same_saved_label_pixels'] for r in records), 'See validation differences; do not claim exact match'

if __name__=='__main__':main()
