"""Student implementation: polar polygons, greedy NMS and label assembly.

NumPy: arrays; pyclipper: polygon intersection only; skimage: polygon rasterization.
No StarDist geometry or NMS routines are called by this module.
Overlap convention and integer clipping are compatible with StarDist 0.9.2.
"""
import numpy as np
import pyclipper
from skimage.draw import polygon


def radial_vertices(distances, origins, native_angles=False):
    d = np.asarray(distances, dtype=np.float32)
    p = np.asarray(origins, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] < 3 or p.shape != (len(d), 2):
        raise ValueError('Expected N x R distances (R>=3) and N x 2 origins')
    if not np.isfinite(d).all() or not np.isfinite(p).all() or np.any(d < 0):
        raise ValueError('Distances must be nonnegative and coordinates finite')
    angles = np.arange(d.shape[1]) * (2*np.pi/d.shape[1])
    if native_angles:
        # The reference C++ clipper constructs float angles before sin/cos.
        angles = (np.arange(d.shape[1], dtype=np.float32) * np.float32(2*np.pi/d.shape[1])).astype(float)
        y = (p[:,0,None].astype(float) + d*np.sin(angles)).astype(np.float32)
        x = (p[:,1,None].astype(float) + d*np.cos(angles)).astype(np.float32)
    else:
        y = (d*np.sin(angles)).astype(np.float32) + p[:,0,None]
        x = (d*np.cos(angles)).astype(np.float32) + p[:,1,None]
    return np.stack((y,x),axis=1)


def area(path):
    v = np.asarray(path, dtype=np.int64)
    if len(v) < 3: return 0.0
    return abs(float(np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1)))) / 2


def intersection_area(a, b):
    clip = pyclipper.Pyclipper()
    clip.AddPath(a, pyclipper.PT_SUBJECT, True)
    clip.AddPath(b, pyclipper.PT_CLIP, True)
    return sum(area(path) for path in clip.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))


def greedy_nms(distances, scores, origins, threshold=0.3):
    """Return original candidate indices in descending score order.

    Greedy selection: keep highest remaining score; suppress lower candidates
    when intersection/min(area1,area2) > threshold. Bounding boxes prune pairs.
    Integer coordinates use truncation, as in reference Clipper conversion.
    """
    scores = np.asarray(scores)
    vertices = radial_vertices(distances, origins, native_angles=True)
    if scores.shape != (len(vertices),) or not np.isfinite(scores).all() or not 0 <= threshold <= 1:
        raise ValueError('Invalid scores or NMS threshold')
    if not len(scores): return np.empty(0, dtype=int)
    paths = np.transpose(vertices[:,::-1,:],(0,2,1)).astype(np.int64).tolist()
    areas = np.array([area(p) for p in paths])
    low, high = vertices.min(axis=2), vertices.max(axis=2)
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros(len(scores), dtype=bool)
    kept = []
    for position, i in enumerate(order):
        if suppressed[i]: continue
        kept.append(int(i))
        later = order[position+1:]
        later = later[~suppressed[later]]
        intersects = np.all(low[later] <= high[i],axis=1) & np.all(low[i] <= high[later],axis=1)
        for j in later[intersects]:
            if min(areas[i],areas[j]) <= 0: continue
            overlap = np.float32(intersection_area(paths[i],paths[j]) / (min(areas[i],areas[j])+1e-10))
            if overlap > np.float32(threshold): suppressed[j] = True
    return np.asarray(kept,dtype=int)


def render_labels(distances, origins, scores, shape):
    vertices = radial_vertices(distances,origins)
    labels = np.zeros(shape,dtype=np.uint16 if len(vertices)<65536 else np.uint32)
    # Low-score polygons first; high-score polygons own overlapping pixels.
    for i in np.argsort(scores,kind='stable'):
        rr,cc=polygon(vertices[i,0],vertices[i,1],shape=shape)
        labels[rr,cc]=i+1
    return labels


def from_maps(probability, distances, shape, grid=(2,2), prob_threshold=0.55, nms_threshold=0.3):
    cells=np.argwhere(probability>prob_threshold)
    origins=cells*np.asarray(grid)
    scores=probability[tuple(cells.T)]
    rays=np.maximum(distances[tuple(cells.T)],1e-3)
    kept=greedy_nms(rays,scores,origins,nms_threshold)
    labels=render_labels(rays[kept],origins[kept],scores[kept],shape)
    return labels,dict(origins=origins,scores=scores,distances=rays,kept=kept)
