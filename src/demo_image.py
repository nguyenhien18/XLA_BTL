"""Read one image, run fresh segmentation, and display the evolving results."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import time

import numpy as np
from PIL import Image
import tifffile
from segmentation import segment_otsu, segment_watershed, StarDistPredictor
from custom_postprocess import from_maps


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=root/'data/prepared/images/img_0320.png')
    parser.add_argument('--profile', choices=['baseline', 'optimized'], default='baseline')
    parser.add_argument('--no-show', action='store_true', help='Save outputs without opening a window')
    args = parser.parse_args()
    path = args.image.resolve()
    with Image.open(path) as source:
        raw = np.array(source)
    if raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError('Choose a 2D grayscale uint8 image from data/prepared/images.')
    config = json.loads((root/f'configs/python_{args.profile}.json').read_text())
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = root/'results/live_demo'/f'{path.stem}_{args.profile}_{stamp}'
    output.mkdir(parents=True)

    import matplotlib
    matplotlib.use('Agg' if args.no_show else 'TkAgg')
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    if not args.no_show:
        plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    low, high = np.percentile(raw, [1, 99.8])
    display = np.clip((raw.astype(float)-low)/max(high-low, 1), 0, 1)
    axes = axes.ravel()
    for ax in axes:
        ax.imshow(display, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    axes[0].set_title(f'Input: {path.name}\n{raw.shape[1]} x {raw.shape[0]} pixels')
    titles = ['Otsu', 'Watershed', 'StarDist Python']
    for ax, title in zip(axes[1:], titles):
        ax.set_title(title+' — waiting')
    fig.suptitle(f'Fresh inference | {args.profile} | input image only; no ground truth')
    def refresh():
        if not args.no_show:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(.05)
    refresh()
    print(f'INPUT: {path}\nShape: {raw.shape}, dtype: {raw.dtype}', flush=True)
    records = []
    predictions = {}
    for index, (method, title) in enumerate(zip(['otsu','watershed','stardist_python'], titles), 1):
        axes[index].set_title(title+' — running')
        refresh()
        print(f'[{index}/3] Running {title} on the input image...', flush=True)
        start = time.perf_counter()
        if method == 'otsu':
            labels, details = segment_otsu(raw, **config[method])
        elif method == 'watershed':
            labels, details = segment_watershed(raw, **config[method])
        else:
            settings = config[method]
            model = StarDistPredictor(root/settings['model_dir'], config['seed']).model
            from csbdeep.utils import normalize
            prob, dist = model.predict(normalize(raw, settings['percentile_low'], settings['percentile_high']), show_tile_progress=False)
            labels, candidates = from_maps(prob, dist, raw.shape, grid=model.config.grid,
                                           prob_threshold=settings['probability'], nms_threshold=settings['nms'])
            details = dict(candidates=len(candidates['scores']), kept=len(candidates['kept']),
                           postprocessing='custom polygon construction and NMS')
        elapsed = time.perf_counter()-start
        count = int(len(np.unique(labels[labels>0])))
        # Save fresh labels only inside this run's output directory.
        tifffile.imwrite(output/f'{method}.tif', labels.astype(np.uint16), photometric='minisblack')
        predictions[method] = labels
        rgb = np.repeat(display[...,None], 3, axis=2)
        rgb[find_boundaries(labels, mode='inner')] = (0.0, 0.85, 1.0)
        axes[index].images[0].set_data(rgb)
        axes[index].set_title(f'{title}: {count} predicted objects')
        refresh()
        records.append(dict(method=method, objects=count, elapsed_seconds_including_initialization=elapsed, details=details))
        print(f'  FINISHED: {count} predicted objects ({elapsed:.2f} s, includes initialization)', flush=True)
    picture = output/'comparison.png'
    fig.savefig(picture, dpi=160)
    metadata = dict(input=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    profile=args.profile, config=config, fresh_inference=True, ground_truth_used=False,
                    results=records, note='Elapsed times are demo logs, not a controlled benchmark.')
    (output/'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'DONE\nResults: {output}\nImage: {picture}', flush=True)
    if args.no_show:
        plt.close(fig)
    else:
        print('Close the result window to return to PowerShell.', flush=True)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    main()
