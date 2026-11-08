"""Segmentation steps shared by the CLI programs and validation experiments."""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def otsu_threshold(image):
    """Implement Otsu's between-class variance search for uint8 grayscale images.

    This part is implemented by the group project; connected-component labeling,
    Gaussian filtering, distance transform and watershed use library routines.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Otsu expects a 2D uint8 grayscale image")
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    occupied = np.flatnonzero(histogram)
    if len(occupied) == 1:
        return int(occupied[0])
    weight_left = np.cumsum(histogram)
    moment_left = np.cumsum(histogram * np.arange(256))
    weight_right = image.size - weight_left
    valid = (weight_left[:-1] > 0) & (weight_right[:-1] > 0)
    variance = np.full(255, -np.inf)
    # Equivalent to w0*w1*(mu0-mu1)^2, without dividing by total pixels.
    numerator = moment_left[-1] * weight_left[:-1] - moment_left[:-1] * image.size
    variance[valid] = numerator[valid] ** 2 / (weight_left[:-1][valid] * weight_right[:-1][valid])
    return int(np.argmax(variance))


def binary_foreground(image, sigma=0.0, min_area=0):
    if sigma < 0 or min_area < 0:
        raise ValueError("sigma and min_area must be nonnegative")
    # Otsu searches a fixed 256-bin histogram. Smoothing is explicitly quantized
    # back to uint8, retaining the same intensity range as the original input.
    processed = image
    if sigma:
        processed = np.clip(np.rint(gaussian(image, sigma=sigma, preserve_range=True)), 0, 255).astype(np.uint8)
    threshold = otsu_threshold(processed)
    mask = processed > threshold
    if min_area:
        mask = remove_small_objects(mask, min_size=int(min_area), connectivity=2)
    return mask, threshold


def segment_otsu(image, sigma=0.0, min_area=0):
    mask, threshold = binary_foreground(image, sigma, min_area)
    return label(mask, connectivity=2), {"otsu_threshold": threshold}


def segment_watershed(image, sigma=0.0, min_area=0, min_distance=5):
    if min_distance < 1:
        raise ValueError("min_distance must be at least 1")
    mask, threshold = binary_foreground(image, sigma, min_area)
    distance = ndi.distance_transform_edt(mask)
    coordinates = peak_local_max(distance, min_distance=int(min_distance), labels=mask, exclude_border=False)
    markers = np.zeros(image.shape, dtype=np.int32)
    if len(coordinates):
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    # Ensure every disconnected component has a seed, including small border fragments.
    components, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    seeded = set(np.unique(components[markers > 0]))
    marker_id = int(markers.max())
    for component in range(1, count + 1):
        if component not in seeded:
            marker_id += 1
            candidate = np.where(components == component, distance, -1)
            markers.flat[int(np.argmax(candidate))] = marker_id
    result = watershed(-distance, markers=markers, mask=mask, connectivity=np.ones((3, 3)))
    return result, {"otsu_threshold": threshold, "seed_count": int(markers.max())}


class StarDistPredictor:
    def __init__(self, model_dir, seed=20261104):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.keras.utils.set_random_seed(seed)
        from stardist.models import StarDist2D
        self.model = StarDist2D(config=None, name=model_dir.name, basedir=str(model_dir.parent))

    def predict(self, image, probability=0.48, nms=0.30, boundary=0, percentile_low=1.0, percentile_high=99.8):
        from csbdeep.utils import normalize
        normalized = normalize(image, percentile_low, percentile_high)
        labels, details = self.model.predict_instances(
            normalized, prob_thresh=probability, nms_thresh=nms,
            predict_kwargs={"b": int(boundary)},
            nms_kwargs={"b": int(boundary)}, show_tile_progress=False,
        )
        return labels, {"polygon_count": len(details["prob"])}
