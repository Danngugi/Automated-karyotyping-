import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from .domain import ChromosomeObject

# 4-connectivity, matching the original baseline's neighbour rule.
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _split_touching(local_mask: np.ndarray, expected_count: int, min_distance: int):
    """Try to split a merged blob into ~expected_count pieces via watershed.

    Distance-transform + watershed is a standard classical technique for
    separating touching/overlapping roughly-convex objects and needs no
    training data. It is NOT a precise fix for chromosomes specifically:
    a single bent/elongated chromosome can have more than one local ridge
    in its own distance transform (e.g. at each arm), which risks
    over-splitting one real chromosome into two. `min_distance` is the
    knob that trades this off -- too small over-splits individual
    chromosomes, too large under-splits genuinely touching ones. Returns
    None if fewer than 2 peaks are found (nothing to split).
    """
    distance = ndimage.distance_transform_edt(local_mask)
    coords = peak_local_max(
        distance, labels=local_mask.astype(int), num_peaks=min(expected_count, 8), min_distance=min_distance
    )
    if len(coords) < 2:
        return None
    markers = np.zeros(local_mask.shape, dtype=int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    return watershed(-distance, markers, mask=local_mask)


def _recursive_split(local_mask: np.ndarray, x_off: int, y_off: int, median: float, min_area: int, min_distance: int, depth: int = 0):
    """Repeatedly attempt watershed splitting until pieces are reasonably
    sized or splitting stops making progress. Capped at depth=3 -- a
    genuinely fused triple/quadruple overlap may never cleanly resolve with
    this heuristic, and infinite recursion on a pathological blob is worse
    than reporting an imperfect result with a note attached.
    """
    area = int(local_mask.sum())
    touching = area > max(1, median) * 2.5
    if not touching or depth >= 3:
        ys, xs = np.where(local_mask)
        bbox = (int(xs.min() + x_off), int(ys.min() + y_off), int(xs.max() + 1 + x_off), int(ys.max() + 1 + y_off))
        centroid = (float(xs.mean() + x_off), float(ys.mean() + y_off))
        notes = "flagged touching; watershed split attempt did not find a clean separation" if touching else ""
        cropped = local_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return [(bbox, area, centroid, touching, notes, cropped)]

    expected_count = max(2, round(area / median))
    split_labels = _split_touching(local_mask, expected_count, min_distance)
    if split_labels is None or split_labels.max() < 2:
        ys, xs = np.where(local_mask)
        bbox = (int(xs.min() + x_off), int(ys.min() + y_off), int(xs.max() + 1 + x_off), int(ys.max() + 1 + y_off))
        centroid = (float(xs.mean() + x_off), float(ys.mean() + y_off))
        cropped = local_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return [(bbox, area, centroid, True, "flagged touching; watershed split attempt did not find a clean separation", cropped)]

    results = []
    for sub_id in range(1, split_labels.max() + 1):
        sub_mask = split_labels == sub_id
        if int(sub_mask.sum()) < min_area:
            continue
        results.extend(_recursive_split(sub_mask, x_off, y_off, median, min_area, min_distance, depth + 1))
    # Tag the leaves of an actual split, but only once (depth 0 callers of a
    # successful split produce >1 result -- a single-result recursion means
    # nothing was actually separated at this level, so leave notes as-is).
    if len(results) > 1:
        results = [
            (bbox, area, centroid, touching, notes or "watershed-split from a touching cluster; verify this boundary", cropped)
            for (bbox, area, centroid, touching, notes, cropped) in results
        ]
    return results


def segment(mask: np.ndarray, min_area: int = 25) -> list[ChromosomeObject]:
    """Label connected foreground regions into draft ChromosomeObject candidates.

    Uses scipy.ndimage's vectorized labelling instead of a pixel-by-pixel
    Python flood fill: on real (larger, noisier) chromosome spread images the
    Python version does not scale, since its cost grows with total foreground
    pixel count rather than component count. Region sizes and centroids are
    computed with np.bincount rather than ndimage.sum/center_of_mass, which
    are much slower once there are more than a few hundred labelled regions.
    Regions smaller than ``min_area`` pixels are dropped as noise, matching
    the original baseline's behaviour.

    Components much larger than the median (the existing "touching" signal)
    get one watershed-based split attempt before being reported. This
    materially helps on real spreads where chromosomes cross/touch (see
    docs/AI_KARYOTYPE_PIPELINE_GUIDE.md's segmentation notes) but is a
    heuristic, not a real instance-segmentation model -- it will sometimes
    over- or under-split. Objects produced this way are flagged in `notes`
    so a reviewer knows to double-check that boundary specifically.
    """
    labelled, count = ndimage.label(mask, structure=_FOUR_CONNECTED)
    if count == 0:
        return []

    flat_labels = labelled.ravel()
    foreground = np.flatnonzero(flat_labels)
    labels_at_foreground = flat_labels[foreground]
    rows, cols = np.unravel_index(foreground, mask.shape)

    sizes = np.bincount(labels_at_foreground, minlength=count + 1)[1:]
    slices = ndimage.find_objects(labelled)

    kept_ids = np.flatnonzero(sizes >= min_area) + 1
    if kept_ids.size == 0:
        return []
    median = float(np.median(sizes[kept_ids - 1]))
    min_distance = max(4, int(np.sqrt(median) / 1.5))

    raw_objects = []  # (bbox, area, centroid, touching, notes, mask)
    for label_id in kept_ids:
        idx = label_id - 1
        y_slice, x_slice = slices[idx]
        local_mask = labelled[y_slice, x_slice] == label_id
        raw_objects.extend(_recursive_split(local_mask, x_slice.start, y_slice.start, median, min_area, min_distance))

    objects = []
    for object_id, (bbox, area, centroid, touching, notes, obj_mask) in enumerate(raw_objects, 1):
        objects.append(ChromosomeObject(object_id, bbox, area, centroid, touching=touching, notes=notes, mask=obj_mask))
    return objects
