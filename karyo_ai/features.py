import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from .domain import ChromosomeObject


def _medial_axis_length(mask: np.ndarray) -> float:
    """Chromosome length along the curved medial axis, not the bounding box.

    Chromosomes bend at the centromere and during preparation; a straight
    bbox height/width or diagonal systematically underestimates true length
    for anything that isn't perfectly straight, and gets worse the more a
    chromosome is bent. Skeletonizes the mask, builds the 8-connected
    adjacency graph between skeleton pixels, and sums every edge once. For
    a branching skeleton (a chromosome's X/V shape branches at the
    centromere) this sums the total extent of all arms combined, which is
    a reasonable "how much chromosome is here" proxy -- it is NOT the same
    as the ISCN p-arm + q-arm definition (the single longest tip-to-tip
    path), so treat it as an improved proxy, not a clinical-grade
    measurement.
    """
    skeleton = skeletonize(mask)
    ys, xs = np.where(skeleton)
    if len(ys) < 2:
        return float(max(mask.shape))
    coords = np.stack([ys, xs], axis=1).astype(float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    connected = (dist <= np.sqrt(2) + 1e-6) & (dist > 0)
    # Every true edge appears twice in this symmetric matrix -- (i,j) and
    # (j,i) -- so summing the full matrix and halving counts each edge once.
    return float(dist[connected].sum() / 2)


def _centromeric_index(mask: np.ndarray):
    """Estimate centromeric index (CI = short-arm / total length) geometrically.

    Projects every foreground pixel onto the mask's major axis (via PCA),
    then uses the distance transform to find where the chromosome is
    narrowest along that axis -- the centromere is a physical constriction
    ("waist"), so this looks for an actual geometric pinch point rather
    than reusing pixel darkness as a proxy (Giemsa banding means the
    darkest band is not reliably the centromere; the previous version of
    this function conflated the two).

    Returns (ci, group) where group is "metacentric" / "submetacentric" /
    "acrocentric" per the standard CI cutoffs, or (None, None) if the mask
    is too small/thin for a stable estimate. This remains a heuristic: on
    a low-resolution or poorly-banded image the true waist can be masked
    by noise, and CI near a cutoff boundary is inherently uncertain.
    """
    ys, xs = np.where(mask)
    if len(ys) < 12:
        return None, None
    coords = np.stack([xs, ys], axis=1).astype(float)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    projection = centered @ major

    distance = ndimage.distance_transform_edt(mask)
    width_at_pixel = distance[ys, xs] * 2

    order = np.argsort(projection)
    proj_sorted = projection[order]
    width_sorted = width_at_pixel[order]
    span = proj_sorted[-1] - proj_sorted[0]
    if span < 6:  # too short/round along its own major axis to have a meaningful arm split
        return None, None

    k = max(3, len(width_sorted) // 15)
    kernel = np.ones(k) / k
    smoothed = np.convolve(width_sorted, kernel, mode="same")
    # Ignore the outer 10% at each end: tapering chromosome tips are
    # naturally narrow and would otherwise masquerade as the waist.
    lo, hi = int(len(smoothed) * 0.1), int(len(smoothed) * 0.9)
    if hi - lo < 3:
        return None, None
    waist_idx = lo + int(np.argmin(smoothed[lo:hi]))

    frac = (proj_sorted[waist_idx] - proj_sorted[0]) / span
    ci = round(min(frac, 1 - frac), 3)
    if ci >= 0.4:
        group = "metacentric"
    elif ci >= 0.25:
        group = "submetacentric"
    else:
        group = "acrocentric"
    return ci, group


def add_features(objects: list[ChromosomeObject], gray: np.ndarray) -> None:
    for obj in objects:
        x0, y0, x1, y1 = obj.bbox
        crop = gray[y0:y1, x0:x1]
        h, w = crop.shape
        long, short = max(h, w), max(1, min(h, w))
        profile = (crop.mean(axis=1) if h >= w else crop.mean(axis=0)).tolist()

        length_px = long
        ci, group = None, None
        if obj.mask is not None and obj.mask.shape == crop.shape and obj.mask.sum() >= 12:
            length_px = round(_medial_axis_length(obj.mask), 1)
            ci, group = _centromeric_index(obj.mask)

        obj.features = {
            "length_px": length_px,
            "width_px": int(short),
            "aspect_ratio": round(length_px / short, 2) if short else 0.0,
            # Kept for backward compatibility with existing reports/consumers;
            # superseded by centromeric_index/morphology_group when a mask is
            # available (see docstrings above for why the old proxy was weak).
            "centromere_proxy": round(float(np.argmin(profile) / max(1, len(profile) - 1)), 2) if profile else 0.5,
            "centromeric_index": ci,
            "morphology_group": group,
            "band_profile": [round(float(v), 1) for v in profile[:: max(1, len(profile) // 32)]],
        }
