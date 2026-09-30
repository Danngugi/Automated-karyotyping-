"""Synthetic metaphase generator (spec sections 10, 30).

Produces a COCO-format dataset (images/ + annotations.json) of elongated
blob "chromosomes" on a blank canvas, purely so the Mask R-CNN dataset
loader, training loop, and inference path can be exercised end-to-end
without any real -- and possibly private -- patient or animal data.

This is a structural smoke-test fixture, not a substitute for real training
data: the blobs carry no genuine chromosome morphology or banding pattern.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..species import DNNSpeciesConfig, get_species


def _ellipse_polygon(cx, cy, rx, ry, angle_deg, n_points=24):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    ex, ey = rx * np.cos(theta), ry * np.sin(theta)
    angle = np.deg2rad(angle_deg)
    xs = cx + ex * np.cos(angle) - ey * np.sin(angle)
    ys = cy + ex * np.sin(angle) + ey * np.cos(angle)
    return xs, ys


def generate_synthetic_metaphase(
    out_dir: str,
    species: str = "human",
    n_chromosomes: int = None,
    seed: int = 0,
    image_size=(720, 480),
    image_name: str = "synthetic_0001.png",
):
    """Write one synthetic metaphase image + COCO annotations.json to out_dir.

    n_chromosomes defaults to the species' full diploid count (46 for human,
    40 for mouse). Pass a different count to simulate missing/extra-
    chromosome cases (spec section 30's test list).
    """
    species_cfg: DNNSpeciesConfig = get_species(species) if isinstance(species, str) else species
    count = n_chromosomes if n_chromosomes is not None else species_cfg.chromosome_count

    rng = np.random.default_rng(seed)
    width, height = image_size
    canvas = np.zeros((height, width), dtype=np.uint8)

    annotations = []
    margin = 40
    for ann_id in range(1, count + 1):
        for _attempt in range(20):
            cx = rng.uniform(margin, width - margin)
            cy = rng.uniform(margin, height - margin)
            rx = rng.uniform(8, 14)
            ry = rng.uniform(28, 55)
            angle = rng.uniform(0, 180)
            xs, ys = _ellipse_polygon(cx, cy, rx, ry, angle)
            if xs.min() < 0 or xs.max() >= width or ys.min() < 0 or ys.max() >= height:
                continue
            break

        mask_img = Image.new("L", (width, height), 0)
        from PIL import ImageDraw

        ImageDraw.Draw(mask_img).polygon(list(zip(xs, ys)), outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8)
        canvas = np.maximum(canvas, mask * 200)

        ys_idx, xs_idx = np.where(mask)
        x0, y0, x1, y1 = xs_idx.min(), ys_idx.min(), xs_idx.max() + 1, ys_idx.max() + 1
        polygon_flat = [coord for point in zip(xs.tolist(), ys.tolist()) for coord in point]

        annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": 1,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "segmentation": [polygon_flat],
                "area": float(mask.sum()),
                "iscrowd": 0,
            }
        )

    out_root = Path(out_dir)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    Image.fromarray(255 - canvas).convert("RGB").save(out_root / "images" / image_name)

    coco = {
        "images": [{"id": 1, "file_name": image_name, "width": width, "height": height}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "chromosome"}],
    }
    (out_root / "annotations.json").write_text(json.dumps(coco, indent=2))
    return out_root
