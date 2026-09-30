"""Convert per-instance PNG masks into this project's dataset format (spec section 15).

Phase 1 supports one input layout -- PNG masks, one file per chromosome
instance:

    <masks_root>/<image_stem>/<instance_id>.png   (binary mask, same size as the image)

CVAT- and LabelMe-format converters are named in the spec but not built yet;
that is future work, not a silent gap -- see docs/ARCHITECTURE_V2.md.

Run as a script:

    python -m karyo_ai_dnn.datasets.convert_masks_to_dataset \
        --images path/to/images --masks path/to/masks --out path/to/dataset
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def convert(images_dir: str, masks_root: str, out_dir: str) -> Path:
    images_dir, masks_root, out_dir = Path(images_dir), Path(masks_root), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    image_records, annotations = [], []
    ann_id = 1
    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})

    for image_id, image_path in enumerate(image_files, 1):
        with Image.open(image_path) as img:
            width, height = img.size
        shutil.copy(image_path, out_dir / "images" / image_path.name)
        image_records.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        instance_dir = masks_root / image_path.stem
        if not instance_dir.exists():
            continue
        for mask_path in sorted(instance_dir.glob("*.png")):
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

            dest_rel = f"masks/{image_path.stem}_{mask_path.stem}.png"
            shutil.copy(mask_path, out_dir / dest_rel)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "segmentation": {"mask_file": dest_rel},
                    "area": int(mask.sum()),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "images": image_records,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "chromosome"}],
    }
    out_path = out_dir / "annotations.json"
    out_path.write_text(json.dumps(coco, indent=2))
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out_path = convert(args.images, args.masks, args.out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
