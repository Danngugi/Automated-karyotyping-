"""COCO-style instance-segmentation dataset loader (spec section 15).

On-disk layout (see ../../data/README.md for the full spec):

    <root>/images/<file_name>
    <root>/annotations.json

annotations.json follows the standard COCO shape (images / annotations /
categories), with one deviation documented here rather than left implicit:
`segmentation` may be either a standard COCO polygon list, or an object
``{"mask_file": "<path relative to root>"}`` pointing at a single-channel
PNG mask. The polygon form is what annotation tools (CVAT, LabelMe) export;
the mask_file form is what our own synthetic generator and PNG-mask
converter (convert_masks_to_dataset.py) produce. Supporting both avoids
forcing every producer through lossy polygon tracing.

Category id 1 is "chromosome" for Stage A (segmentation only -- this loader
does not carry chromosome identity labels; see spec section 3-4 and
docs/ARCHITECTURE_V2.md for why identity is a separate later stage).
"""
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw


class CocoChromosomeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "annotations.json",
        transforms: Optional[Callable] = None,
    ):
        self.root = Path(root)
        with open(self.root / annotations_file) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = sorted(self.images)
        self.annotations_by_image: dict = {}
        for ann in coco["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        info = self.images[image_id]
        image = Image.open(self.root / "images" / info["file_name"]).convert("RGB")
        width, height = info["width"], info["height"]

        anns = self.annotations_by_image.get(image_id, [])
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            masks.append(self._load_mask(ann["segmentation"], width, height))
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(
                np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)
            ),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)
            return image, target
        return TF.to_tensor(image), target

    def _load_mask(self, segmentation, width: int, height: int) -> np.ndarray:
        if isinstance(segmentation, dict) and "mask_file" in segmentation:
            mask = Image.open(self.root / segmentation["mask_file"]).convert("L")
            return (np.array(mask) > 0).astype(np.uint8)

        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
        for poly in polygons:
            points = list(zip(poly[0::2], poly[1::2]))
            draw.polygon(points, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)
