#!/usr/bin/env python3
"""Mask R-CNN inference entry point -- Phase 1 scaffold.

Runs a trained checkpoint over one image and writes raw Stage A detections
(boxes/masks/scores; chromosome-vs-background only, no chromosome identity
yet) as JSON. See docs/ARCHITECTURE_V2.md for how this feeds the later
identity-classification and karyotype-assembly stages.
"""
import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from karyo_ai_dnn.models.maskrcnn import build_maskrcnn, detect_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def main():
    args = parse_args()
    device = detect_device(args.device)

    model = build_maskrcnn(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    image = Image.open(args.image).convert("RGB")
    tensor = TF.to_tensor(image).to(device)

    with torch.no_grad():
        [output] = model([tensor])

    keep = output["scores"] >= args.score_threshold
    detections = []
    for box, score, mask in zip(output["boxes"][keep], output["scores"][keep], output["masks"][keep]):
        detections.append(
            {
                "bbox": [float(v) for v in box.tolist()],
                "score": float(score),
                "mask_area": int((mask[0] > 0.5).sum().item()),
            }
        )

    result = {"source": str(args.image), "num_detections": len(detections), "detections": detections}
    out_path = Path(args.out) if args.out else Path(args.image).with_suffix(".maskrcnn.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(detections)} detections to {out_path}")


if __name__ == "__main__":
    main()
