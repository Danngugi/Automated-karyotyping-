#!/usr/bin/env python3
"""Mask R-CNN training entry point (spec section 16) -- Phase 1 scaffold.

PHASE 1 STATUS: this script runs correctly end-to-end against the synthetic
dataset in karyo_ai_dnn/datasets/synthetic.py (see tests/test_maskrcnn_smoke.py)
as a structural check. It has not been trained on any real chromosome data --
no annotated dataset exists yet (that is Phase 2). Do not treat a checkpoint
produced by this script as a usable chromosome detector.

Reads defaults from train_config.json when --config is given (species,
segmentation.epochs, segmentation.batch_size), so the existing hand-off
config from the classical-baseline package keeps meaning something. Explicit
CLI flags override the config file.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from karyo_ai_dnn.datasets.coco_dataset import CocoChromosomeDataset
from karyo_ai_dnn.models.maskrcnn import build_maskrcnn, detect_device
from karyo_ai_dnn.species import get_species


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args():
    p = argparse.ArgumentParser(description="Train Mask R-CNN for chromosome instance segmentation.")
    p.add_argument("--dataset", required=True, help="Path to a dataset root (images/ + annotations.json)")
    p.add_argument("--species", default=None, choices=["human", "mouse"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--output", default="runs/maskrcnn")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--config", default=None, help="Optional train_config.json to source defaults from")
    p.add_argument("--pretrained", action="store_true", default=False,
                    help="Load COCO-pretrained backbone weights (needs network access to download.pytorch.org)")
    return p.parse_args()


def apply_config_defaults(args):
    if not args.config:
        return args
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.species is None:
        args.species = config.get("species", "human")
    seg_cfg = config.get("segmentation", {})
    if args.epochs is None:
        args.epochs = seg_cfg.get("epochs", 1)
    if args.batch_size is None:
        args.batch_size = seg_cfg.get("batch_size", 2)
    return args


def main():
    args = parse_args()
    args = apply_config_defaults(args)
    args.species = args.species or "human"
    args.epochs = args.epochs or 1
    args.batch_size = args.batch_size or 2

    get_species(args.species)  # validates the name; raises on typo
    device = detect_device(args.device)

    dataset = CocoChromosomeDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Stage A only: background + chromosome (spec section 3). Identity
    # classification is a separate later-phase model, not this one.
    model = build_maskrcnn(num_classes=2, pretrained=args.pretrained).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss={epoch_loss / max(1, len(loader)):.4f}")

    checkpoint_path = Path(args.output) / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
