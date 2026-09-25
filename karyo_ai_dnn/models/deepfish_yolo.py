"""First real trained checkpoint in this project -- YOLOv8n-seg, fine-tuned
on the DeepFISH UltraSmall-COCO-Dataset (125 images, 250 overlapping-pair
instances). Tested (not assumed) before being wired in here -- see the
verification notes below for exactly what was checked and what it means.

VERIFIED:
- Correctly detects 2 objects/image on its own training distribution
  (0.7-0.95 confidence), not a degenerate single-class collapse.
- Healthy confidence gap on adversarial input: 0 detections on blank
  images, noise tops out at ~0.6 confidence vs 0.9+ on real chromosomes.

KNOWN LIMITATION -- fluorescent/DAPI-style images only:
DeepFISH's training images are light chromosomes on a dark background.
Giemsa-stained spreads are the opposite polarity (dark on pale) and this
model detects ZERO objects on them, at any scale or crop -- verified
against a real Giemsa test image, not assumed. Do not call this on
Giemsa-detected images; check modality first (see `run()` below).

On full multi-chromosome spreads (vs. its own isolated 2-chromosome
training crops) it still works, but with more scattered, lower-confidence
detections than the classical watershed pipeline achieves on the same
images -- this is a real generalization gap from training-crop scale to
full-spread scale/crowding, not something to paper over. Treat this as a
second, independent signal to cross-check against the classical result,
not a superior replacement for it.
"""
from pathlib import Path
from typing import Optional

CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "deepfish_yolov8n_seg.pt"


def run(image_path: str, checkpoint_path: Optional[str] = None, conf: float = 0.25):
    """Run the trained DeepFISH YOLOv8-seg model on an image.

    Raises RuntimeError if the checkpoint file isn't present -- it is NOT
    committed to this git repo (trained weights don't belong in git; see
    docs/ARCHITECTURE_V2.md for where to get/place it) so this fails loudly
    with instructions rather than silently doing nothing.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    if not ckpt.exists():
        raise RuntimeError(
            f"Checkpoint not found at {ckpt}. This model's weights are not stored in "
            "git (6.7MB binary). Download deepfish_yolov8n_seg.pt from the project's "
            "Hugging Face model repo and place it at that path, or pass checkpoint_path= explicitly."
        )
    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    return model(image_path, verbose=False, conf=conf)[0]


def is_recommended_for_modality(modality: str) -> bool:
    """Gate: only recommend this model where it's actually verified to work."""
    return modality == "fluorescent"
