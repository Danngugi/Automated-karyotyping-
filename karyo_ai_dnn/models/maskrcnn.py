"""Mask R-CNN (ResNet-50-FPN) builder for chromosome instance segmentation.

Stage A only (spec section 3): this model learns "where are the chromosome
instances", using num_classes=2 (background, chromosome). Chromosome
identity (1-22/X/Y) is a separate later-phase classifier that consumes this
model's crops -- it is not folded into this network's class head.
"""
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_maskrcnn(num_classes: int = 2, pretrained: bool = True) -> torch.nn.Module:
    """Build Mask R-CNN with a replaced box/mask head for num_classes.

    pretrained=True loads COCO-pretrained detection weights AND an
    ImageNet-pretrained ResNet-50 backbone, both requiring network access to
    download.pytorch.org. pretrained=False disables both explicitly --
    torchvision's `weights=None` alone is not enough, since
    maskrcnn_resnet50_fpn still defaults `weights_backbone` to an
    ImageNet-pretrained ResNet-50 and will try to download it regardless.
    This was caught by actually running the offline smoke test, not by
    inspection -- worth remembering if this function is ever "simplified".
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    weights_backbone = "DEFAULT" if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=weights_backbone)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def detect_device(preferred: str = "auto") -> torch.device:
    """Auto-detect CUDA / MPS / CPU (spec section 16)."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
