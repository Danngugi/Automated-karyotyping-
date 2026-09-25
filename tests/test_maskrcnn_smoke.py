"""Structural smoke tests for the Mask R-CNN scaffold.

These prove the model builds and a train/inference step runs without
crashing, on CPU, using tiny synthetic images. They do NOT test detection
quality -- there is no real dataset yet, and one CPU step does not train a
usable model. See train_maskrcnn.py's module docstring for the same caveat.
"""
import tempfile
import unittest

import torch

from karyo_ai_dnn.datasets.coco_dataset import CocoChromosomeDataset
from karyo_ai_dnn.datasets.synthetic import generate_synthetic_metaphase
from karyo_ai_dnn.models.maskrcnn import build_maskrcnn, detect_device


def _collate(batch):
    return tuple(zip(*batch))


class TestMaskRCNNSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        generate_synthetic_metaphase(
            self.tmp, species="human", n_chromosomes=5, seed=1, image_size=(256, 192)
        )
        self.dataset = CocoChromosomeDataset(self.tmp)

    def test_model_builds_offline(self):
        model = build_maskrcnn(num_classes=2, pretrained=False)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_device_detection_falls_back_to_cpu_here(self):
        # This sandbox has no CUDA/MPS -- confirms the fallback path works,
        # not that GPU detection works (untestable without GPU hardware).
        self.assertEqual(detect_device("auto").type, "cpu")

    def test_one_training_step_runs(self):
        device = torch.device("cpu")
        model = build_maskrcnn(num_classes=2, pretrained=False).to(device)
        model.train()
        images, targets = _collate([self.dataset[0]])
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_inference_step_runs(self):
        model = build_maskrcnn(num_classes=2, pretrained=False)
        model.eval()
        image, _ = self.dataset[0]
        with torch.no_grad():
            [output] = model([image])
        for key in ("boxes", "labels", "scores", "masks"):
            self.assertIn(key, output)


if __name__ == "__main__":
    unittest.main()
