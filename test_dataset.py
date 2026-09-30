import tempfile
import unittest
from pathlib import Path

from karyo_ai_dnn.datasets.coco_dataset import CocoChromosomeDataset
from karyo_ai_dnn.datasets.synthetic import generate_synthetic_metaphase
from karyo_ai_dnn.species import get_species


class TestSyntheticDataset(unittest.TestCase):
    def _build(self, species, n_chromosomes=None, seed=0, image_size=(360, 240)):
        tmp = tempfile.mkdtemp()
        generate_synthetic_metaphase(
            tmp, species=species, n_chromosomes=n_chromosomes, seed=seed, image_size=image_size
        )
        return CocoChromosomeDataset(tmp)

    def test_human_full_count(self):
        dataset = self._build("human")
        _, target = dataset[0]
        self.assertEqual(target["boxes"].shape[0], get_species("human").chromosome_count)
        self.assertTrue((target["labels"] == 1).all())

    def test_mouse_full_count(self):
        dataset = self._build("mouse")
        _, target = dataset[0]
        self.assertEqual(target["boxes"].shape[0], get_species("mouse").chromosome_count)

    def test_missing_chromosome(self):
        dataset = self._build("human", n_chromosomes=45)
        _, target = dataset[0]
        self.assertEqual(target["boxes"].shape[0], 45)

    def test_extra_chromosome(self):
        dataset = self._build("human", n_chromosomes=47)
        _, target = dataset[0]
        self.assertEqual(target["boxes"].shape[0], 47)

    def test_mask_and_image_shapes_match(self):
        dataset = self._build("human", n_chromosomes=6, image_size=(200, 150))
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape[-2:]), (150, 200))
        self.assertEqual(target["masks"].shape[-2:], (150, 200))

    def test_boxes_within_image_bounds(self):
        dataset = self._build("human", n_chromosomes=10, image_size=(300, 200))
        _, target = dataset[0]
        boxes = target["boxes"]
        self.assertTrue((boxes[:, 0] >= 0).all() and (boxes[:, 2] <= 300).all())
        self.assertTrue((boxes[:, 1] >= 0).all() and (boxes[:, 3] <= 200).all())


if __name__ == "__main__":
    unittest.main()
