# Dataset format

**No real data lives in this repository or was used to build Phase 1.** This describes the format the Phase 1 code expects, so real (de-identified, appropriately licensed/consented) data can be dropped in later without code changes.

## Directory layout

```
data/
  raw/
    human/
      giemsa/
      fish/
    mouse/
      giemsa/
      fish/
  annotations/
    human/
      metaphase/
      chromosomes/
      fish/
    mouse/
      metaphase/
      chromosomes/
      fish/
  processed/
  splits/
    train.txt
    val.txt
    test.txt
```

`raw/` and `processed/` are gitignored (see `.gitignore` at the repo root) — never commit patient or animal images to this repository, de-identified or not, unless you've explicitly decided this repo is meant to carry demo/synthetic data only.

## Per-dataset format Mask R-CNN training actually reads

`karyo_ai_dnn.datasets.coco_dataset.CocoChromosomeDataset` expects a dataset **root** (not necessarily this `data/` tree — could be any path you pass to `--dataset`) shaped like:

```
<root>/
  images/
    <file_name>
  annotations.json
```

`annotations.json` is COCO-shaped:

```json
{
  "images": [{"id": 1, "file_name": "sample_0001.png", "width": 900, "height": 600}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": ...,
      "area": 1234,
      "iscrowd": 0
    }
  ],
  "categories": [{"id": 1, "name": "chromosome"}]
}
```

`segmentation` accepts two forms:

- **Polygon** (standard COCO, what CVAT/LabelMe typically export): `[[x1, y1, x2, y2, ...]]`
- **Mask file reference** (what our own tools produce): `{"mask_file": "masks/sample_0001_0.png"}`, a path relative to `<root>` pointing at a single-channel binary PNG the same size as the image.

Category id `1` is `chromosome` for Phase 1 (Stage A: segmentation only, no identity). Additional categories (`chromosome_fragment`, `debris`, `touching_cluster`) are named in the spec for later phases but not used by any code yet.

## Getting a dataset into this format

- **No real annotations yet, just want to test the pipeline:** `karyo_ai_dnn.datasets.synthetic.generate_synthetic_metaphase(...)` writes a ready-to-use synthetic dataset in this exact format.
- **You have per-instance PNG masks** (one binary mask file per chromosome, organized as `<masks_root>/<image_stem>/<instance_id>.png`): run `python -m karyo_ai_dnn.datasets.convert_masks_to_dataset --images <dir> --masks <masks_root> --out <dataset_root>`.
- **You have CVAT or LabelMe exports:** not supported yet — this is named explicitly in the spec (section 15) as future work, not a silent gap.

## Splitting

`splits/train.txt` / `val.txt` / `test.txt` don't exist yet because there's no real dataset to split. When one exists: **split by image/metaphase/biological sample, never by individual chromosome** — chromosomes from the same metaphase (or the same patient/animal/cell line/slide) must stay in the same split, or evaluation numbers will be inflated by leakage. Suggested default 70/15/15, configurable.
