---
license: other
license_name: dan36karyo
license_link: LICENSE
---

# Model Card for KaryoAI

Chromosome image-analysis pipeline: a working classical (non-neural) segmentation baseline, plus a deep-learning track with **one real trained checkpoint** (fluorescent images only) and one still-untrained scaffold (Mask R-CNN). Not a diagnostic tool.

## Extended Description

Karyotyping — arranging an individual's chromosomes into a standardized layout for visual review — is normally done by a trained cytogeneticist working from a metaphase spread image. KaryoAI's goal is to give that reviewer a faster starting point: an automated first-pass segmentation and draft layout they correct, not a replacement for their judgment.

The project separates two concerns that are easy to conflate: **finding** chromosome objects in an image, and **identifying** which chromosome each one is. Three things exist today, at different levels of maturity:

- **The classical track** (`karyo_ai/`) is working software. It thresholds the image, separates touching/overlapping chromosomes via watershed splitting, pairs objects into homologous slots (1:1, 1:2, 2:1, 2:2 ... through the sex pair), and flags whole spreads as good/review/poor based on how much area remains unresolved. It also filters out non-chromosome objects — screenshot borders, burnt-in text, hand-drawn annotation marks, cell nuclei, background debris — using shape and (on colour-stained images) stain-saturation checks, so they're excluded from pairing and visibly flagged rather than silently corrupting a real chromosome count. It still doesn't claim to know that a given blob is "chromosome 7" — the number is a size-rank slot for a human to relabel, not an identification.
- **The DeepFISH-trained track** (`karyo_ai_dnn/checkpoints/deepfish_yolov8n_seg.pt`) is a YOLOv8n-seg model fine-tuned on 125 real fluorescent chromosome images (250 overlapping-pair instances). This is a real, verified trained model — not a scaffold. See Training Details and Evaluation below for exactly what it can and can't do; the short version is it works well on fluorescent/DAPI images and fails completely on Giemsa-stained ones.
- **The Mask R-CNN scaffold** (`karyo_ai_dnn/models/maskrcnn.py`) is still untrained. Code builds, trains a step, and runs inference correctly on synthetic data only — no real chromosome image has been used to train it.

Both the classical track and the Mask R-CNN scaffold support human and mouse chromosome counts (46 and 40). The DeepFISH-trained model is single-class (chromosome vs. background) and species-agnostic by construction. Every output across all tracks is framed as a draft for expert correction, not a result to act on.

## Model Details

### Model Description

Three components:

1. **Classical pipeline** — thresholding, watershed-based touching-chromosome separation, homologous pairing, artifact/debris/annotation filtering, whole-spread quality gating, draft karyogram and report generation. Rule-based throughout; nothing here is "trained" in the ML sense.
2. **DeepFISH YOLOv8n-seg checkpoint** — real trained weights (6.7MB), verified against both its own training distribution and out-of-distribution images (see Evaluation). Single class: `chromosome`.
3. **Untrained Mask R-CNN scaffold** — PyTorch/torchvision, ResNet-50-FPN backbone, structurally verified on synthetic data only.

- **Developed by:** Dan Ngugi
- **Funded by:** Independent research project; not currently externally funded
- **Shared by:** Dan Ngugi
- **Model type:** A rule-based image-processing pipeline (classical track), plus one trained YOLOv8n-seg instance-segmentation model (fluorescent chromosome detection), plus one untrained Mask R-CNN scaffold
- **Language(s):** Not applicable (computer vision, not NLP)
- **License:** Other — `dan36karyo` (custom; **LICENSE file in this repo is currently empty**, terms not yet written)
- **Finetuned from model:** `yolov8n-seg.pt` (Ultralytics COCO-pretrained) → fine-tuned on the DeepFISH UltraSmall-COCO-Dataset. The Mask R-CNN scaffold remains un-fine-tuned.

### Model Sources

- **Repository:** [github.com/Danngugi/Karyotype-ai-pipeline](https://github.com/Danngugi/Karyotype-ai-pipeline)
- **Training data source:** [github.com/Deep4FISH/DeepFISH](https://github.com/Deep4FISH/DeepFISH) (GPL-3.0)
- **Paper:** None
- **Demo:** None currently live

## Uses

### Direct Use

Classical pipeline: draft segmentation, pairing, and report from a Giemsa or fluorescent metaphase image, for expert review. DeepFISH model: chromosome instance detection/segmentation on **fluorescent/DAPI images specifically** — see Bias/Risks/Limitations before using it on anything else.

### Downstream Use

First-pass input to an expert-reviewed cytogenetics workflow; corrected outputs could become training data for further model development.

### Out-of-Scope Use

Clinical diagnosis or any patient-care decision. The DeepFISH model on Giemsa-stained images (verified non-functional, see below). Any use presenting a draft as a confirmed result without expert review.

## Bias, Risks, and Limitations

- Classical pipeline's chromosome-numbering remains a size-rank heuristic, not identification.
- **The DeepFISH model detects zero objects on Giemsa-stained images, at any scale or crop position** — verified directly, not assumed. Root cause: it trains exclusively on light-chromosomes-on-dark-background images; Giemsa is the opposite polarity. Use `karyo_ai_dnn.models.deepfish_yolo.is_recommended_for_modality()` to gate this automatically.
- On full multi-chromosome spreads (as opposed to its own isolated 2-chromosome training crops), the DeepFISH model still works but with more scattered, lower-confidence detections than on its training distribution — a real train/deploy scale gap.
- No performance characterization exists for any component against expert-verified ground truth — only against the model's own training/validation split (DeepFISH) or synthetic data (Mask R-CNN, classical pipeline's test suite).
- Risk of over-trust: a real trained model existing now makes it easier to mistake this for validated diagnostic support. It is not.

### Recommendations

Always have a qualified cytogeneticist review output before it informs any decision. Do not run the DeepFISH model on Giemsa images. Validate against your own site's images and staining protocol before relying on this beyond exploratory research.

## How to Get Started with the Model

```bash
git clone https://github.com/Danngugi/Karyotype-ai-pipeline.git
cd Karyotype-ai-pipeline

# Classical baseline (works today, any modality):
pip install -r requirements-app.txt
python3 karyo_ai_cli.py path/to/image.png --out-dir runs/case_001 --species human

# DeepFISH-trained detector (fluorescent images only):
pip install -r requirements-dnn.txt
python3 -c "
from karyo_ai_dnn.models.deepfish_yolo import run, is_recommended_for_modality
from karyo_ai import preprocessing
img = preprocessing.load_image('path/to/image.png')
modality = preprocessing.detect_modality(img)
if is_recommended_for_modality(modality):
    result = run('path/to/image.png')
    print(f'{len(result.boxes)} chromosomes detected')
else:
    print(f'Not recommended for {modality} images')
"

# Mask R-CNN scaffold (still untrained -- structural smoke test only):
python3 -m unittest discover tests
```

## Training Details

### Training Data

**DeepFISH YOLOv8n-seg:** 125 real fluorescent chromosome images, 250 overlapping-pair instances, COCO polygon annotations (Deep4FISH/DeepFISH, GPL-3.0). 80/20 train/val split by image.

**Mask R-CNN scaffold:** None real. Synthetic generator only (`karyo_ai_dnn/datasets/synthetic.py`).

**Classical pipeline:** Not applicable — rule-based, not trained.

### Training Procedure

**DeepFISH YOLOv8n-seg:** Full training run completed on a Colab GPU. `yolov8n-seg.pt` base, 100 epochs, batch 16, image size 224px, standard Ultralytics training loop.

**Mask R-CNN scaffold:** One structural smoke-test step on CPU with synthetic data — not a real training run.

#### Training Hyperparameters

DeepFISH model: epochs=100, batch=16, imgsz=224, base model `yolov8n-seg.pt`, default Ultralytics optimizer settings.

#### Speeds, Sizes, Times

DeepFISH model: 6.7MB checkpoint. Training wall-clock time not recorded. Mask R-CNN's one smoke-test step: seconds, on 1 CPU core — not representative of real training.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

**DeepFISH model:** its own held-out validation split (20% of the 125-image set) for the metrics below. Separately, verified qualitatively against: (a) real project reference images of both modalities, (b) synthetic noise/blank/white adversarial images, (c) scale-matched crops from full spreads.

#### Metrics

mAP50, mAP50-95, precision, recall (box and mask), reported separately below.

### Results

**DeepFISH YOLOv8n-seg, on its own validation split:**

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.980 | 0.905 |
| Recall | 0.994 | 0.958 |
| mAP50 | 0.994 | 0.942 |
| mAP50-95 | 0.798 | 0.690 |

**Qualitative verification beyond the validation split** (this is the part a raw metrics table doesn't show, and why it's included here rather than left implicit):
- Correctly detects 2 objects/image on held-out training-distribution images, 0.7–0.95 confidence
- Noise/blank adversarial images: 0 detections on blank, noise tops out ~0.6 confidence vs. 0.9+ on real chromosomes — a healthy, non-degenerate confidence gap
- **0 detections on Giemsa images, at any scale** — a real, verified limitation, not a gap in testing
- Reduced confidence and more scattered detections on full multi-chromosome spreads vs. its isolated-pair training crops

**Mask R-CNN scaffold:** no results to report — untrained.

**Classical pipeline:** no formal metrics (rule-based); qualitatively verified end-to-end against real Giemsa and fluorescent project images, including edge cases (zero-detection blank images, severely over-segmented spreads correctly caught by the quality gate).

#### Summary

One component of this project now has real, verified performance data (the DeepFISH model), scoped honestly to what it was actually tested on and where it's known to fail. The rest remains either rule-based (no training claim to make) or untrained (Mask R-CNN). This section should be updated again once the classical pipeline or Mask R-CNN track gets any real-data evaluation.

## Technical Specifications

### Model Architecture and Objective

Classical baseline: thresholding + watershed instance separation, heuristic pairing. DeepFISH model: YOLOv8n-seg (single-stage instance segmentation), class `chromosome` only, Stage A (segmentation, not identity). Mask R-CNN scaffold: ResNet-50-FPN backbone, Stage A, untrained.

### Compute Infrastructure

DeepFISH model trained on a Colab GPU. Everything else (classical pipeline, Mask R-CNN scaffold, all verification/testing) run on a 1-CPU-core, no-GPU environment.

#### Software

Python 3.12, PyTorch, torchvision, Ultralytics (YOLOv8), NumPy, SciPy, scikit-image, Pillow, Streamlit.

## Citation

```bibtex
@software{karyoai,
  author = {Ngugi, Dan},
  title = {KaryoAI: A research-assistance pipeline for chromosome image analysis},
  url = {https://github.com/Danngugi/Karyotype-ai-pipeline},
  year = {2026}
}
```

Training data: Deep4FISH/DeepFISH (GPL-3.0), https://github.com/Deep4FISH/DeepFISH

## Model Card Authors

Dan Ngugi

## Model Card Contact

Via Hugging Face: [@ngugdan](https://huggingface.co/ngugdan), or GitHub issues on [Danngugi/Karyotype-ai-pipeline](https://github.com/Danngugi/Karyotype-ai-pipeline)ugi/Karyotype-ai-pipeline)
