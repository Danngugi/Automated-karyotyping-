---
license: other
license_name: dan36karyo
license_link: LICENSE
---

# Model Card for KaryoAI (DN_CYTOGENETICS_KARYOTYPE)

A research-only chromosome image-analysis pipeline: a working classical (non-neural) segmentation baseline, plus an in-progress, **untrained** Mask R-CNN instance-segmentation scaffold. Not a diagnostic tool.

> **Repository status note:** the top-level scripts and docs are uploaded, but the `karyo_ai/`, `karyo_ai_dnn/`, `docs/`, `data/`, and `tests/` folders are not yet present in this repo — none of the scripts here will run until those are added (drag the actual folders in via "Add file → Upload files", or use `hf upload <repo> <local-dir> .` to push the whole tree at once).

## Extended Description

Karyotyping — arranging an individual's chromosomes into a standardized layout for visual review — is normally done by a trained cytogeneticist working from a metaphase spread image captured under a microscope. It's slow, skill-intensive, and the same manual segmentation-and-arrangement work is repeated for every sample. KaryoAI's goal is to give that reviewer a faster starting point: an automated first-pass segmentation and draft layout they correct, rather than build from scratch — not to replace their judgment.

The project deliberately separates two concerns that are easy to conflate: **finding** chromosome objects in an image, and **identifying** which chromosome each one is. Mixing these into one step makes it hard to tell whether a bad result comes from bad segmentation or bad classification, and makes the whole system harder to validate incrementally. So KaryoAI is built in stages, and ships two tracks at different levels of maturity rather than pretending both are equally ready:

- **The classical track** (`karyo_ai/`) is genuinely working software today. It thresholds the image, finds connected components as candidate chromosome objects, and applies a simple, explicitly low-confidence heuristic (ordering by size) to lay them out into a draft karyogram. It doesn't claim to know that a given blob is "chromosome 7" — it claims to have found 46 (or however many) chromosome-shaped objects and arranged them in a plausible grid for a human to relabel correctly.
- **The deep-learning track** (`karyo_ai_dnn/`) is where chromosome *identification* is meant to eventually live, using a Mask R-CNN model trained specifically to segment chromosome instances (as a first stage, separate from identifying which number they are). As of this upload, that track is real, tested code — it builds, trains a step, and runs inference correctly — but it has never seen a real chromosome image and has no trained weights. It exists so that once real annotated data is available, there's a working pipeline to train it into, rather than starting from nothing.

Both tracks support human and mouse chromosome counts (46 and 40 respectively), reflecting the project's origin in comparative cytogenetics research rather than a single-species clinical tool. Every output — from either track — is explicitly framed as a draft: annotated images, JSON results, and Markdown reports are all designed to be corrected by an expert, not consumed directly.

## Model Details

### Model Description

KaryoAI is a chromosome-spread image-analysis project aimed at assisting (not replacing) expert cytogenetic review. Two things live in this repository:

1. **A working classical pipeline** — thresholding + connected-component segmentation, producing a segmented/annotated image, a draft karyogram layout, a JSON result, and a Markdown report. Chromosome-to-number assignment uses a low-confidence size-ordering heuristic, explicitly not a validated classifier.
2. **An untrained deep-learning scaffold** — a PyTorch/torchvision Mask R-CNN (ResNet-50-FPN backbone), currently trained only on synthetic smoke-test data on CPU, not on any real chromosome image. There is no usable checkpoint from this yet.

- **Developed by:** Dan Ngugi
- **Funded by:** Independent research project; not currently externally funded
- **Shared by:** Dan Ngugi
- **Model type:** Not a single trained model — a classical (rule-based) image-processing pipeline, plus an untrained Mask R-CNN (ResNet-50-FPN) instance-segmentation network intended for chromosome-vs-background segmentation
- **Language(s):** Not applicable (computer vision, not NLP)
- **License:** Other — `dan36karyo` (custom; **LICENSE file in this repo is currently empty**, terms not yet written)
- **Finetuned from model:** None yet. The Mask R-CNN builder can optionally start from a COCO/ImageNet-pretrained `torchvision.models.detection.maskrcnn_resnet50_fpn` backbone once real training begins — no fine-tuning has occurred yet.

### Model Sources

- **Repository (source, full commit history/branches):** [github.com/Danngugi/Karyotype-ai-pipeline](https://github.com/Danngugi/Karyotype-ai-pipeline)
- **Paper:** None
- **Demo:** None currently live (a Hugging Face Space was planned but does not exist on this account yet)

## Uses

### Direct Use

Running the classical pipeline locally or via the (planned) demo to get a draft segmentation, draft karyogram layout, and structured report from a Giemsa-stained metaphase image, for a human expert to review and correct.

### Downstream Use

As the first-pass, machine-assisted step in a larger expert-reviewed cytogenetics workflow — e.g., feeding draft segmentations to a cytogeneticist for correction and annotation, which could in turn become training data for the in-progress Mask R-CNN track.

### Out-of-Scope Use

Clinical diagnosis, patient care decisions, or any use where output is treated as a confirmed karyotype without independent expert review. Not validated for FISH imagery (Giemsa only, currently). Not intended for autonomous/unreviewed use of any kind.

## Bias, Risks, and Limitations

- The draft chromosome-numbering heuristic (size-based ordering) will frequently be wrong on real spreads — it was never designed to be a real classifier, just a placeholder layout aid.
- Classical segmentation degrades on touching/overlapping chromosomes and on debris-heavy or unevenly stained spreads.
- No performance characterization exists on real data of any kind — only synthetic test images have been used so far (see Evaluation, below).
- Risk of over-trust: "AI-assisted" framing could be mistaken for validated diagnostic support if the disclaimers here and in the repository aren't carried through to wherever this is deployed.

### Recommendations

Always have a qualified cytogeneticist review any output before it informs any decision. Validate against your own site's images and staining protocol before relying on this for anything beyond exploratory research. Do not remove the research-only disclaimers when redistributing.

## How to Get Started with the Model

There's no `from_pretrained`-style checkpoint to load — this repo hosts source code, not trained weights (see Model Description, above).

```bash
# Download the repo
hf download ngugdan/DN-Cytogenetic-karyotyping --local-dir ./karyoai
# (or: git clone https://huggingface.co/ngugdan/DN-Cytogenetic-karyotyping karyoai)
cd karyoai

# Classical baseline (works today):
pip install -r requirements-app.txt
python3 karyo_ai_cli.py path/to/image.png --out-dir runs/case_001 --species human

# Mask R-CNN scaffold (untrained — structural smoke test only):
pip install -r requirements-dnn.txt
python3 -m unittest discover tests
```

## Training Details

### Training Data

None yet for the Mask R-CNN track — no real annotated dataset exists. All current testing uses a synthetic generator (`karyo_ai_dnn/datasets/synthetic.py`) that draws elongated blob shapes; it carries no real chromosome morphology. The classical pipeline is rule-based and was never "trained" in the ML sense.

### Training Procedure

Not yet run to completion. `train_maskrcnn.py` implements a standard single-phase fine-tuning loop (SGD, CUDA/MPS/CPU auto-detect) and has been verified to execute one training step correctly on CPU with synthetic data — that is the extent of what has actually happened.

#### Preprocessing

Classical pipeline: contrast/focus QC, grayscale normalization, thresholding. DNN track: standard `torchvision` tensor conversion; no augmentation pipeline implemented yet.

#### Training Hyperparameters

Not applicable yet — no real training run has been performed to report hyperparameters for.

#### Speeds, Sizes, Times

Not applicable — the one verified training step was a single-image, single-epoch structural smoke test on a 1-CPU-core machine, taking on the order of seconds. Not representative of any real training run.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Synthetic-only: generated human (46-chromosome) and mouse (40-chromosome) metaphases, plus missing/extra-chromosome variants, used to smoke-test the dataset loader and model — not to measure detection quality.

#### Factors

Not applicable yet — no real-data evaluation has been run to disaggregate by.

#### Metrics

Not applicable yet. Planned (per project roadmap, not yet implemented): precision/recall/F1/IoU/Dice/AP50/AP75 for segmentation, plus chromosome-count error and classification accuracy once those stages exist.

### Results

None yet — `evaluate_segmentation.py` (the planned evaluation harness) has not been built.

#### Summary

No trained-model performance can be reported honestly at this stage. This section will be updated once real training and evaluation occur.

## Model Examination

Not applicable yet — no trained model to examine for interpretability.

## Environmental Impact

No dedicated training run has occurred. The only compute used so far was a few seconds of CPU-only structural smoke-testing (1 core, no GPU) to verify the code runs — not a training run in any meaningful sense, so a carbon-emission estimate would not be meaningful here. This section will be completed once real training happens.

- **Hardware Type:** None yet (smoke tests only, 1 CPU core, no GPU)
- **Hours used:** Not applicable
- **Cloud Provider:** Not applicable
- **Compute Region:** Not applicable
- **Carbon Emitted:** Not applicable

## Technical Specifications

### Model Architecture and Objective

Classical baseline: image thresholding + 4-connected component labelling (via `scipy.ndimage`), heuristic size-based chromosome ordering. DNN track: Mask R-CNN, ResNet-50-FPN backbone, currently configured for Stage A only — binary instance segmentation (background vs. chromosome). Chromosome identity (1–22/X/Y), homolog pairing, and karyotype assembly are separate, not-yet-built later stages by design.

### Compute Infrastructure

#### Hardware

Developed and smoke-tested on a 1-CPU-core, ~4GB RAM, no-GPU environment. Real training will require a GPU (not available in the environment this was built in).

#### Software

Python 3.12, PyTorch 2.14, torchvision 0.29, NumPy, SciPy, Pillow, Streamlit (for the planned demo app).

## Citation

**BibTeX:**

```bibtex
@software{karyoai,
  author = {Ngugi, Dan},
  title = {KaryoAI: A research-assistance pipeline for chromosome image analysis},
  url = {https://github.com/Danngugi/Karyotype-ai-pipeline},
  year = {2026}
}
```

**APA:**

Ngugi, D. (2026). *KaryoAI: A research-assistance pipeline for chromosome image analysis* [Computer software]. https://github.com/Danngugi/Karyotype-ai-pipeline

## Glossary

- **Karyogram:** a standardized visual arrangement of an individual's chromosomes, ordered by size/type, used for cytogenetic review.
- **Metaphase spread:** a microscopy image of condensed chromosomes captured during cell division, the standard input for karyotyping.
- **Stage A / Stage B (this project's terms):** Stage A = detecting/segmenting chromosome instances only; Stage B = identifying which chromosome number each instance is. Kept as separate models by design so segmentation quality isn't entangled with identity accuracy.
- **Aneuploidy:** an abnormal chromosome count (e.g., trisomy, monosomy) — a planned future analysis stage, not yet implemented.

## More Information

Full architecture status (what's built vs. planned, section by section) is in `docs/ARCHITECTURE_V2.md` inside the bundled source.

## Model Card Authors

Dan Ngugi

## Model Card Contact

Via Hugging Face: [@ngugdan](https://huggingface.co/ngugdan), or GitHub issues on [Danngugi/Karyotype-ai-pipeline](https://github.com/Danngugi/Karyotype-ai-pipeline)
