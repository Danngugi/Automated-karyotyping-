# KaryoAI pipeline guide

## Scope and safety

This is a research/workflow-assistance prototype. A certified cytogeneticist must independently review every result. Before use in a laboratory SOP or patient-care workflow, perform formal analytical validation for accuracy, precision, and reproducibility under the laboratory quality system (for example ISO 15189 requirements). The application and report intentionally label all results as draft until human sign-off.

## Stages

1. Validate image and assess focus/contrast.
2. Route to Giemsa or fluorescent preprocessing (automatic heuristic or manual override).
3. Locate chromosome-like foreground and segment components. The included baseline uses thresholding; touching objects are flagged.
4. Extract length, aspect ratio, centromere proxy, and a simple intensity profile.
5. Assign labels from supplied annotations, or use an explicitly low-confidence morphology heuristic.
6. Pair putative homologs, assemble a draft karyogram, and flag count mismatches for review.
7. Export annotated PNG, draft layout, JSON, and Markdown report. A reviewer must approve before a record can be marked reviewed.

## Species

Human: 22 autosome pairs plus XX or XY (typically 46 chromosomes). Mouse: 19 autosome pairs plus sex chromosomes (typically 40). Mouse SKY/M-FISH is treated as a colour-signature workflow; human Denver-group morphology rules are not applied to mouse.

## Dataset layout

Point `dataset_connector_cli.py` at a local folder. Class-labelled images can be arranged as `root/1/*.png`, `root/X/*.png`; COCO JSON files are detected and recorded. Expected future training data include image masks/bounding boxes, chromosome labels, stain modality, and reviewer corrections.

Useful source references to evaluate under their own terms: BioImLab (Padova), UT Austin LIVE/ADIR M-FISH, AutoKary2022, and relevant Kaggle chromosome and overlapping-chromosome datasets. Do not treat a dataset listing as clinical validation.
