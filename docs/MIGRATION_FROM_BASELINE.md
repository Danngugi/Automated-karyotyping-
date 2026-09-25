# Migration from the classical baseline

## Nothing is being replaced yet

`karyo_ai/` (thresholding + connected-component segmentation, heuristic labelling) is untouched by this Phase 1 work. `karyo_ai_cli.py`, `streamlit_app.py`, and `api_app.py` all still run exactly as they did before. This is intentional: the classical pipeline is your only *working, tested* pipeline right now, and it's what your peers are (or will be) testing on Hugging Face — Phase 1 shouldn't put that at risk.

## Where the new code lives

Everything from the DNN upgrade spec lives in a new, separate package and set of scripts so the two tracks can't accidentally interfere:

```
karyo_ai/          <- unchanged classical baseline
karyo_ai_dnn/       <- new Phase 1 scaffold
    species.py
    datasets/
        coco_dataset.py
        synthetic.py
        convert_masks_to_dataset.py
    models/
        maskrcnn.py
train_maskrcnn.py    <- new Mask R-CNN trainer (spec section 16)
predict_maskrcnn.py   <- new Mask R-CNN inference script
train_model.py        <- existing hand-off stub, now just points at train_maskrcnn.py
tests/
    test_dataset.py
    test_maskrcnn_smoke.py
docs/
    ARCHITECTURE_V2.md
    MIGRATION_FROM_BASELINE.md
data/
    README.md
```

`karyo_ai_dnn.species.DNNSpeciesConfig` is a separate class from `karyo_ai.species.SpeciesConfig`, even though they overlap conceptually. This is deliberate for now — the two tracks may need different fields as Phase 1 evolves, and keeping them separate means neither can be broken by changes to the other. If they're still just as similar once identity classification (Phase 4) exists, consolidating them into one shared config would be a reasonable cleanup at that point.

## What "rename the baseline to ClassicalBaseline" (spec section 31) actually needs

The spec asks for the classical algorithm to be renamed `ClassicalBaseline` and compared head-to-head against Mask R-CNN. That rename is deferred, on purpose, to whenever `evaluate_segmentation.py` (spec section 19) exists — renaming now, before there's anything to compare against or an evaluation harness to run, would just be churn that risks breaking the working CLI/Streamlit imports for no immediate benefit. When that phase starts, the cleanest path is a thin alias module (e.g. `karyo_ai_dnn/classical_baseline.py` that imports and re-exports `karyo_ai.pipeline`) rather than moving or renaming the actual working files.

## Branches

- `pre-dnn-baseline` — tags the repository exactly as it was before this Phase 1 work started (i.e., identical to what's on GitHub right now — the refined classical baseline zip). This is your rollback point.
- `feature/maskrcnn-cytogenetics` — this Phase 1 work, branched from the same point.

I don't have push access to `Danngugi/Karyotype-ai-pipeline` (no connected GitHub credential), so these exist as local branches bundled into a `.bundle` file alongside this delivery — see the top-level message for the exact commands to pull them into your repo and push to GitHub.

## What to do next

1. Pull the two branches in (commands provided with the delivery).
2. Skim `docs/ARCHITECTURE_V2.md`'s status table — it's the honest source of truth for what's real vs. stubbed.
3. Decide on Phase 2 priorities: realistically, the next hard requirement is a real, expert-annotated dataset (even a small one — tens of images) before any of the segmentation/classification/pairing work can move from "code exists" to "code works." Everything else in the spec (FISH, aneuploidy, active learning, review UI) depends on segmentation actually working on real images first.
