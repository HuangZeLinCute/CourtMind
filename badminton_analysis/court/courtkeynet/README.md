# CourtKeyNet (vendored copy)

CourtKeyNet is a badminton court corner keypoint-detection model
(octave features + polar transform attention + geometric constraints),
originally developed as a standalone repository:

    CourtKeyNet-main/courtkeynet

This directory contains a **self-contained copy of the inference-relevant
subset**, vendored into Good-Badminton so the analysis pipeline no longer
depends on an external absolute path:

```
courtkeynet/
├── __init__.py          # exports CourtKeyNet + load_weights
├── models/              # model architecture (courtkeynet / octave / polar / qcm)
├── utils/_safetensors.py# safetensors weight loading (obfuscated, do not edit)
└── configs/courtkeynet.yaml  # model configuration
```

## Weights

The finetuned weights live at the project-level weights folder (next to the
ball / pose models):

    weights/courtkeynet_finetuned.safetensors

## Sync / update

The original repository is **not modified**. If the upstream model changes,
re-copy the changed files from `CourtKeyNet-main/courtkeynet` into this folder
and keep `__init__.py` / `utils/__init__.py` adapters in sync. Note that the
vendored `utils/__init__.py` intentionally does NOT import the training-only
`dataloader` / `metrics` modules (they need albumentations / shapely, which
conflict with this project's dependency stack).
