# Professional Gaming Software - `playing-card-detection` Context

## Scope
Applies to:
- `/Users/tejascheeti/Projects/Professional Gaming Software/playing-card-detection`

For workspace-level context across all repos, see:
- `/Users/tejascheeti/Projects/Professional Gaming Software/AGENTS.md`

## Repo Purpose
`playing-card-detection` is the computer vision prototype repo.
It focuses on converting camera/video frames of blackjack into card detections and tracked card-state signals.

## Current Implementation
Main scripts:
- `test_image.py`
  - smoke test for image-based inference
  - outputs annotated detections
- `webcam_cards_iphone_tracker.py`
  - live camera inference path on macOS (AVFoundation)
  - card persistence/removal tracking
  - running count and true count overlays
- `convert_voc_yolo.py`
  - VOC XML to YOLO label conversion utility
- `creating_playing_cards_dataset.ipynb`
  - dataset generation/augmentation workflow

## Runtime Dependencies
Core:
- Python 3.10+
- `ultralytics`
- `opencv-python`
- `numpy`

Notebook/data tooling:
- `imgaug`
- `shapely`
- `tqdm`
- `matplotlib`

## Quick Commands
Image inference:
```bash
cd "/Users/tejascheeti/Projects/Professional Gaming Software/playing-card-detection"
python test_image.py --model best_yolo11.pt --image IMG_8515.jpeg --confidence 0.4
```

Live camera tracking:
```bash
cd "/Users/tejascheeti/Projects/Professional Gaming Software/playing-card-detection"
python webcam_cards_iphone_tracker.py --confidence 0.4 --stability 5 --removal 60
```

## Integration Role In Product
This repo is intended to produce structured perception outputs for downstream systems:
1. camera feed input
2. card detections and temporal tracking
3. hand/player event construction (future integration step)
4. analytics/dashboard consumption in sibling repos

## Constraints and Editing Rules
- Keep large model artifacts and demo assets intact unless explicitly replacing with validated versions.
- Preserve CLI behavior for existing scripts unless there is a clear migration path.
- If class labels or detection schema changes, update all dependent scripts and docs.
- Treat this as prototype code, but keep changes reproducible and well documented.
