# AGENTS.md

## Scope
Applies to:
- `/Users/tejascheeti/Projects/Professional Gaming Software/playing-card-detection`

## Repo Purpose
Prototype computer-vision stack for detecting playing cards from live camera/video and turning detections into trackable card-state data.

## Current Code Surface
- `webcam_cards_iphone_tracker.py`
  - Real-time inference from iPhone/macOS camera input.
  - Tracks detected cards over time, card persistence/removal, and running/true count.
- `test_image.py`
  - Single-image model smoke test and annotated output generation.
- `convert_voc_yolo.py`
  - Converts VOC XML annotations to YOLO txt format and builds image list files.
- `creating_playing_cards_dataset.ipynb`
  - Dataset generation/augmentation workflow.

## Runtime Dependencies
- Python 3.10+ recommended.
- Core: `ultralytics`, `opencv-python`, `numpy`.
- Notebook/data tooling: `imgaug`, `shapely`, `tqdm`, `matplotlib`.

## Quick Run Commands
- Image test:
  - `python test_image.py --model best_yolo11.pt --image IMG_8515.jpeg --confidence 0.4`
- Live tracking:
  - `python webcam_cards_iphone_tracker.py --confidence 0.4 --stability 5 --removal 60`

## Working Rules
- Keep model weights and large binaries intact unless explicitly replacing with a new validated artifact.
- Prefer small, testable script changes over notebook-only logic changes.
- If changing class labels/detection schema, update all scripts that read class names.
- Preserve CLI flags where possible to avoid breaking demo workflows.
- Document any camera/backend assumptions (e.g., macOS `CAP_AVFOUNDATION`, camera index fallback).

## Integration Intent (Project-Level)
This repo is the perception layer in the broader product:
- Camera feed -> card detections -> structured hand/player events -> analytics/dashboard consumption in other repos.
