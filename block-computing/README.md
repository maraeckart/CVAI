# Block Computing CV Mockup

A modular OpenCV prototype for testing “block computing” on prerecorded video first, with a clean path toward real-time projection later.

## Stage 1 detection design

This version uses a unified detection pipeline:

1. YOLO segmentation detects the person first.
2. The person mask is used to filter out body/clothing pixels before obstacle detection.
3. HSV/color detectors find pink post-its, yellow tape, and black tape.
4. `object_rules` decides which detections become physics obstacles, occluders, or filters.

This means the app no longer has a separate `person_motion` obstacle detector. The person is handled only by YOLO.

## Later custom YOLO plan

Later, replace the HSV detectors with a custom YOLO segmentation model trained on classes such as:

- `person`
- `pink_postit`
- `yellow_tape`
- `black_tape`

The rest of the app can keep using the same `object_rules` structure.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Put your test video at:

```text
videos/test.mp4
```

or change `input.path` in `config.yaml`.

## Important config

```yaml
object_rules:
  person_yolo:
    use_as_filter: true
    use_as_occluder: true
    use_as_obstacle: false
```

Set `use_as_obstacle: true` only if you want the dots to bounce off the person.
