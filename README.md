# Block Computing Prototype

Small computer vision prototype where digital balls are overlaid on video and interact with detected physical obstacles.

## Concept

The system processes a video frame by frame. YOLO is used for person detection/segmentation, while HSV color detectors are used for simple physical obstacles such as tape or post-its. The detected objects are converted into masks and used for filtering, occlusion, and collision with simulated balls.

## Project Structure

```text
block-computing/
├── main.py
├── config.yaml
├── detection/
│   ├── base.py              # shared Detection object / detector interface
│   ├── pipeline.py          # combines YOLO, HSV detectors, masks, and object rules
│   ├── person_yolo.py       # YOLOv8 segmentation for person detection
│   ├── person_filter.py     # person filtering helpers
│   ├── color_detector.py    # generic HSV color detector
│   ├── black_tape.py        # black tape detector
│   ├── yellow_tape.py       # yellow tape detector
│   ├── pink_postit.py       # pink post-it detector
│   └── registry.py          # detector setup from config
├── input/
│   ├── video_source.py      # video-file input
│   └── camera_source.py     # camera input
├── physics/
│   ├── ball.py              # ball model and movement
│   ├── spawner.py           # continuous ball spawning
│   ├── collision.py         # ball-obstacle collisions
│   └── particle_collision.py# ball-ball collisions
└── rendering/
    └── overlay.py           # drawing detections, balls, masks, and output video
```

## Run

```bash
pip install -r requirements.txt
python3 main.py
```

## Configuration

Most characteristics can be changed in `config.yaml` without editing code.

Important settings include:

- input video path
- enabled detectors
- YOLO person filtering
- object rules: filter, occluder, obstacle
- ball color, size, gravity, spawning, and collisions
- output/debug visualization

Example:

```yaml
input:
  mode: video
  path: videos/me_v1.mov

physics:
  ball_radius: 20
  ball_color: white
  gravity: 0.35
  spawn_enabled: true
  lower_boundary_mode: bounce
```

## Detection

- `person_filter`: YOLOv8 segmentation for person masks
- `black_tape`, `yellow_tape`, `pink_postit`: HSV/color-based detectors
- `object_rules`: controls whether detections are used as filters, occluders, or physical obstacles

## Physics

Balls can:

- spawn continuously
- fall with gravity
- bounce, collect, or fall through at the lower boundary
- collide with obstacles
- collide with each other

## Output

The result can be displayed live and/or saved as a video in the `output/` folder.

Debug options such as masks, detections, and person overlays can be enabled in `config.yaml`.

## Have Fun

Try different videos, obstacle types, colors, gravity settings, and ball behaviors. The whole point of the prototype is to experiment, tune, break things, and discover what kind of physical-digital interaction feels interesting. :)
