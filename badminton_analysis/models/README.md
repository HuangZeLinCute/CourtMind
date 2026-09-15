# Shuttlecock model source layout

This directory is the project-local home of model definitions used by
Good-Badminton.  Detection/video orchestration stays in
`badminton_analysis/shuttlecock`; model architecture and construction stay
here.

| Directory | Contents | Default weights |
| --- | --- | --- |
| `tracknet_v3/` | PyTorch tracker and trajectory rectifier architectures plus strict checkpoint loaders | `weights/tracknet/*.pt` |
| `tracknet_v4/` | Keras 3 portable Type-A/Type-B architecture; production uses Type B | `weights/tracknet_v4/*.keras` |
| `yolo11/` | Project-local validated loader for the Ultralytics YOLO11 runtime | `weights/yolo11s-ball.pt` |

`catalog.py` is the central inventory for names, frameworks, implementation
modules, and default weight locations.  Importing the catalog does not load an
ML framework.

## Source provenance

- TrackNetV3 was adapted for the checkpoint format used by the sibling
  `BadmintonTrackNet` project.  Its upstream MIT license is copied beside the
  implementation as `tracknet_v3/LICENSE.upstream`.
- TrackNetV4 was adapted from the sibling `TrackNetV4/src/models/TrackNetV4.py`
  TensorFlow implementation to Keras 3 portable operations.  Its upstream MIT
  license is copied as `tracknet_v4/LICENSE.upstream`.
- YOLO11's generic architecture/parser is provided by the declared
  `ultralytics` dependency.  Good-Badminton owns the integration boundary and
  keeps its trained shuttlecock checkpoint locally.  Vendoring an entire copy
  of Ultralytics would create a second package that can silently drift from the
  installed runtime, so it is deliberately not duplicated here.

The sibling source projects are reference copies and are not modified or
deleted by this integration.

