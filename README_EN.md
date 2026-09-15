# Good-Badminton: AI Badminton Hawk-Eye System 🏸

## Related Projects

Good-Badminton, Good-Tennis, and Good-Pickleball are part of the same family of computer-vision sports video analysis projects. They share the same core ideas: player detection, ball trajectory tracking, court coordinate mapping, movement statistics, and visualized outputs. Each project adapts the court model, ball target, and sport-specific rules to a different sport.

| Project | Sport | Stars |
| --- | --- | --- |
| [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) | Badminton video analysis | [![Good-Badminton stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Badminton?style=social)](https://github.com/yo-WASSUP/Good-Badminton/stargazers) |
| [Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis) | Tennis video analysis | [![Good-Tennis stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Tennis?style=social)](https://github.com/yo-WASSUP/Good-Tennis/stargazers) |
| [Good-Pickleball](https://github.com/yo-WASSUP/Good-Pickleball) | Pickleball video analysis | [![Good-Pickleball stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Pickleball?style=social)](https://github.com/yo-WASSUP/Good-Pickleball/stargazers) |

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Badminton?style=social)](https://github.com/yo-WASSUP/Good-Badminton/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yo-WASSUP/Good-Badminton?style=social)](https://github.com/yo-WASSUP/Good-Badminton/network/members)
[![GitHub license](https://img.shields.io/github/license/yo-WASSUP/Good-Badminton)](https://github.com/yo-WASSUP/Good-Badminton/blob/main/LICENSE)
[![RedNote](https://img.shields.io/badge/RedNote-ff2442)](https://www.xiaohongshu.com/explore/6a37b1d20000000011016229?xsec_token=ABod3wXBTiDppp6W2Ou0QHlu2eotUkeu27-ha64nFRR74=&xsec_source=pc_user)

**A computer-vision toolkit for badminton match video analysis**

[中文](README.md) | [English](README_EN.md)

</div>

## 🎬 Preview

![Good-Badminton analysis preview](assets/demo_en.gif)


## 🆕 Changelog

- **2026-06-27**: Improved automatic court-line detection.
- **2026-06-23**: Added automatic court boundary detection.
- **2026-06-20**: Initial open-source release.
- **2026-06-17**: Project documentation cleanup.
- **Current version**: Supports player pose detection, shuttlecock detection, court coordinate mapping, trajectory statistics, heatmaps, scatter plots, and annotated video output.
- **Experimental features**: Hit-point analysis and stroke statistics are still under active iteration and are mainly intended for research and secondary development.

## 🔮 Roadmap

- [x] Frame-by-frame badminton match video analysis
- [x] RTMPose / RTMO / YOLO Pose model support
- [x] YOLO shuttlecock detection model integration
- [x] Manual court annotation and court coordinate mapping
- [x] Player movement trajectory, speed, distance, and rally statistics
- [x] Chinese / English visualization text
- [x] Heatmap, scatter plot, and detection data export
- [ ] More stable hit-point recognition
- [ ] More accurate shuttlecock detection model
- [ ] More complete stroke statistics
- [x] Automatic court keypoint detection
- [x] React + FastAPI browser application
- [ ] Batch video analysis workflow

---

## ✨ Features

- **Player pose detection** - Supports RTMPose, RTMO, and Ultralytics YOLO Pose for human keypoint and skeleton detection.
- **Shuttlecock detection** - Uses a YOLO model to detect shuttlecock positions and draw trajectories in the output video.
- **Court coordinate mapping** - Manually annotates court keypoints and maps image coordinates to standard badminton court coordinates.
- **Automatic court detection** - Matches visible white/yellow court lines against a standard badminton court model, with manual corner correction in the React application.
- **Player position tracking** - Tracks upper-court and lower-court players separately and records movement trajectories.
- **Rally detection** - Detects rally start/end states from continuous court-view matching and records rally IDs in overlays and detection data.
- **Motion statistics** - Computes movement distance, current speed, maximum speed, and rally counts.
- **Visual output** - Generates annotated videos with skeletons, trajectories, statistics, and court trajectory overlays.
- **Position charts** - Automatically generates player position heatmaps and scatter plots.
- **Chinese / English display** - Switch visualization text with `--language zh/en`.
- **React application** - Upload videos, detect the court, configure parameters, and view results in the browser.
- **Local execution** - Videos, models, and analysis outputs stay on your local machine.

## Requirements

- Python 3.8+
- FFmpeg available in system `PATH`
- Shuttlecock YOLO detection weight, downloaded from [GitHub Releases](https://github.com/yo-WASSUP/Good-Badminton/releases/latest)

## Performance Requirements and Reference Speed

Recommended setup:

- GPU with 6GB+ VRAM. More VRAM helps with higher-resolution videos and larger pose models.
- 16GB+ system RAM.
- SSD storage for output videos, `detections.jsonl`, and visualization images.
- CPU execution is supported, but pose detection and shuttlecock detection will be much slower. It is best suited for short clips or feature checks.

Actual speed depends on the GPU, video resolution, pose model, preview display, and audio export settings.

For a 720p video with `--pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt` and `weights/yolo11s-ball.pt`, GPU timing logs are typically close to:

```text
pose 0.02s, shuttlecock 0.02s, shuttle draw 0.00s, players draw 0.01s, court draw 0.00s
```

Use `--performance-stats` to print a compact timing summary about every 5 seconds and identify whether the bottleneck is pose inference, shuttlecock detection, or drawing.

## 🚀 Installation

The default dependencies use CPU PyTorch and ONNX Runtime.

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Acceleration (Windows / NVIDIA)

Prerequisites:

- NVIDIA driver installed, and `nvidia-smi` works correctly.
- CUDA 12.1 PyTorch wheels are recommended.

PowerShell:

```bash
.\.venv\Scripts\activate

pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.20.1
```

Verify GPU availability:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Expected output includes:

```text
cuda: True
CUDAExecutionProvider
```

> Note: after installing GPU ONNX Runtime, `pip check` may report `rtmlib requires onnxruntime, which is not installed`. If provider verification shows `CUDAExecutionProvider`, do not reinstall CPU `onnxruntime`, because it may overwrite the GPU package.

Switch back to CPU dependencies:

```bash
pip install --force-reinstall -r requirements.txt
```


## 📝 Usage

### First Run Workflow (CLI)

1. Prepare the input video and shuttlecock detection weight.
2. Run the basic command:

```bash
python main.py --video-path videos/demo.mp4
```

3. If `--template-path` is not provided, the program opens a file picker for a court template image. Usually, choose a stable frame with clear court lines.
4. The program first tries to detect the court boundary automatically and saves `outputs/<video_name>/auto_court_preview.png`. Press Enter/Y in the preview window to accept it, or press M/R/Esc to switch to manual four-corner annotation.
5. If manual annotation is used, follow the prompt at the top of the image and click the four court corners in order: top-left, top-right, bottom-right, bottom-left.

![Court annotation example](assets/label_court_example.png)

6. After the four points are selected, the window shows a green court box and a blue pose-detection ROI. The ROI is generated automatically from the court area.
7. The annotation is saved to `outputs/<video_name>/court_annotations.txt`. Re-running with the same output directory reuses this file.
8. After analysis finishes, check `outputs/<video_name>/detect_<video_name>.mp4`, `detections.jsonl`, and `position_visualizations/`.

Why four court points are required:

- The four corners establish the mapping from image coordinates to standard badminton court coordinates.
- Player filtering mainly depends on court coordinates, which helps remove spectators, referees, and people outside the court.
- Upper/lower court player assignment, movement distance, speed, rally statistics, heatmaps, and scatter plots all depend on this mapping.
- Rally detection uses court template matching: consecutive court-view frames start a rally, and consecutive non-court-view frames end it.
- The pose ROI only reduces the inference area and improves speed. It is automatically expanded from the court area.
- Shuttlecock detection still runs on the full frame, with basic filtering based on the horizontal court range plus padding.

If the video angle, crop, or template image changes, delete the corresponding `court_annotations.txt` and annotate the four points again.


### Pose Model Selection

```bash
# Default: two-stage RTMPose balanced
python main.py --video-path videos/demo.mp4 --pose-family rtmpose --pose-mode balanced

# Lighter one-stage RTMO
python main.py --video-path videos/demo.mp4 --pose-family rtmo --pose-mode lightweight

# Use Ultralytics YOLO Pose
python main.py --video-path videos/demo.mp4 --pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt
```

RTMPose / RTMO modes:

- `lightweight`: prioritizes speed.
- `balanced`: default tradeoff between speed and quality.
- `performance`: larger model, slower, usually better for detection quality.

### Common Arguments

```text
--video-path                 Input video path, required
--output-dir                 Output directory, default outputs/<video_name>
--ball-model                 YOLO shuttlecock detection model path, default weights/yolo11s-ball.pt
--pose-family                Pose model family: rtmpose, rtmo, or yolo-pose
--pose-mode                  RTMPose / RTMO mode: lightweight, balanced, performance
--yolo-pose-model            YOLO pose model path or model name, default yolo11n-pose.pt
--template-path              Court template image path; opens a file picker if omitted
--pose-roi true|false                Show pose-detection ROI box, default true
--display true|false                 Show OpenCV preview window, default true
--skeletons true|false               Show human skeletons, default true
--player-trajectories true|false     Show player trajectories, default true
--court-trajectory true|false        Show court trajectory overlay, default true
--shuttlecock-trajectory true|false  Show shuttlecock trajectory, default true
--player-stats true|false            Show player statistics, default true
--performance-stats                  Print performance timings
--save-images                        Save processed frame images
--visualize-positions true|false     Generate heatmaps and scatter plots, default true
--audio true|false                   Keep original video audio, default true
--language {zh,en}                   Visualization language
```

## 📊 Outputs

Default output directory: `outputs/<video_name>/`.

- `metadata.json`: metadata for video, models, court annotation, and output files.
- `detections.jsonl`: per-frame detection records, including rally ID, players, hands, court coordinates, speed, and shuttlecock coordinates.
- `detect_<video_name>.mp4`: annotated output video with skeletons, trajectories, statistics, and rally IDs.
- `court_annotations.txt`: cached court annotation coordinates.
- `position_visualizations/heatmaps/`: player position heatmaps.
- `position_visualizations/scatter_plots/`: player position scatter plots.

### Position Visualization Examples

| Heatmap | Scatter Plot |
| --- | --- |
| ![Player position heatmap example](assets/match_heatmap_en.png) | ![Player position scatter plot example](assets/match_scatter_en.png) |

## 🧩 Project Structure

```text
main.py              # CLI entry and argument parsing; keeps python main.py ... usage
badminton_analysis/
├── system.py        # Main video analysis pipeline: BadmintonAnalysisSystem
├── court/           # Court annotation and coordinate mapping
├── data/            # JSON / JSONL output
├── detection/       # Shuttlecock detection and pose detection
├── media/           # Video/audio processing
├── models/          # TrackNetV3, TrackNetV4, and YOLO11 model code/catalog
├── shuttlecock/     # Model-independent detectors and three-model fusion
├── tracking/        # Player tracking
└── visualization/   # Video overlays, statistics charts, and position plots
backend/
├── api/             # FastAPI route handlers
└── services/analysis_pipeline.py  # Headless analysis orchestration
```

The React UI also provides **Batch Analysis**. It accepts up to 50 uploaded
videos, applies one shared configuration, auto-detects the court for each
video, and processes the queue sequentially to keep GPU/RAM usage bounded.

## Acknowledgements

Thanks to the RTMPose, RTMO, and OpenMMLab ecosystem for the pose-estimation foundations, and to [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) for the lightweight pose-estimation runtime.

Thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO object-detection algorithms and tooling.

Thanks to [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) for organizing and releasing badminton datasets, which provided important references for shuttlecock detection and trajectory analysis in this project.

## 📄 License

Project code and `weights/yolo11s-ball.pt` are licensed under Apache License 2.0. RTMPose / RTMO / YOLOX ONNX weights provided in Releases come from the OpenMMLab / RTMPose ecosystem, are used under their upstream Apache License 2.0, and retain their original attribution.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yo-WASSUP/Good-Badminton&type=Date)](https://www.star-history.com/#yo-WASSUP/Good-Badminton&Date)
