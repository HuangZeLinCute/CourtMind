import os
import tempfile
from tkinter import filedialog
import tkinter as tk
import time
import argparse
from dataclasses import dataclass


@dataclass
class _DetectionRecord:
    """One cached TrackNetV3 detection (matches ShuttleDetection.to_dict)."""
    x: float
    y: float
    visible: bool
    confidence: float = None


def load_runtime_dependencies():
    """Load heavy runtime dependencies after argparse has handled --help."""
    global cv2, np, YOLO, CourtMapper, annotate_court, compute_expanded_roi, PlayerTracker
    global CourtTrajectoryVisualizer, ShuttlecockTracker
    global PlayerPoseVisualizer, StatsVisualizer, RTMPoseProcessor, YOLOPoseProcessor, vap
    global JsonlDetectionWriter, write_json, SCHEMA_VERSION

    yolo_config_dir = os.path.join(tempfile.gettempdir(), "good-badminton-ultralytics")
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)

    try:
        import cv2 as _cv2
        import numpy as _np
        from ultralytics import YOLO as _YOLO
        from .court.mapper import CourtMapper as _CourtMapper, annotate_court as _annotate_court
        from .court.mapper import compute_expanded_roi as _compute_expanded_roi
        from .tracking.player import PlayerTracker as _PlayerTracker
        from .visualization.court_trajectory import CourtTrajectoryVisualizer as _CourtTrajectoryVisualizer
        from .detection.shuttlecock import ShuttlecockTracker as _ShuttlecockTracker
        from .visualization.player_pose import PlayerPoseVisualizer as _PlayerPoseVisualizer
        from .visualization.stats import StatsVisualizer as _StatsVisualizer
        from .detection.rtmpose import RTMPoseProcessor as _RTMPoseProcessor
        from .detection.yolo_pose import YOLOPoseProcessor as _YOLOPoseProcessor
        from .media import video_audio as _vap
        from .data.writer import JsonlDetectionWriter as _JsonlDetectionWriter
        from .data.writer import write_json as _write_json
        from .data.writer import SCHEMA_VERSION as _SCHEMA_VERSION
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name}. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    cv2 = _cv2
    np = _np
    YOLO = _YOLO
    CourtMapper = _CourtMapper
    annotate_court = _annotate_court
    compute_expanded_roi = _compute_expanded_roi
    PlayerTracker = _PlayerTracker
    CourtTrajectoryVisualizer = _CourtTrajectoryVisualizer
    ShuttlecockTracker = _ShuttlecockTracker
    PlayerPoseVisualizer = _PlayerPoseVisualizer
    StatsVisualizer = _StatsVisualizer
    RTMPoseProcessor = _RTMPoseProcessor
    YOLOPoseProcessor = _YOLOPoseProcessor
    vap = _vap
    JsonlDetectionWriter = _JsonlDetectionWriter
    write_json = _write_json
    SCHEMA_VERSION = _SCHEMA_VERSION

class BadmintonAnalysisSystem:
    def __init__(self, video_path, show_display=True, 
                 show_skeletons=True, show_player_trajectories=True, 
                 show_court_trajectory=True, show_shuttlecock_trajectory=True,
                 show_player_stats=True, show_performance_stats=False, 
                 save_images=False, language='zh', output_dir=None,
                 ball_model_path='weights/yolo11s-ball.pt', template_path=None,
                 pose_mode='balanced', pose_family='rtmpose',
                 yolo_pose_model='yolo11n-pose.pt', show_pose_roi=True,
                 court_corners=None, auto_template_frame=None,
                 shuttle_model='yolo', tracknet_tracker_path=None,
                 tracknet_rectifier_path=None, tracknet_v4_weights_path=None):
        self.video_path = video_path
        self.show_display = show_display
        self.language = language
        self.template_path = template_path
        self.court_corners = court_corners
        self.auto_template_frame = auto_template_frame
        self.court_detector = None
        self.ball_model_path = ball_model_path
        self.pose_mode = pose_mode
        self.pose_family = pose_family
        self.yolo_pose_model = yolo_pose_model
        self.show_pose_roi = show_pose_roi

        # Shuttlecock detection backend: YOLO, TrackNetV3, or TrackNetV4.
        supported_shuttle_models = ("yolo", "tracknet", "tracknet_v4", "ensemble")
        self.shuttle_model = shuttle_model if shuttle_model in supported_shuttle_models else "yolo"
        self.tracknet_tracker_path = tracknet_tracker_path or "weights/tracknet/tracknet_v3_tracker.pt"
        self.tracknet_rectifier_path = tracknet_rectifier_path or "weights/tracknet/tracknet_v3_rectifier.pt"
        self.tracknet_v4_weights_path = (
            tracknet_v4_weights_path
            or "weights/tracknet_v4/tracknet_v4_type_b.keras"
        )
        self.tracknet_trajectory = None
        self.tracknet_trajectories = {}
        self.ensemble_tracker = None
        self.tracknet_total_frames = 0


        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_court_trajectory = show_court_trajectory
        self.show_shuttlecock_trajectory = show_shuttlecock_trajectory
        self.show_player_stats = show_player_stats
        self.show_performance_stats = show_performance_stats
        self.save_images = save_images  

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Input video not found: {self.video_path}\n"
                "Pass a valid video file with --video-path."
            )
        # YOLO ball model: always loaded so that (a) legacy yolo mode works
        # unchanged and (b) TrackNetV3 failures can fall back to YOLO11.
        self.yolo_ball_model = None
        if os.path.exists(self.ball_model_path):
            self.yolo_ball_model = YOLO(self.ball_model_path)
        elif self.shuttle_model in ("yolo", "ensemble"):
            raise FileNotFoundError(
                f"Ball detection model not found: {self.ball_model_path}\n"
                "Download or train a YOLO shuttlecock model and place it at "
                "weights/yolo11s-ball.pt, or pass its path with --ball-model."
            )
        else:
            print(
                f"[Shuttlecock] {self.ball_model_path} not found; YOLO fallback "
                "will be unavailable if the selected temporal model fails."
            )

        if self.pose_family == 'yolo-pose':
            self.rtmpose_processor = YOLOPoseProcessor(model_path=self.yolo_pose_model)
        else:
            self.rtmpose_processor = RTMPoseProcessor(mode=self.pose_mode, pose_family=self.pose_family)

        self.last_stats_update_frame = 0


        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path)[:-4]
        self.save_dir = output_dir or os.path.join('outputs', self.video_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.images_save_dir = os.path.join(self.save_dir, 'detect_images')
        os.makedirs(self.images_save_dir, exist_ok=True)
        

        self.metadata_path = os.path.join(self.save_dir, "metadata.json")
        self.detections_path = os.path.join(self.save_dir, "detections.jsonl")
        self.output_video_path = os.path.join(self.save_dir, f"detect_{self.video_name}.mp4")
        self.detection_writer = None
        

        self.player_1_hand = "right"  
        self.player_2_hand = "right"  
        self.start_time = None
        self.end_time = None
        

        self.shuttlecock_tracker = ShuttlecockTracker(
            yolo_ball_model=self.yolo_ball_model,
            trajectory_length=30,
            show_trajectory=self.show_shuttlecock_trajectory,
            show_performance_stats=False
        )
        
        self.player_pose_visualizer = PlayerPoseVisualizer(
            rtmpose_processor=self.rtmpose_processor,
            show_skeletons=self.show_skeletons,
            show_player_trajectories=self.show_player_trajectories,
            show_performance_stats=False
        )
        

        self.court_trajectory_visualizer = CourtTrajectoryVisualizer()
        

        self.stats_update_interval_frames = 0
        self.cached_movement_stats = {}

        self.is_court_view_count = 0
        self.consecutive_non_court_frames = 0
        self.rally_active = False
        self.rally_count = 0  
        self.fps = 30  
        self.court_view_frames_threshold = 5
        self.non_court_frames_threshold = 5

        self.frame_width = 0
        self.frame_height = 0
        self.performance_log_interval_frames = 150
    def process_video(self, progress_callback=None):
        """Process the input video.

        Args:
            progress_callback: Optional callable(frame_count, total_frames)
                invoked after each frame so callers can report progress.
        """
        self.start_time = time.time()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
        video_duration = total_frames / fps
        

        self.fps = fps
        self.performance_log_interval_frames = max(1, int(fps * 5))
        
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = self._setup_video_writer(self.frame_width, self.frame_height, fps)


        template_path = None
        if self.template_path:
            # --- Legacy path: user-supplied court template image ---
            template_path = self._get_template_path()
            template_gray, template_color = self._load_template(template_path, cap)
            corners, roi_corners, mid_height = self._setup_court_annotation(template_color)
            self.court_detector = "template"
        elif self.court_corners is not None and len(self.court_corners) == 4:
            # --- Court corners provided (e.g. CourtKeyNet detected in WebUI) ---
            corners = [(int(x), int(y)) for x, y in self.court_corners]
            roi_corners = compute_expanded_roi(corners, (self.frame_height, self.frame_width, 3))
            mid_height = CourtMapper(corners).mid_height
            self.court_detector = "courtkeynet"
            if self.auto_template_frame is not None:
                template_color = cv2.resize(self.auto_template_frame, (self.frame_width, self.frame_height))
                template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
            else:
                template_color = None
                template_gray = None
        else:
            # --- Fully automatic: CourtKeyNet from video, original detector as fallback ---
            from .court.courtkeynet_detector import resolve_court_from_video
            corners, roi_corners, mid_height, auto_frame, detector_name = resolve_court_from_video(
                self.video_path
            )
            if corners is None:
                raise RuntimeError(
                    "自动球场检测失败：CourtKeyNet 与原检测器均未能从视频中检测到球场。\n"
                    "请使用 --template-path 提供球场模板图片，或更换视频片段后重试。"
                )
            self.court_detector = detector_name
            if auto_frame is not None:
                template_color = cv2.resize(auto_frame, (self.frame_width, self.frame_height))
                template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
            else:
                template_color = None
                template_gray = None


        self.court_corners = corners
        self.court_roi_corners = roi_corners

        self._write_metadata(fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height)
        self.detection_writer = JsonlDetectionWriter(self.detections_path)
        

        self.court_mapper = CourtMapper(corners)
        self.player_pose_visualizer.court_mapper = self.court_mapper
        self.player_tracker = PlayerTracker(corners=corners, threshold=mid_height, history_size=30,
                                          detection_writer=self.detection_writer, fps=fps)
        

        self.stats_visualizer = StatsVisualizer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            language=self.language
        )
        
        if self.shuttle_model == "ensemble":
            failures = []
            for model_name in ("tracknet_v4", "tracknet"):
                try:
                    self._prepare_tracknet_trajectory(total_frames, model_name=model_name)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{model_name}: {exc}")
            if failures:
                raise RuntimeError(
                    "Three-model ensemble requires TrackNetV4, TrackNetV3, and YOLO11; "
                    + "; ".join(failures)
                )
            print("[Shuttlecock] Three-model ensemble ready: TrackNetV4 + TrackNetV3 + YOLO11.")
        elif self.shuttle_model in ("tracknet", "tracknet_v4"):
            self._prepare_tracknet_trajectory(total_frames)
        
        frame_count = 0
        detect_frame_count = 0


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame, detect_frame_count = self._process_frame(frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count)
            if progress_callback is not None:
                progress_callback(frame_count, total_frames)

        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        print(f"\n处理完成:")
        print(f"原始视频时长: {video_duration:.2f} 秒")
        print(f"处理耗时: {processing_time:.2f} 秒")
        print(f"处理速度比: {processing_time/video_duration:.2f}x")
        
        self._cleanup(cap)

    def _prepare_tracknet_trajectory(self, total_frames, model_name=None):
        """Run the selected TrackNet model once over the whole video and
        store the 0-based per-frame trajectory.  Falls back to YOLO11 if
        the temporal model cannot load or fails."""
        import json

        from .shuttlecock import TrackNetV3Detector, TrackNetV4Detector

        selected_model = model_name or self.shuttle_model
        model_label = "TrackNetV4" if selected_model == "tracknet_v4" else "TrackNetV3"
        cache_path = os.path.join(self.save_dir, f"{selected_model}_trajectory.json")
        detector = None
        try:
            if os.path.isfile(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and len(cached) == total_frames:
                    trajectory = [_DetectionRecord(**d) for d in cached]
                    self.tracknet_trajectories[selected_model] = trajectory
                    if model_name is None:
                        self.tracknet_trajectory = trajectory
                    self.tracknet_total_frames = total_frames
                    print(
                        f"[{model_label}] Reusing cached trajectory "
                        f"({total_frames} frames): {cache_path}"
                    )
                    return
        except Exception as exc:  # noqa: BLE001
            print(f"[{model_label}] Cache unreadable ({exc}); re-running tracking.")

        try:
            if selected_model == "tracknet_v4":
                detector = TrackNetV4Detector(weights_path=self.tracknet_v4_weights_path)
            else:
                detector = TrackNetV3Detector(
                    tracker_path=self.tracknet_tracker_path,
                    rectifier_path=self.tracknet_rectifier_path,
                    device="auto",
                )
            trajectory = detector.process_video(self.video_path)
            detector.close()
            detector = None
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.tracknet_trajectories[selected_model] = trajectory
            if model_name is None:
                self.tracknet_trajectory = trajectory
            self.tracknet_total_frames = len(trajectory)
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump([det.to_dict() for det in trajectory], f, ensure_ascii=False)
                print(f"[{model_label}] Trajectory cached: {cache_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{model_label}] Cache write failed ({exc})")
            print(f"[Shuttlecock] Using {model_label} detector.")
        except Exception as exc:  # noqa: BLE001
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
            print(f"[{model_label}] Failed: {exc}")
            self.tracknet_trajectories.pop(selected_model, None)
            if model_name is not None:
                raise
            self.tracknet_trajectory = None
            if self.yolo_ball_model is not None:
                print("[Shuttlecock] Falling back to YOLO11.")
                self.shuttle_model = "yolo"
            else:
                raise RuntimeError(
                    f"{model_label} failed and no YOLO ball model is available: "
                    f"{exc}"
                ) from exc

    def _tracknet_position(self, frame_count):
        """Return (x, y, visible) for a 1-based video frame count."""
        if self.tracknet_trajectory is None:
            return None
        idx = frame_count - 1
        if idx < 0 or idx >= len(self.tracknet_trajectory):
            return None
        det = self.tracknet_trajectory[idx]
        return det.x, det.y, det.visible

    def _ensemble_detection(self, frame, roi_corners, frame_count):
        """Run YOLO for this frame and fuse it with both cached TrackNets."""
        from .shuttlecock import ShuttleDetection
        from .shuttlecock.temporal_ensemble import TemporalEnsemble

        if self.ensemble_tracker is None:
            self.ensemble_tracker = TemporalEnsemble(
                (self.frame_width, self.frame_height), self.fps
            )

        detections = []
        for source in ("tracknet_v4", "tracknet"):
            trajectory = self.tracknet_trajectories.get(source, [])
            idx = frame_count - 1
            if 0 <= idx < len(trajectory):
                detection = trajectory[idx]
                if self.shuttlecock_tracker.point_in_roi(
                    (detection.x, detection.y), roi_corners
                ):
                    detections.append((source, detection))

        yolo_candidates = self.shuttlecock_tracker.detect_candidates(
            frame, roi_corners=roi_corners
        )
        for candidate in yolo_candidates:
            detections.append(("yolo", ShuttleDetection(
                *candidate["point"], True, candidate["confidence"])))
        fused = self.ensemble_tracker.update(detections, frame_count)
        return self.shuttlecock_tracker.set_detection(
            fused.x,
            fused.y,
            visible=fused.visible,
            confidence=fused.confidence,
        )

    def _write_metadata(self, fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height):
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "video": {
                "path": self.video_path,
                "name": self.video_name,
                "fps": float(fps),
                "total_frames": int(total_frames),
                "duration_sec": float(video_duration),
                "width": int(self.frame_width),
                "height": int(self.frame_height),
            },
            "models": {
                "shuttlecock": (
                    "tracknet_v4_tracknet_v3_yolo11_ensemble"
                    if self.shuttle_model == "ensemble"
                    else "tracknet_v4_type_b"
                    if self.shuttle_model == "tracknet_v4"
                    else "tracknet_v3"
                    if self.shuttle_model == "tracknet"
                    else self.ball_model_path
                ),
                "shuttle_detector": self.shuttle_model,
                "shuttle_components": (
                    ["tracknet_v4_type_b", "tracknet_v3", self.ball_model_path]
                    if self.shuttle_model == "ensemble"
                    else None
                ),
                "shuttle_fusion": (
                    "multi_hypothesis_kalman_v2"
                    if self.shuttle_model == "ensemble"
                    else None
                ),
            },
            "court": {
                "template_path": template_path,
                "detector": self.court_detector,
                "corners": corners,
                "roi_corners": roi_corners,
                "mid_height": mid_height,
                "coordinate_system": {
                    "unit": "meter",
                    "width": 6.1,
                    "length": 13.4,
                },
            },
            "outputs": {
                "video": self.output_video_path,
                "detections": self.detections_path,
            },
        }
        write_json(self.metadata_path, metadata)

    def _process_frame(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count):

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame = self.draw_court_roi(frame, corners, roi_corners)

        is_court = self.is_court_view(gray_frame, template_gray)
        
        if is_court:
            self.is_court_view_count += 1
            self.consecutive_non_court_frames = 0
        else:
            self.consecutive_non_court_frames += 1
            self.is_court_view_count = 0
            

        if self.is_court_view_count >= self.court_view_frames_threshold and not self.rally_active:
            self.rally_active = True

            self.rally_count += 1

            self.player_tracker.start_new_rally()
            

        if self.consecutive_non_court_frames >= self.non_court_frames_threshold and self.rally_active:
            self.rally_active = False

            self.shuttlecock_tracker.clear_trajectory()


        if not is_court:
            if self.ensemble_tracker is not None:
                self.ensemble_tracker.reset()
                self.shuttlecock_tracker.clear_trajectory()
            return frame, detect_frame_count

        detect_frame_count += 1

        # Fusion must see the original pixels, before ROI boxes or pose drawing.
        ensemble_position = None
        if self.shuttle_model == "ensemble":
            ensemble_position = self._ensemble_detection(frame, roi_corners, frame_count)

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        roi = frame[y1:y2, x1:x2]
        if self.show_pose_roi:
            cv2.rectangle(frame, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
            cv2.putText(frame, "Pose ROI", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


        pose_t0 = time.time()
        centroids, point_left_hands, point_right_hands = self.player_pose_visualizer.detect_players(roi, x1, y1)
        pose_elapsed = time.time() - pose_t0

        ball_t0 = time.time()
        if self.shuttle_model == "ensemble":
            detected_ball_position = ensemble_position
        elif self.shuttle_model in ("tracknet", "tracknet_v4"):
            tracknet_pos = self._tracknet_position(frame_count)
            if tracknet_pos is None:
                detected_ball_position = [0, 0]
            else:
                x, y, visible = tracknet_pos
                detected_ball_position = self.shuttlecock_tracker.set_detection(
                    x, y, visible=visible, confidence=0.5
                )
        else:
            detected_ball_position = self.shuttlecock_tracker.detect_ball(frame, roi_corners=roi_corners)
        ball_elapsed = time.time() - ball_t0
        ball_position = self.shuttlecock_tracker.update_trajectory(
            detected_ball_position, roi_corners,
            temporal_validated=self.shuttle_model == "ensemble",
        )
        

        shuttle_draw_t0 = time.time()
        self.shuttlecock_tracker.handle_visualization(frame)
        shuttle_draw_elapsed = time.time() - shuttle_draw_t0
        

        players = self.player_tracker.update(frame_count, centroids, ball_position, 
                                             point_left_hands, point_right_hands, detect_frame_count,
                                             fusion=(self.ensemble_tracker.diagnostics
                                                     if self.shuttle_model == "ensemble" else None))
        

        if frame_count == 1 or not self.cached_movement_stats:
            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.stats_update_interval_frames = int(self.player_tracker.fps * 0.5)

        if frame_count - self.last_stats_update_frame >= self.stats_update_interval_frames:

            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.last_stats_update_frame = frame_count


        should_log_performance = (
            self.show_performance_stats
            and self.performance_log_interval_frames > 0
            and frame_count % self.performance_log_interval_frames == 0
        )

        t0 = time.time()

        self.player_pose_visualizer.draw_players(
            frame=frame, 
            player_tracker=self.player_tracker, 
            cached_movement_stats=self.cached_movement_stats,
            stats_visualizer=self.stats_visualizer if self.show_player_stats else None,
            rally_count=self.rally_count
        )
        t1 = time.time()
        players_draw_elapsed = t1 - t0
        

        court_draw_elapsed = 0.0
        if self.show_court_trajectory:
            t0 = time.time()
            frame = self.court_trajectory_visualizer.draw_overlay(frame, self.player_tracker.court_history)
            t1 = time.time()
            court_draw_elapsed = t1 - t0

        if should_log_performance:
            print(
                f"Frame {frame_count}: pose {pose_elapsed:.2f}s, "
                f"shuttlecock {ball_elapsed:.2f}s, "
                f"shuttle draw {shuttle_draw_elapsed:.2f}s, "
                f"players draw {players_draw_elapsed:.2f}s, "
                f"court draw {court_draw_elapsed:.2f}s"
            )
        

        if frame is not None:
            if self.show_display:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
            out.write(frame)

            if self.save_images:
                cv2.imwrite(os.path.join(self.images_save_dir, f"{frame_count}.png"), frame)
        return frame, detect_frame_count

    def _get_template_path(self):
        """Get the court template image path."""
        if self.template_path:
            if not os.path.exists(self.template_path):
                raise FileNotFoundError(
                    f"Court template image not found: {self.template_path}"
                )
            return self.template_path

        try:
            root = tk.Tk()
            root.withdraw()
            template_path = filedialog.askopenfilename(
                title="Select court template image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            root.destroy()
        except Exception as exc:
            raise RuntimeError(
                "Unable to open the template picker. In headless environments, "
                "pass a court template image path with --template-path."
            ) from exc

        if not template_path:
            raise RuntimeError(
                "No court template image selected. Pass --template-path to run "
                "without the file picker."
            )
        return template_path

    def _load_template(self, template_path, cap):
        """Load and resize the court template image."""
        template_gray = cv2.imread(template_path, 0)
        template_color = cv2.imread(template_path)
        if template_gray is None or template_color is None:
            raise RuntimeError(f"Unable to read court template image: {template_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        template_gray = cv2.resize(template_gray, (frame_width, frame_height))
        template_color = cv2.resize(template_color, (frame_width, frame_height))
        
        return template_gray, template_color

    def _setup_video_writer(self, frame_width, frame_height, fps):

        self.temp_output_video_path = os.path.join(self.save_dir, f"temp_detect_{self.video_name}.mp4")
        

        self.video_writer = vap.setup_video_writer(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            temp_output_path=self.temp_output_video_path
        )
        
        return self.video_writer

    def _setup_court_annotation(self, template_color):
        """Set up court annotation."""

        if os.path.exists(os.path.join(self.save_dir, 'court_annotations.txt')):
            with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'r') as f:
                corners = eval(f.readline().split('=')[1])
                f.readline()
                mid_height = eval(f.readline().split('=')[1])
                roi_corners = compute_expanded_roi(corners, template_color.shape)
        else:
            auto_preview_path = os.path.join(self.save_dir, 'auto_court_preview.png')
            corners, roi_corners, mid_height = annotate_court(template_color, auto_preview_path=auto_preview_path)
       
        if not corners or not roi_corners or len(corners) != 4 or len(roi_corners) != 2:
            raise RuntimeError("Court annotation is incomplete: click 4 court corners in order. ROI is generated automatically.")

        with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'w') as f:
            f.write(f"corners={corners}\n")
            f.write(f"roi_corners={roi_corners}\n")
            f.write(f"mid_height={mid_height}\n")
        return corners, roi_corners, mid_height

    def _cleanup(self, cap):
        """Clean up resources and merge audio when needed."""
        if self.detection_writer is not None:
            self.detection_writer.close()
            self.detection_writer = None

        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            time.sleep(1)

        cap.release()

        if self.show_display:
            cv2.destroyAllWindows()

        if hasattr(self, 'keep_audio') and self.keep_audio:
            vap.process_video_with_audio(
                video_path=self.video_path,
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path,
                save_dir=self.save_dir
            )
        else:
            vap.process_video_without_audio(
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path
            )

    def analyze_shuttlecock(self, roi_corners, corners):
        """Hit-point analysis is currently disabled."""
        raise RuntimeError(
            "Hit-point analysis is disabled until it is migrated to detections.jsonl."
        )

    def is_court_view(self, frame, template_gray, threshold=0.75):
        """Return whether the frame matches the court template.

        Without a template (fully automatic CourtKeyNet flow could not produce
        one) every frame is treated as a court view so the rally logic keeps
        working instead of crashing."""
        if template_gray is None:
            return True
        result = cv2.matchTemplate(frame, template_gray, cv2.TM_CCOEFF_NORMED)
        # print("match score: ", result)
        return np.max(result) >= threshold

    def draw_court_roi(self, frame, corners, roi_corners):
        self.court_mapper = CourtMapper(corners)
        overlay, mid_height_int = self.court_mapper.draw_court_overlay(frame)
        cv2.rectangle(overlay, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
        return overlay
