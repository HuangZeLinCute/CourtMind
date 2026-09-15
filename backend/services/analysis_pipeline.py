import os
import shutil
import subprocess

import cv2
import numpy as np

from badminton_analysis.court.mapper import (
    CourtMapper,
    auto_detect_preview,
    compute_expanded_roi,
    resolve_court_corners,
)
from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies

_dependencies_loaded = False


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def imread_safe(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread with fallback for Unicode paths on Windows."""
    img = cv2.imread(path, flags)
    if img is None and os.path.isfile(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, flags)
        except Exception:
            pass
    return img


def _ensure_dependencies():
    global _dependencies_loaded
    if not _dependencies_loaded:
        load_runtime_dependencies()
        _dependencies_loaded = True


def _normalize_corners(corners):
    """Return four integer (x, y) tuples or raise ValueError."""
    if corners is None:
        raise ValueError("Court corners are required.")

    arr = np.asarray(corners, dtype=np.float32)
    if arr.shape != (4, 2) or not np.isfinite(arr).all():
        raise ValueError(f"Expected court corners with shape (4, 2), got {arr.shape}.")

    return [(int(round(float(x))), int(round(float(y)))) for x, y in arr]


# -----------------------------------------------------------------------------
# Court calibration helpers
# -----------------------------------------------------------------------------

def prepare_court(template_path, manual_corners=None):
    """Detect (or apply manual) court corners on a court template image.

    Returns:
        dict with keys: corners, roi_corners, mid_height, preview_bgr.
        On failure corners/roi_corners/mid_height are None.
    """
    _ensure_dependencies()
    template_color = imread_safe(template_path)
    if template_color is None:
        raise FileNotFoundError(f"Cannot read template image: {template_path}")

    if manual_corners and len(manual_corners) == 4:
        corners, roi_corners, mid_height = resolve_court_corners(
            template_color, manual_corners=manual_corners
        )
        preview = template_color.copy()
        if corners:
            pts = [list(c) for c in corners]
            cv2.polylines(preview, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 3)
            for idx, pt in enumerate(corners, 1):
                cv2.circle(preview, pt, 6, (0, 0, 255), -1)
                cv2.putText(
                    preview,
                    str(idx),
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
    else:
        corners_auto, preview = auto_detect_preview(template_color)
        if preview is not None:
            h, w = template_color.shape[:2]
            preview = cv2.resize(preview, (w, h))
        if corners_auto:
            corners, roi_corners, mid_height = resolve_court_corners(
                template_color, manual_corners=corners_auto
            )
        else:
            corners, roi_corners, mid_height = None, None, None

    return {
        "corners": corners,
        "roi_corners": roi_corners,
        "mid_height": mid_height,
        "preview_bgr": preview,
    }


# -----------------------------------------------------------------------------
# Browser-compatible output video
# -----------------------------------------------------------------------------

def _find_ffmpeg():
    """Locate an ffmpeg binary — prefer imageio_ffmpeg (bundled with moviepy)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    path = shutil.which("ffmpeg")
    if path:
        return path
    return None


def _reencode_for_browser(video_path, output_dir):
    """Re-encode video to H.264 so browsers can play it.

    OpenCV's mp4v codec is not browser-compatible.  This converts to H.264 via
    ffmpeg and returns the web-friendly path, or the original path if ffmpeg is
    unavailable/fails.
    """
    if not os.path.isfile(video_path):
        return video_path

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return video_path

    web_path = os.path.join(output_dir, "web_" + os.path.basename(video_path))
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                video_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-movflags",
                "faststart",
                web_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
            check=True,
        )
        if os.path.isfile(web_path) and os.path.getsize(web_path) > 0:
            return web_path
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return video_path


# -----------------------------------------------------------------------------
# Court coordinate helpers
# -----------------------------------------------------------------------------

def _scale_corners_to_video(corners, template_path, video_path):
    """Scale court corners from template resolution to video-frame resolution.

    When no template is used (the normal CourtKeyNet video flow), corners are
    already expressed in video-frame coordinates and are returned unchanged.
    """
    if not template_path or not os.path.isfile(template_path):
        return corners

    template_img = imread_safe(template_path)
    if template_img is None:
        return corners

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return corners
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tmpl_h, tmpl_w = template_img.shape[:2]
    if tmpl_w == frame_w and tmpl_h == frame_h:
        return corners

    sx = frame_w / tmpl_w
    sy = frame_h / tmpl_h
    return [(int(x * sx), int(y * sy)) for x, y in corners]


# -----------------------------------------------------------------------------
# Default automatic video court detection
# -----------------------------------------------------------------------------

def detect_court_from_video(video_path):
    """Auto-detect court corners directly from a video.

    The detector implementation in ``courtkeynet_detector`` is responsible for
    the detector chain.  In the intended setup this is:

        CourtKeyNet -> original Good-Badminton CV detector fallback

    Returns the same court contract as ``prepare_court`` plus:
        best_frame: raw representative video frame used for calibration/rally matching
        detector:   name of the detector that produced the accepted result
    """
    _ensure_dependencies()
    from badminton_analysis.court.courtkeynet_detector import resolve_court_from_video

    corners, roi_corners, mid_height, best_frame, detector_name = resolve_court_from_video(
        video_path
    )

    if corners is None or best_frame is None:
        return {
            "corners": None,
            "roi_corners": None,
            "mid_height": None,
            "preview_bgr": best_frame,
            "best_frame": best_frame,
            "detector": detector_name,
        }

    corners = _normalize_corners(corners)

    preview = best_frame.copy()
    pts = np.asarray(corners, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(preview, [pts], True, (0, 255, 0), 3)
    for idx, (x, y) in enumerate(pts.tolist(), 1):
        cv2.circle(preview, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(
            preview,
            str(idx),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return {
        "corners": corners,
        "roi_corners": roi_corners,
        "mid_height": mid_height,
        "preview_bgr": preview,
        "best_frame": best_frame,
        "detector": detector_name,
    }


# -----------------------------------------------------------------------------
# Full headless analysis
# -----------------------------------------------------------------------------

def run_analysis(video_path, template_path, corners, options, progress_cb=None,
                 auto_template_frame=None, output_dir=None):
    """Run the full Good-Badminton analysis pipeline headlessly.

    Args:
        video_path: Input video path.
        template_path: Legacy court-template path, or None for the normal
            CourtKeyNet-from-video workflow.
        corners: Four confirmed court corners.  They are in video resolution when
            template_path is None, otherwise template resolution.
        options: Analysis options mirroring CLI flags.
        progress_cb: Optional callable(frame_count, total_frames).
        auto_template_frame: Raw representative BGR frame selected during automatic
            court detection.  It is passed to the existing rally/template-matching
            logic when no legacy template was provided.
        output_dir: Optional explicit output directory. When omitted, an
            ``outputs/analysis_<name>`` directory is used.

    Returns:
        dict containing output paths.
    """
    _ensure_dependencies()
    # Court calibration must already have been explicitly confirmed by the React application.
    # run_analysis never performs hidden re-detection.
    corners = _normalize_corners(corners)
    corners = _scale_corners_to_video(corners, template_path, video_path)
    corners = _normalize_corners(corners)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    roi_corners = compute_expanded_roi(corners, (frame_h, frame_w, 3))
    mapper = CourtMapper(corners)
    mid_height = mapper.mid_height

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir:
        output_dir = os.path.abspath(output_dir)
    else:
        output_dir = os.path.abspath(os.path.join("outputs", f"analysis_{video_name}"))
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "court_annotations.txt"), "w", encoding="utf-8") as f:
        f.write(f"corners={corners}\n")
        f.write(f"roi_corners={roi_corners}\n")
        f.write(f"mid_height={mid_height}\n")

    language = options.get("language", "zh")
    pose_family = options.get("pose_family", "yolo-pose")
    pose_mode = options.get("pose_mode", "balanced")
    yolo_pose_model = options.get("yolo_pose_model", "weights/yolo11n-pose.pt")
    ball_model = options.get("ball_model", "weights/yolo11s-ball.pt")
    shuttle_model = options.get("shuttle_model", "yolo")
    keep_audio = options.get("audio", True)
    # Render policy: keep pose skeletons, trajectories, and the top-down court
    # as visual evidence. Numeric statistics remain in the synchronized UI.
    show_skeletons = True
    show_player_trajectories = options.get("show_player_trajectories", True)
    show_court_trajectory = options.get("show_court_trajectory", True)
    show_shuttlecock_trajectory = options.get("show_shuttlecock_trajectory", True)
    show_player_stats = False
    show_pose_roi = False
    visualize_positions = options.get("visualize_positions", True)

    from backend.core.config import TRACKNET_V4_WEIGHTS_PATH

    system = BadmintonAnalysisSystem(
        video_path,
        show_display=False,
        show_skeletons=show_skeletons,
        show_player_trajectories=show_player_trajectories,
        show_court_trajectory=show_court_trajectory,
        show_shuttlecock_trajectory=show_shuttlecock_trajectory,
        show_player_stats=show_player_stats,
        show_performance_stats=False,
        save_images=False,
        language=language,
        output_dir=output_dir,
        ball_model_path=ball_model,
        template_path=template_path,
        pose_mode=pose_mode,
        pose_family=pose_family,
        yolo_pose_model=yolo_pose_model,
        show_pose_roi=show_pose_roi,
        court_corners=corners,
        auto_template_frame=auto_template_frame,
        shuttle_model=shuttle_model,
        tracknet_v4_weights_path=TRACKNET_V4_WEIGHTS_PATH,
    )
    system.keep_audio = keep_audio
    system.process_video(progress_callback=progress_cb)

    # Event post-processing runs after detections.jsonl is closed.  It adds
    # explainable hit/rally analytics to metadata and writes the full report as
    # a separate artifact consumed by both API and React UI.
    from badminton_analysis.events import generate_match_report

    report_path = os.path.join(output_dir, "match_report.json")
    generate_match_report(
        detections_path=system.detections_path,
        metadata_path=system.metadata_path,
        output_path=report_path,
        language=language,
    )

    if visualize_positions:
        if language == "en":
            from badminton_analysis.visualization.player_positions_en import analyze_player_positions
        else:
            from badminton_analysis.visualization.player_positions_zh import analyze_player_positions
        vis_dir = os.path.join(output_dir, "position_visualizations")
        analyze_player_positions(system.detections_path, vis_dir, fps=system.fps)

    web_video_path = _reencode_for_browser(system.output_video_path, output_dir)

    result = {
        "output_dir": output_dir,
        "video": web_video_path,
        "metadata": system.metadata_path,
        "detections": system.detections_path,
        "report": report_path,
        "visualizations": [],
    }

    vis_dir = os.path.join(output_dir, "position_visualizations")
    if os.path.isdir(vis_dir):
        for root, _dirs, files in os.walk(vis_dir):
            for fname in sorted(files):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    result["visualizations"].append(os.path.join(root, fname))

    return result
