# -*- coding: utf-8 -*-
"""CourtKeyNet detector integration for Good-Badminton.

Replaces the manual "upload court template image" flow with automatic court
corner prediction from video frames using the CourtKeyNet model:

    video -> sample frames -> CourtKeyNet -> 4 corners (TL, TR, BR, BL)
            -> geometric sanity check -> median fusion -> Homography (existing)

The CourtKeyNet code is vendored inside this project at
``badminton_analysis/court/courtkeynet/`` and the weights live at
``weights/courtkeynet_finetuned.safetensors`` — no external path is required.
Set the ``COURTKEYNET_*`` environment variables to override.

Design notes
------------
* The CourtKeyNet model is loaded exactly ONCE per :class:`CourtKeyNetDetector`
  instance (``model.eval()`` + ``torch.inference_mode()`` inside predict).
* Device selection: CUDA if available, otherwise CPU.
* Preprocessing mirrors CourtKeyNet's official ``inference.py``:
  resize (640, 640) -> BGR2RGB -> float / 255 -> (1, 3, 640, 640).
* Corner order is verified against CourtKeyNet's trained order
  ``[TL, TR, BR, BL]`` — identical to what ``CourtMapper`` expects, so no
  reordering is performed.
* No GUI (no cv2.imshow / namedWindow / tkinter) and no subprocess: prediction
  is a plain ``outputs = model(tensor)`` call.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("goodbadminton.courtkeynet")

# --------------------------------------------------------------------------- #
# Unified configuration: environment variables override the defaults.
#   COURTKEYNET_ROOT     -> the vendored courtkeynet subpackage directory
#   COURTKEYNET_WEIGHTS  -> safetensors weight file
#   COURTKEYNET_CONFIG   -> model config yaml
# --------------------------------------------------------------------------- #
_PACKAGE_DIR = Path(__file__).resolve().parent            # badminton_analysis/court/
_PROJECT_ROOT = _PACKAGE_DIR.parents[1]                   # Good-Badminton/
_VENDORED_ROOT = _PACKAGE_DIR / "courtkeynet"             # vendored subpackage

COURTKEYNET_ROOT = Path(
    os.environ.get("COURTKEYNET_ROOT", str(_VENDORED_ROOT))
)
_PROJECT_WEIGHTS = _PROJECT_ROOT / "weights" / "courtkeynet_finetuned.safetensors"
_ROOT_WEIGHTS = COURTKEYNET_ROOT / "weights" / "courtkeynet_finetuned.safetensors"
COURTKEYNET_WEIGHTS = Path(
    os.environ.get(
        "COURTKEYNET_WEIGHTS",
        str(_PROJECT_WEIGHTS if _PROJECT_WEIGHTS.is_file() else _ROOT_WEIGHTS),
    )
)
COURTKEYNET_CONFIG = Path(
    os.environ.get(
        "COURTKEYNET_CONFIG",
        str(COURTKEYNET_ROOT / "configs" / "courtkeynet.yaml"),
    )
)

_IMGSZ = 640  # CourtKeyNet inference resolution (matches official inference.py)


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class CourtKeyNetDetector:
    """Loads CourtKeyNet once and predicts the 4 court corners per frame."""

    def __init__(self, weights_path=None, root=None, config_path=None, device=None):
        self.root = Path(root) if root else COURTKEYNET_ROOT
        self.weights_path = Path(weights_path) if weights_path else COURTKEYNET_WEIGHTS
        self.config_path = Path(config_path) if config_path else COURTKEYNET_CONFIG

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"CourtKeyNet weights not found: {self.weights_path}. "
                f"Expected at {_PROJECT_ROOT / 'weights' / 'courtkeynet_finetuned.safetensors'}. "
                f"Set COURTKEYNET_WEIGHTS or pass weights_path."
            )
        self.device = device or self._auto_device()

        logger.info("[CourtKeyNet] Loading model...")
        self.model, self.config = self._load_model()
        self.model.eval()
        logger.info("[CourtKeyNet] Device: %s", self.device)
        logger.info("[CourtKeyNet] Model loaded from %s", self.weights_path)

    @staticmethod
    def _auto_device():
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load_model(self):
        import torch
        import yaml

        # Import the vendored model (badminton_analysis/court/courtkeynet)
        from .courtkeynet import CourtKeyNet, load_weights  # noqa: PLC0415

        ckpt = load_weights(str(self.weights_path), device=self.device)

        if isinstance(ckpt, dict) and ckpt.get("config"):
            config = ckpt["config"]
        else:
            if not self.config_path.is_file():
                raise FileNotFoundError(
                    f"CourtKeyNet config not found: {self.config_path}"
                )
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        model = CourtKeyNet(config)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("[CourtKeyNet] Missing state keys: %s", missing)
        if unexpected:
            logger.warning("[CourtKeyNet] Unexpected state keys: %s", unexpected[:5])
        return model.to(self.device), config

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Predict the 4 court corners of one BGR frame.

        Args:
            frame_bgr: BGR image (any size).

        Returns:
            ``np.ndarray`` of shape (4, 2) with float32 pixel coordinates in the
            order ``[TL, TR, BR, BL]`` (same order CourtMapper expects).
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr must be a numpy array")
        import torch

        h, w = frame_bgr.shape[:2]
        # Preprocessing identical to CourtKeyNet inference.py
        img = cv2.resize(frame_bgr, (_IMGSZ, _IMGSZ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self.device == "cuda" and torch.cuda.is_available():
                with torch.autocast(device_type="cuda"):
                    outputs = self.model(tensor)
            else:
                outputs = self.model(tensor)

        kpts = outputs["kpts_refined"][0].cpu().numpy().astype(np.float32)  # (4, 2) normalized
        kpts_px = np.empty_like(kpts)
        kpts_px[:, 0] = kpts[:, 0] * w
        kpts_px[:, 1] = kpts[:, 1] * h
        return kpts_px


# --------------------------------------------------------------------------- #
# Geometric sanity checks (basic only — no camera / color / position heuristics)
# --------------------------------------------------------------------------- #
def validate_court_corners(corners, frame_width, frame_height):
    """Basic geometric validity of a predicted court quad.

    Returns ``(valid, reasons)``. Checks: 4 finite points, inside image bounds,
    edges non-zero, area not tiny, convex (no self-intersection),
    TL/TR/BR/BL order consistent with the quad centre.
    """
    reasons = []
    try:
        pts = np.asarray(corners, dtype=np.float32)
    except Exception:
        return False, ["unable to parse corners"]
    if pts.ndim != 2 or pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return False, [f"expected (4, 2) finite points, got {getattr(pts, 'shape', '?')}"]

    w, h = int(frame_width), int(frame_height)
    margin = 4.0
    if (
        np.any(pts[:, 0] < -margin)
        or np.any(pts[:, 0] > w - 1 + margin)
        or np.any(pts[:, 1] < -margin)
        or np.any(pts[:, 1] > h - 1 + margin)
    ):
        reasons.append("points outside image bounds")

    min_dim = max(5.0, min(w, h) * 0.02)
    edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    if min(edges) < min_dim:
        reasons.append(f"edge too short ({min(edges):.1f}px)")

    area = abs(float(cv2.contourArea(pts)))
    if area < (w * h) * 0.01:
        reasons.append(f"area too small ({area:.0f}px^2)")

    if not cv2.isContourConvex(pts.astype(np.int32)):
        reasons.append("quadrilateral is not convex / self-intersecting")

    center = pts.mean(axis=0)
    order_ok = (
        pts[0, 0] < center[0] and pts[0, 1] < center[1]
        and pts[1, 0] > center[0] and pts[1, 1] < center[1]
        and pts[2, 0] > center[0] and pts[2, 1] > center[1]
        and pts[3, 0] < center[0] and pts[3, 1] > center[1]
    )
    if not order_ok:
        reasons.append("corner order is not TL/TR/BR/BL")

    return len(reasons) == 0, reasons


# --------------------------------------------------------------------------- #
# Frame sampling + multi-frame median fusion
# --------------------------------------------------------------------------- #
def _sample_timestamps(duration_sec, start_sec=0.5, end_sec=4.0, n=8):
    """Uniformly sample ~n timestamps inside [start_sec, end_sec] (clamped)."""
    end_sec = min(end_sec, max(start_sec, duration_sec - 0.25))
    if end_sec <= start_sec or duration_sec <= 0:
        return [max(0.0, start_sec)]
    step = (end_sec - start_sec) / max(1, n - 1)
    return [round(start_sec + i * step, 3) for i in range(n)]


def _read_frame_at(video_path, timestamp_sec):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def detect_court_corners_from_video(video_path, detector=None, logger=None):
    """Predict court corners from a video without per-frame inference.

    Strategy:
        1. sample ~8 frames from the first 0.5-4.0 s of the video;
        2. run CourtKeyNet once per sampled frame;
        3. drop geometrically invalid predictions;
        4. reject multi-frame outliers, fuse the rest with the median.

    Returns ``(corners, info)`` where *corners* is ``np.ndarray (4,2) float32``
    in original-video pixel coordinates (or None if detection failed) and
    *info* carries diagnostics plus the best representative frame.
    """
    log = logger or globals()["logger"]
    if detector is None:
        detector = CourtKeyNetDetector()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total / fps if fps > 0 and total > 0 else 0.0

    timestamps = _sample_timestamps(duration)
    log.info("[CourtKeyNet] Sampling %d frames...", len(timestamps))

    predictions = []  # [(ts, corners_px)]
    cap = cv2.VideoCapture(str(video_path))
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        try:
            corners = detector.predict(frame)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill detection
            log.warning("[CourtKeyNet] Frame %.2fs prediction failed: %s", ts, exc)
            continue
        valid, reasons = validate_court_corners(corners, frame_w, frame_h)
        if valid:
            predictions.append((ts, corners))
        else:
            log.info("[CourtKeyNet] Frame %.2fs invalid: %s", ts, "; ".join(reasons))
    cap.release()

    info = {
        "detector": "courtkeynet",
        "sampled_timestamps": timestamps,
        "valid_timestamps": [ts for ts, _ in predictions],
        "valid_count": len(predictions),
        "kept_frames": len(predictions),
        "best_frame": None,
        "best_timestamp": None,
        "message": "",
    }
    if not predictions:
        info["message"] = "all sampled frames invalid"
        log.warning("[CourtKeyNet] All sampled frames invalid.")
        return None, info

    # Multi-frame fusion: median with outlier rejection
    arr = np.stack([c for _, c in predictions])  # (N, 4, 2)
    median0 = np.median(arr, axis=0)
    diag = float(np.hypot(frame_w, frame_h))
    dists = np.linalg.norm(arr - median0, axis=(1, 2))
    keep = dists <= 0.12 * diag
    if int(keep.sum()) >= 2:
        arr = arr[keep]
        info["kept_frames"] = int(keep.sum())
    final = np.median(arr, axis=0).astype(np.float32)

    # Representative frame: the sampled frame whose corners are closest to the fused result
    best_ts = min(predictions, key=lambda p: float(np.linalg.norm(p[1] - final)))[0]
    best_frame = _read_frame_at(video_path, best_ts)
    info["best_frame"] = best_frame
    info["best_timestamp"] = best_ts

    log.info("[CourtKeyNet] Valid predictions: %d/%d", info["valid_count"], len(timestamps))
    log.info("[CourtKeyNet] Court corners detected successfully.")
    return final, info


# --------------------------------------------------------------------------- #
# High-level entry: CourtKeyNet -> original detector -> manual
# --------------------------------------------------------------------------- #
def resolve_court_from_video(video_path, use_fallback=True, detector=None):
    """Resolve court corners from a video.

    Priority:
        1. CourtKeyNet (+ geometric validation + median fusion);
        2. original Good-Badminton CV detector on the best sampled frame;
        3. None (caller may then ask the user for manual corners).

    Returns ``(corners, roi_corners, mid_height, best_frame, detector_name)``.
    """
    log = logger
    best_frame = None
    try:
        corners, info = detect_court_corners_from_video(video_path, detector=detector)
        best_frame = info.get("best_frame") if info else None
    except Exception as exc:  # noqa: BLE001
        log.warning("[CourtKeyNet] Detection failed: %s", exc)
        corners = None

    if corners is not None and best_frame is not None:
        from .mapper import CourtMapper, compute_expanded_roi  # noqa: PLC0415

        h, w = best_frame.shape[:2]
        roi_corners = compute_expanded_roi(corners, (h, w, 3))
        mid_height = CourtMapper(corners).mid_height
        log.info("[Court] Using CourtKeyNet detector.")
        return corners, roi_corners, mid_height, best_frame, "courtkeynet"

    if not use_fallback:
        return None, None, None, best_frame, None

    log.warning("[CourtKeyNet] Detection failed.")
    log.info("[Court] Falling back to original detector.")
    if best_frame is None:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        best_frame = _read_frame_at(video_path, max(0.5, total / 60.0)) if total > 0 else None
    if best_frame is None:
        return None, None, None, None, None

    from .mapper import auto_detect_preview, resolve_court_corners  # noqa: PLC0415

    try:
        corners_auto, _preview = auto_detect_preview(best_frame)
        if corners_auto:
            corners, roi_corners, mid_height = resolve_court_corners(
                best_frame, manual_corners=corners_auto
            )
            if corners:
                log.info("[Court] Using original detector (fallback).")
                return corners, roi_corners, mid_height, best_frame, "original"
    except Exception as exc:  # noqa: BLE001
        log.warning("[Court] Original detector failed: %s", exc)

    log.warning("[Court] Original detector also failed. Manual annotation required.")
    return None, None, None, best_frame, None
