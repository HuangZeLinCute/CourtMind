"""TrackNetV3Detector — temporal shuttlecock trajectory tracking.

Vendored, self-contained port of the BadmintonTrackNet inference pipeline
(``tracknet.inference.pipeline``) adapted to the V3 checkpoints shipped in
``BadmintonTrackNet/ckpts`` (new ``model_state_dict`` format):

* Tracker: U-Net, input ``(L+1)*3`` channels (L RGB frames + median background),
  output ``L`` heatmaps.  Frames are processed in sliding windows, each
  position's heatmap is decoded to an original-resolution coordinate and
  window predictions are combined by real frame id with edge re-normalized
  weights (``weight`` ensemble mode).
* Rectifier (InpaintNet, optional): 1D U-Net consuming
  ``[x_norm, y_norm, inpaint_mask, visibility]`` and emitting refined
  normalized coordinates, repairing masked gaps in the raw trajectory.

Coordinates returned are always in original video pixels, indexed by 0-based
video frame id.  The model is loaded exactly once per detector instance.
"""

import os
import time
from collections import defaultdict, deque
from typing import List, Optional

import cv2
import numpy as np
import torch

from .base import ShuttleDetection, ShuttleDetector
from ..models.tracknet_v3 import (
    TRACKNET_HEIGHT,
    TRACKNET_WIDTH,
    load_tracknet_v3_rectifier,
    load_tracknet_v3_tracker,
)

_DELTA_T = 1.0 / np.sqrt(TRACKNET_HEIGHT**2 + TRACKNET_WIDTH**2)
_COOR_TH = _DELTA_T * 50


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if name not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {name}")
    return torch.device(name)


def _iter_windows(frame_ids, frames, seq_len, step, pad=False):
    """Yield (ids, frames, valid_length) sliding windows (copy of the
    BadmintonTrackNet ``iter_windows`` helper)."""
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    frames = np.asarray(frames)
    if len(frame_ids) != len(frames):
        raise ValueError("frame_ids and frames must have the same length")
    for start in range(0, len(frame_ids), step):
        end = min(start + seq_len, len(frame_ids))
        valid_length = end - start
        if valid_length < seq_len and not pad:
            break
        window_ids = frame_ids[start:end]
        window_frames = frames[start:end]
        if valid_length < seq_len:
            count = seq_len - valid_length
            window_ids = np.concatenate((window_ids, np.repeat(window_ids[-1], count)))
            window_frames = np.concatenate(
                (window_frames, np.repeat(window_frames[-1][None], count, axis=0)), axis=0
            )
        yield window_ids.copy(), window_frames.copy(), valid_length
        if end == len(frame_ids):
            break


def _decode_heatmap(heatmap, width, height, threshold=0.5):
    """Largest connected region centre -> original video coordinates."""
    binary = (heatmap > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x + w / 2) * width / TRACKNET_WIDTH, (y + h / 2) * height / TRACKNET_HEIGHT


class TrackNetV3Detector(ShuttleDetector):
    """Temporal TrackNetV3 shuttlecock tracker with optional rectifier."""

    name = "tracknet"

    def __init__(
        self,
        tracker_path: str,
        rectifier_path: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 4,
        threshold: float = 0.5,
        max_sample_num: int = 1800,
        rectifier_in_channels: int = 4,
    ):
        if not os.path.isfile(tracker_path):
            raise FileNotFoundError(f"TrackNetV3 tracker checkpoint not found: {tracker_path}")
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        self.max_sample_num = int(max_sample_num)
        self.rectifier_in_channels = int(rectifier_in_channels)
        self.rectifier_path = rectifier_path

        print(f"[TrackNetV3] Device: {self.device}")
        print(f"[TrackNetV3] Loading tracker... ({tracker_path})")
        t0 = time.time()
        self.tracker = load_tracknet_v3_tracker(tracker_path, device=str(self.device))
        self.seq_len = self.tracker.seq_len
        print(f"[TrackNetV3] Tracker loaded in {time.time() - t0:.2f}s (seq_len={self.seq_len})")

        self.rectifier = None
        self.rectifier_seq_len = 16
        if rectifier_path and os.path.isfile(rectifier_path):
            print(f"[TrackNetV3] Loading rectifier... ({rectifier_path})")
            t0 = time.time()
            try:
                self.rectifier = load_tracknet_v3_rectifier(rectifier_path, device=str(self.device))
                self.rectifier_seq_len = 16
                print(f"[TrackNetV3] Rectifier loaded in {time.time() - t0:.2f}s")
            except Exception as exc:  # noqa: BLE001
                self.rectifier = None
                print(f"[TrackNetV3] Rectifier load failed ({exc}); running tracker-only.")
        elif rectifier_path:
            print(f"[TrackNetV3] Rectifier checkpoint missing ({rectifier_path}); tracker-only.")

    # ------------------------------------------------------------------ API

    def process_video(self, video_path: str, progress_cb=None) -> List[ShuttleDetection]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if width <= 0 or height <= 0 or fps <= 0 or total <= 0:
            raise RuntimeError(f"Invalid video metadata: {video_path}")
        print(f"[TrackNetV3] Video: {width}x{height} @ {fps:.2f} FPS, {total} frames")

        sample_count = min(31, self.max_sample_num, total)
        print(f"[TrackNetV3] Computing resized background median over {sample_count} frames...")
        median = self._compute_background(video_path, total, sample_count)

        print(f"[TrackNetV3] Streaming {total} frames (batch_size={self.batch_size})...")
        t0 = time.time()
        trajectory = self._track_stream(
            video_path, width, height, median, progress_cb, total
        )
        print(f"[TrackNetV3] Tracking completed in {time.time() - t0:.2f}s.")

        if self.rectifier is not None:
            print("[TrackNetV3] Running trajectory rectification...")
            trajectory = self._rectify(trajectory, width, height)

        visible = int(np.count_nonzero(trajectory["visible"]))
        print(f"[TrackNetV3] Final visible points: {visible} / {len(trajectory['visible'])}")

        result: List[ShuttleDetection] = []
        for i in range(len(trajectory["frame_ids"])):
            fid = int(trajectory["frame_ids"][i])
            result.append(
                ShuttleDetection(
                    x=float(trajectory["x"][i]),
                    y=float(trajectory["y"][i]),
                    visible=bool(trajectory["visible"][i]),
                    confidence=0.5,
                )
            )
        # Align to every video frame id (0-based), filling missing ids as invisible.
        by_id = {int(fid): det for fid, det in zip(trajectory["frame_ids"], result)}
        aligned = [by_id.get(i, ShuttleDetection(0.0, 0.0, False)) for i in range(total)]
        return aligned

    # ------------------------------------------------------------ internals

    @staticmethod
    def _resize_frame(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (TRACKNET_WIDTH, TRACKNET_HEIGHT)).transpose(2, 0, 1)

    def _compute_background(self, video_path, total, sample_count):
        """Approximate the median background without retaining full-res video frames."""
        cap = cv2.VideoCapture(video_path)
        samples = []
        try:
            for frame_id in np.linspace(0, max(0, total - 1), sample_count, dtype=np.int64):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
                ok, frame = cap.read()
                if ok:
                    samples.append(self._resize_frame(frame))
        finally:
            cap.release()
        if not samples:
            raise RuntimeError(f"Could not sample background frames from {video_path}")
        return np.median(np.stack(samples), axis=0).astype(np.uint8)

    def _stream_windows(self, video_path):
        """Yield resized sliding windows while holding at most ``seq_len`` frames."""
        cap = cv2.VideoCapture(video_path)
        frames = deque(maxlen=self.seq_len)
        frame_ids = deque(maxlen=self.seq_len)
        decoded = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(self._resize_frame(frame))
                frame_ids.append(decoded)
                decoded += 1
                if len(frames) == self.seq_len:
                    yield np.stack(frames), np.asarray(frame_ids, dtype=np.int64), self.seq_len

            if 0 < decoded < self.seq_len:
                valid_length = len(frames)
                padded_frames = list(frames)
                padded_ids = list(frame_ids)
                while len(padded_frames) < self.seq_len:
                    padded_frames.append(padded_frames[-1])
                    padded_ids.append(padded_ids[-1])
                yield (
                    np.stack(padded_frames),
                    np.asarray(padded_ids, dtype=np.int64),
                    valid_length,
                )
        finally:
            cap.release()

    @staticmethod
    def _prepare_resized_window(window_frames, median):
        stacked = np.concatenate((median[None], window_frames), axis=0)
        return stacked.reshape(-1, TRACKNET_HEIGHT, TRACKNET_WIDTH).astype(np.float32) / 255.0

    def _track_stream(self, video_path, width, height, median, progress_cb, total):
        """Batch-infer streaming windows and ensemble overlapping heatmaps."""
        seq_len = self.seq_len
        weights = np.minimum(
            np.arange(1, seq_len + 1), np.arange(seq_len, 0, -1)
        ).astype(np.float64)
        totals = {}
        total_weights = defaultdict(float)
        decoded = {}
        pending = []

        def flush_ready(before=None):
            ready = sorted(fid for fid in totals if before is None or fid < before)
            for fid in ready:
                decoded[fid] = _decode_heatmap(
                    totals.pop(fid) / total_weights.pop(fid),
                    width,
                    height,
                    self.threshold,
                )

        def infer_pending():
            if not pending:
                return
            inputs = torch.from_numpy(
                np.stack([
                    self._prepare_resized_window(window, median)
                    for window, _ids, _valid in pending
                ])
            ).float().to(self.device)
            with torch.inference_mode():
                outputs = self.tracker(inputs).detach().cpu().numpy()
            for (_window, ids, valid_length), predictions in zip(pending, outputs):
                flush_ready(int(ids[0]))
                for position in range(valid_length):
                    fid = int(ids[position])
                    totals[fid] = (
                        totals.get(fid, np.zeros_like(predictions[position]))
                        + predictions[position] * weights[position]
                    )
                    total_weights[fid] += weights[position]
            if progress_cb is not None:
                progress_cb(min(int(pending[-1][1][-1]) + 1, total), total)
            pending.clear()

        for window in self._stream_windows(video_path):
            pending.append(window)
            if len(pending) >= self.batch_size:
                infer_pending()
        infer_pending()
        flush_ready()

        if not decoded:
            raise RuntimeError("TrackNetV3 produced no decodable frames")
        ordered = np.asarray(sorted(decoded), dtype=np.int64)
        coords = np.asarray([decoded[fid] for fid in ordered])
        return {
            "frame_ids": ordered,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "visible": np.any(coords != 0, axis=1),
        }

    def _read_frames(self, video_path, total):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_ids = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                frame_ids.append(len(frame_ids))
        finally:
            cap.release()
        if not frames:
            raise RuntimeError(f"No frames could be read from {video_path}")
        return frames, np.asarray(frame_ids, dtype=np.int64)

    def _prepare_frames(self, window_frames, median):
        rgb_frames = window_frames[..., ::-1]  # BGR -> RGB
        rgb_median = median[..., ::-1]
        channels = []
        for frame in rgb_frames:
            resized = cv2.resize(frame, (TRACKNET_WIDTH, TRACKNET_HEIGHT)).transpose(2, 0, 1)
            channels.append(resized)
        background = cv2.resize(rgb_median.astype(np.uint8), (TRACKNET_WIDTH, TRACKNET_HEIGHT)).transpose(2, 0, 1)
        channels.insert(0, background)
        return np.concatenate(channels).astype(np.float32) / 255.0

    def _track(self, frames, frame_ids, width, height, fps, median, progress_cb, total):
        seq_len = self.seq_len
        step = 1
        pad = len(frames) < seq_len
        windows = list(_iter_windows(frame_ids, np.asarray(frames), seq_len, step, pad=pad))
        weights = np.minimum(np.arange(1, seq_len + 1), np.arange(seq_len, 0, -1)).astype(np.float64)

        totals = {}
        total_weights = defaultdict(float)
        decoded = {}

        def flush(before=None):
            ready = sorted(fid for fid in totals if before is None or fid < before)
            for fid in ready:
                decoded[fid] = _decode_heatmap(
                    totals.pop(fid) / total_weights.pop(fid), width, height, self.threshold
                )

        done = 0
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                batch = windows[start : start + self.batch_size]
                inputs = torch.from_numpy(
                    np.stack([self._prepare_frames(w[1], median) for w in batch])
                ).float().to(self.device)
                outputs = self.tracker(inputs).detach().cpu().numpy()
                for window, predictions in zip(batch, outputs):
                    flush(int(window[0][0]))
                    for position in range(window[2]):
                        fid = int(window[0][position])
                        totals[fid] = totals.get(fid, np.zeros_like(predictions[position])) + predictions[position] * weights[position]
                        total_weights[fid] += weights[position]
                done += len(batch)
                if progress_cb is not None:
                    progress_cb(min(done, total), total)
        flush()
        if not decoded:
            raise RuntimeError("TrackNetV3 produced no decodable frames")

        ordered = np.array(sorted(decoded), dtype=np.int64)
        coords = np.asarray([decoded[fid] for fid in ordered])
        trajectory = {
            "frame_ids": ordered,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "visible": np.any(coords != 0, axis=1),
        }
        return trajectory

    def _rectify(self, trajectory, width, height):
        ids = trajectory["frame_ids"]
        x = trajectory["x"].astype(np.float64)
        y = trajectory["y"].astype(np.float64)
        vis = trajectory["visible"].astype(np.float32)
        mask = self._inpaint_mask(ids, y, vis, height).astype(np.float32)
        coords = np.column_stack((x / width, y / height))

        seq_len = self.rectifier_seq_len
        step = 1
        pad = len(coords) < seq_len
        coord_windows = list(_iter_windows(ids, coords, seq_len, step, pad=pad))
        mask_windows = list(_iter_windows(ids, np.concatenate((mask[:, None], vis[:, None]), axis=1), seq_len, step, pad=pad))

        predictions = []
        with torch.inference_mode():
            for start in range(0, len(coord_windows), self.batch_size):
                cb = coord_windows[start : start + self.batch_size]
                mb = mask_windows[start : start + self.batch_size]
                cw = np.stack([w[1] for w in cb])
                mw = np.stack([w[1] for w in mb])
                # (B, L, 2) + (B, L, 2) -> (B, L, 4) -> (B, 4, L) for Conv1d
                inp = np.concatenate((cw, mw), axis=2).transpose(0, 2, 1)
                out = self.rectifier(torch.from_numpy(inp).float().to(self.device))
                predictions.extend(out.detach().cpu().numpy())

        ids2, combined = _ensemble_coords(
            np.stack([w[0] for w in coord_windows]),
            np.asarray(predictions),
            [w[2] for w in coord_windows],
        )
        combined[(combined[:, 0] < _COOR_TH) & (combined[:, 1] < _COOR_TH)] = 0.0
        pixels = combined * np.array([width, height])  # before clamping, for quality checks
        repaired_raw = {
            "frame_ids": ids2,
            "x": pixels[:, 0],
            "y": pixels[:, 1],
            "visible": np.any(pixels != 0, axis=1),
        }
        repaired = {
            "frame_ids": ids2,
            "x": np.clip(pixels[:, 0], 0, width),
            "y": np.clip(pixels[:, 1], 0, height),
            "visible": np.any(pixels != 0, axis=1),
        }
        if not self._rectify_quality_ok(trajectory, repaired, repaired_raw, width, height):
            print("[TrackNetV3] Rectification output rejected (unstable); keeping raw trajectory.")
            return trajectory
        print(f"[TrackNetV3] Rectification done: {int(np.count_nonzero(repaired['visible']))} visible")
        return repaired

    @staticmethod
    def _rectify_quality_ok(raw, repaired, repaired_raw, width, height):
        """Sanity gate: rectified trajectory must not jump wildly, drift out of
        bounds, or stick to frame edges relative to the raw one.  Falls back to
        the raw trajectory when the rectifier produces unstable output
        (possible input-format mismatch)."""
        def jump_p90(xs, ys, vis):
            pts = np.column_stack([np.asarray(xs), np.asarray(ys)])[np.asarray(vis, dtype=bool)]
            if len(pts) < 3:
                return float("inf")
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            return float(np.percentile(d, 90))

        def edge_ratio(xs, ys, vis):
            arr = np.column_stack([np.asarray(xs), np.asarray(ys)])[np.asarray(vis, dtype=bool)]
            if len(arr) == 0:
                return 1.0
            margin = 0.02
            on_edge = (
                (arr[:, 0] <= width * margin) | (arr[:, 0] >= width * (1 - margin))
                | (arr[:, 1] <= height * margin) | (arr[:, 1] >= height * (1 - margin))
            )
            return float(np.mean(on_edge))

        raw_vis = np.asarray(raw["visible"], dtype=bool)
        rp = jump_p90(repaired["x"], repaired["y"], repaired["visible"])
        rw = jump_p90(raw["x"], raw["y"], raw["visible"])
        if rp > rw * 2.0 + 40.0:
            return False
        # Out-of-bounds measured BEFORE clamping.
        oob = np.mean(
            (repaired_raw["x"] < 0) | (repaired_raw["x"] > width)
            | (repaired_raw["y"] < 0) | (repaired_raw["y"] > height)
        )
        if oob > 0.05:
            return False
        edge_repaired = edge_ratio(repaired["x"], repaired["y"], repaired["visible"])
        edge_raw = edge_ratio(raw["x"], raw["y"], raw["visible"])
        if edge_repaired > edge_raw + 0.2:
            return False
        repaired_vis = int(np.count_nonzero(repaired["visible"]))
        raw_vis_n = int(np.count_nonzero(raw_vis))
        if repaired_vis < raw_vis_n * 0.5:
            return False
        return True

    @staticmethod
    def _inpaint_mask(ids, y, vis, height):
        """Mark gaps between consecutive visible points for repair (same rule
        as the BadmintonTrackNet pipeline: interior of a gap, both endpoints
        below 5% of frame height)."""
        mask = np.zeros(len(ids), dtype=np.float32)
        visible = np.flatnonzero(vis.astype(bool))
        for left, right in zip(visible, visible[1:]):
            if right > left + 1 and y[left] > height * 0.05 and y[right] > height * 0.05:
                mask[left + 1 : right] = 1.0
        return mask


def _ensemble_coords(frame_ids, predictions, valid_lengths):
    """Combine per-window coordinate predictions (B, 2, L) by real frame id
    with renormalized edge weights.  ``predictions`` entries are (2, L):
    channel 0 = x, channel 1 = y, per window position."""
    seq_len = frame_ids.shape[1]
    weights = np.minimum(np.arange(1, seq_len + 1), np.arange(seq_len, 0, -1)).astype(np.float64)
    totals = {}
    total_weights = defaultdict(float)
    for ids, values, valid_length in zip(frame_ids, predictions, valid_lengths):
        for position in range(int(valid_length)):
            fid = int(ids[position])
            weight = weights[position]
            vec = np.asarray(values[:, position], dtype=np.float64)
            totals[fid] = totals.get(fid, np.zeros(2, dtype=np.float64)) + vec * weight
            total_weights[fid] += weight
    ordered = np.array(sorted(totals), dtype=np.int64)
    combined = np.stack([totals[fid] / total_weights[fid] for fid in ordered])
    return ordered, combined
