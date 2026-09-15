"""TrackNetV4 adapter for Good-Badminton's per-frame trajectory contract.

The upstream TensorFlow model consumes three RGB frames as nine channels and
returns three heatmaps.  Inference is deliberately loaded lazily so the
TensorFlow runtime is not imported for YOLO or TrackNetV3 analyses.
"""

import os
import time
from typing import List

import cv2
import numpy as np

from .base import ShuttleDetection, ShuttleDetector

TRACKNET_V4_HEIGHT = 288
TRACKNET_V4_WIDTH = 512


def _decode_heatmap(heatmap, width, height, threshold):
    binary = (np.asarray(heatmap) > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ShuttleDetection(0.0, 0.0, False, 0.0)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    confidence = float(np.max(heatmap))
    return ShuttleDetection(
        x=(x + w / 2.0) * width / TRACKNET_V4_WIDTH,
        y=(y + h / 2.0) * height / TRACKNET_V4_HEIGHT,
        visible=True,
        confidence=confidence,
    )


class TrackNetV4Detector(ShuttleDetector):
    """TensorFlow TrackNetV4 Type-B detector (three frames in, three out)."""

    name = "tracknet_v4"

    def __init__(self, weights_path: str, threshold: float = 0.5, batch_size: int = 1):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"TrackNetV4 checkpoint not found: {weights_path}")

        # Keras 3 can execute this TensorFlow-authored architecture on the
        # project's existing PyTorch runtime, avoiding a second ML stack.
        os.environ.setdefault("KERAS_BACKEND", "torch")
        try:
            import keras
        except ImportError as exc:
            raise RuntimeError(
                "TrackNetV4 requires Keras 3. Install project requirements "
                "and restart the backend."
            ) from exc

        from ..models.tracknet_v4 import TrackNetV4

        self.threshold = float(threshold)
        self.batch_size = max(1, int(batch_size))
        self.weights_path = weights_path
        self._keras = keras

        print(f"[TrackNetV4] Loading Type-B model... ({weights_path})")
        started = time.time()
        self.model = TrackNetV4(TRACKNET_V4_HEIGHT, TRACKNET_V4_WIDTH, "TypeB")
        # The provided .keras checkpoint contains the same convolutional
        # backbone.  Loading with skip_mismatch also tolerates the upstream
        # checkpoint's older serialized name for the stateless fusion layer.
        self.model.load_weights(weights_path, skip_mismatch=True)
        print(f"[TrackNetV4] Model loaded in {time.time() - started:.2f}s")

    def process_video(self, video_path: str, progress_cb=None) -> List[ShuttleDetection]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"No decodable frames in video: {video_path}")

        print(
            f"[TrackNetV4] Streaming {expected_total or '?'} frames "
            f"(batch_size={self.batch_size})..."
        )
        started = time.time()
        result: List[ShuttleDetection] = []
        pending_groups = []
        pending_lengths = []

        def flush():
            if not pending_groups:
                return
            batch = np.stack(pending_groups)
            predictions = self._keras.ops.convert_to_numpy(self.model(batch, training=False))
            if predictions.ndim != 4 or predictions.shape[1:] != (
                3,
                TRACKNET_V4_HEIGHT,
                TRACKNET_V4_WIDTH,
            ):
                raise RuntimeError(
                    "Unexpected TrackNetV4 output shape: "
                    f"{tuple(predictions.shape)}; expected (batch, 3, 288, 512)"
                )
            for heatmaps, valid in zip(predictions, pending_lengths):
                for heatmap in heatmaps[:valid]:
                    result.append(_decode_heatmap(heatmap, width, height, self.threshold))
            pending_groups.clear()
            pending_lengths.clear()
            if progress_cb is not None:
                progress_cb(len(result), expected_total or len(result))

        try:
            while True:
                group = []
                for _ in range(3):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    group.append(frame)
                if not group:
                    break
                valid = len(group)
                while len(group) < 3:
                    group.append(group[-1])
                pending_groups.append(self._prepare_group(group))
                pending_lengths.append(valid)
                if len(pending_groups) >= self.batch_size:
                    flush()
                if valid < 3:
                    break
            flush()
        finally:
            cap.release()

        total = len(result)
        if not result:
            raise RuntimeError(f"No decodable frames in video: {video_path}")

        print(
            f"[TrackNetV4] Tracking completed in {time.time() - started:.2f}s "
            f"({sum(item.visible for item in result)} / {total} visible)."
        )
        # CAP_PROP_FRAME_COUNT can be imprecise; the decoded frame count is the
        # authoritative alignment used by the downstream analysis loop.
        if expected_total > 0 and expected_total != total:
            print(
                f"[TrackNetV4] Container reported {expected_total} frames; "
                f"decoded {total}."
            )
        return result

    @staticmethod
    def _prepare_group(frames):
        channels = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (TRACKNET_V4_WIDTH, TRACKNET_V4_HEIGHT))
            channels.extend(resized.transpose(2, 0, 1))
        return np.asarray(channels, dtype=np.float32) / 255.0

    def close(self) -> None:
        self.model = None
        try:
            import keras

            keras.backend.clear_session()
        except ImportError:
            pass
