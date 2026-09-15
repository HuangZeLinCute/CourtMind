"""Legacy v1 spatial fusion, retained for comparison and old callers.

The production ensemble route uses TemporalEnsemble in temporal_ensemble.py.
"""

from math import hypot
from typing import Iterable, Sequence

from .base import ShuttleDetection


MODEL_RELIABILITY = {
    "tracknet_v4": 1.0,
    "tracknet": 0.85,
    "yolo": 0.75,
}


def fuse_detections(
    detections: Iterable[tuple[str, ShuttleDetection]],
    frame_size: tuple[int, int],
    history: Sequence[tuple[float, float]] = (),
) -> ShuttleDetection:
    """Fuse visible detections using local consensus and motion continuity.

    Confidence values from heatmap and bounding-box models are not calibrated
    against each other, so they only modulate a stable per-model reliability
    weight. A distant outlier is never averaged into an agreeing cluster.
    """

    width, height = frame_size
    candidates = []
    for source, detection in detections:
        if not detection.visible:
            continue
        if not (0.0 <= detection.x < width and 0.0 <= detection.y < height):
            continue
        confidence = 0.5 if detection.confidence is None else detection.confidence
        confidence = min(1.0, max(0.0, float(confidence)))
        reliability = MODEL_RELIABILITY.get(source, 0.7)
        weight = reliability * (0.5 + 0.5 * confidence)
        candidates.append((source, detection, weight))

    if not candidates:
        return ShuttleDetection(0.0, 0.0, False, None)
    if len(candidates) == 1:
        return candidates[0][1]

    agreement_gate = max(35.0, hypot(width, height) * 0.08)

    clusters = []
    for _, seed, _ in candidates:
        cluster = [
            item
            for item in candidates
            if hypot(item[1].x - seed.x, item[1].y - seed.y) <= agreement_gate
        ]
        clusters.append(cluster)

    best_cluster = max(clusters, key=lambda items: (len(items), sum(x[2] for x in items)))
    if len(best_cluster) >= 2:
        total_weight = sum(item[2] for item in best_cluster)
        x = sum(item[1].x * item[2] for item in best_cluster) / total_weight
        y = sum(item[1].y * item[2] for item in best_cluster) / total_weight
        confidence = min(1.0, total_weight / len(best_cluster))
        return ShuttleDetection(x, y, True, confidence)

    # No models agree spatially. Prefer the candidate closest to the projected
    # next position; on the first frame fall back to model reliability.
    if history:
        if len(history) >= 2:
            px = history[-1][0] + (history[-1][0] - history[-2][0])
            py = history[-1][1] + (history[-1][1] - history[-2][1])
        else:
            px, py = history[-1]
        chosen = min(
            candidates,
            key=lambda item: hypot(item[1].x - px, item[1].y - py) - 40.0 * item[2],
        )
    else:
        chosen = max(candidates, key=lambda item: item[2])
    return chosen[1]
