"""Multi-hypothesis association. Scores are evidence, not probabilities.

Coordinates use a 512-pixel diagonal, making gates resolution independent.
Predictions guide association; only observed positions are emitted.
"""
from dataclasses import dataclass
from itertools import combinations, product
import math

import numpy as np

from .base import ShuttleDetection

ALGORITHM = "multi_hypothesis_kalman_v2"


@dataclass
class Hypothesis:
    state: np.ndarray
    covariance: np.ndarray
    score: float
    hits: int
    misses: int


class TemporalEnsemble:
    """Bounded competing velocity tracks with maneuver uncertainty.

    TrackNet agreement is discounted relative to agreement with YOLO because
    their errors may be correlated. Same-source boxes cannot add votes.
    """

    def __init__(self, frame_size, fps=30.0, beam_size=6):
        self.width, self.height = frame_size
        self.scale = 512.0 / math.hypot(*frame_size)
        self.fps = max(1.0, float(fps))
        self.beam_size = beam_size
        self.reset()

    def reset(self):
        self.tracks = []
        self.last_frame = None
        self.diagnostics = {}

    def _measurements(self, detections):
        sources = {}
        for source, d in detections:
            if not d.visible or not np.isfinite([d.x, d.y]).all():
                continue
            if not (0 <= d.x < self.width and 0 <= d.y < self.height):
                continue
            confidence = float(d.confidence) if d.confidence is not None else 0.5
            if not math.isfinite(confidence):
                continue
            # V3 exports a placeholder score, not a calibrated probability.
            quality = 0.5 if source == "tracknet" else float(np.clip(confidence, 0, 1))
            sources.setdefault(source, []).append(
                (np.array([d.x, d.y]) * self.scale, quality))
        sources = {key: sorted(values, key=lambda x: -x[1])[:5]
                   for key, values in sources.items()}
        measurements = []
        keys = sorted(sources)
        for count in range(1, len(keys) + 1):
            for group in combinations(keys, count):
                for items in product(*(sources[k] for k in group)):
                    points = np.array([x[0] for x in items])
                    # Every pair must agree: about 39px at 1080p, not 176px.
                    if any(np.linalg.norm(a - b) > 9 for a, b in combinations(points, 2)):
                        continue
                    weights = np.array([0.75 + 0.25 * x[1] for x in items])
                    center = np.average(points, axis=0, weights=weights)
                    spread = np.average(np.sum((points - center) ** 2, axis=1), weights=weights)
                    quality = float(np.mean([x[1] for x in items]))
                    independent = "yolo" in group and count > 1
                    support = (1.3 if independent else 0.75) if count > 1 else 0
                    support += 0.35 if count == 3 else 0
                    evidence = 0.6 + 0.5 * quality + support - 0.015 * spread
                    variance = max(1.5, 5 - 2 * quality + spread / 2)
                    measurements.append((center, variance, evidence, group))
        return measurements

    def update(self, detections, frame_index):
        if self.last_frame is not None and frame_index <= self.last_frame:
            self.reset()
        gap = 1 if self.last_frame is None else frame_index - self.last_frame
        if gap > max(5, round(self.fps * 0.25)):
            self.reset()
            gap = 1
        self.last_frame = frame_index
        dt = gap * 30.0 / self.fps
        measurements = self._measurements(detections)
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = dt
        acceleration = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        candidates = []
        for track in self.tracks:
            predicted = transition @ track.state
            covariance = transition @ track.covariance @ transition.T
            covariance += acceleration @ acceleration.T * 36.0
            for index, (point, variance, evidence, group) in enumerate(measurements):
                innovation = point - predicted[:2]
                error = covariance[:2, :2] + np.eye(2) * variance
                distance = float(innovation @ np.linalg.solve(error, innovation))
                if distance > 25:
                    continue
                gain = np.linalg.solve(error, covariance[:2, :]).T
                state = predicted + gain @ innovation
                residual = np.eye(4)
                residual[:, :2] -= gain
                posterior = residual @ covariance @ residual.T + gain @ (np.eye(2) * variance) @ gain.T
                score = track.score * 0.85 ** gap + evidence - 0.12 * min(distance, 25)
                candidates.append((Hypothesis(state, posterior, score, track.hits + 1, 0), index))
            if track.misses + gap <= max(3, round(self.fps * 0.15)):
                candidates.append((Hypothesis(predicted, covariance,
                    track.score * 0.85 ** gap - 0.8 * gap, track.hits, track.misses + gap), None))
        for index, (point, variance, evidence, group) in enumerate(measurements):
            candidates.append((Hypothesis(np.r_[point, 0., 0.],
                np.diag([variance, variance, 225., 225.]), evidence - 0.45, 1, 0), index))
        candidates.sort(key=lambda x: -x[0].score)
        retained = []
        for candidate in candidates:
            track, _ = candidate
            if any(np.linalg.norm(track.state[:2] - old.state[:2]) < 4 and
                   np.linalg.norm(track.state[2:] - old.state[2:]) < 8 for old, _ in retained):
                continue
            retained.append(candidate)
            if len(retained) == self.beam_size:
                break
        self.tracks = [track for track, _ in retained]
        self.diagnostics = {"algorithm": ALGORITHM, "candidates": len(measurements),
                            "hypotheses": len(retained), "sources": [], "reason": "missing"}
        if not retained:
            return ShuttleDetection(0, 0, False, None)
        best, index = retained[0]
        if index is None:
            self.diagnostics["reason"] = "prediction_only"
            return ShuttleDetection(0, 0, False, None)
        point, _variance, _evidence, group = measurements[index]
        alternatives = [track.score for track, other in retained[1:]
                        if other is not None and np.linalg.norm(measurements[other][0] - point) > 9]
        margin = best.score - max(alternatives) if alternatives else None
        self.diagnostics.update(sources=list(group), score=float(best.score), margin=margin)
        if len(group) == 1 and best.hits < 2:
            self.diagnostics["reason"] = "unconfirmed_single_source"
            return ShuttleDetection(0, 0, False, None)
        if margin is not None and margin < 0.2:
            self.diagnostics["reason"] = "ambiguous"
            return ShuttleDetection(0, 0, False, None)
        self.diagnostics["reason"] = "observed"
        x, y = point / self.scale
        return ShuttleDetection(float(x), float(y), True, None)
