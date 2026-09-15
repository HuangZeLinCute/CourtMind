# Temporal fusion v2

The API selection remains `ensemble`. Metadata identifies the algorithm as
`multi_hypothesis_kalman_v2`; existing single-model modes remain available.

Each frame contributes one coordinate from each TrackNet and up to five valid
YOLO boxes. All observations use original, unannotated video pixels. Candidate
coordinates are normalized to a 512-pixel diagonal. Groups require every pair
to be within 9 normalized pixels (approximately 39 pixels at 1920x1080).
Multiple boxes from the same model never count as independent votes.

The filter retains at most six competing [x, y, vx, vy] hypotheses. Each has
a Kalman covariance and accumulated, exponentially decayed evidence. Prediction
uses actual frame indices and frame rate. Association uses normalized innovation
(Mahalanobis distance), not a fixed pixel jump limit. Process uncertainty allows
acceleration, while new hypotheses allow reacquisition after large trajectory
changes. TrackNet-only agreement receives a smaller evidence bonus than
agreement between TrackNet and YOLO because TrackNet errors can be correlated.
Confidence has only a small influence: the models are not probability-calibrated,
and V3's existing 0.5 placeholder is treated as neutral.

Outputs are observed coordinates, not predicted coordinates or smoothed positions
that can lag behind a hit. An isolated one-frame, single-model observation is
unconfirmed; competing distinct hypotheses within a score margin of 0.2 cause
abstention. Prediction-only hypotheses may survive a short gap but emit no
visible ball. The filter resets on non-court frames or long frame gaps. The old
220/260-pixel trajectory gate is bypassed only for already-validated ensemble
points; ROI validation and trajectory drawing remain active.

`detections.jsonl` records `shuttlecock.fusion` with the algorithm, number of
candidates/hypotheses, contributing sources, evidence score/margin and decision
reason. Scores are not probabilities. These diagnostics support manual review
and later tuning against ground truth.

Memory is bounded by the candidate and hypothesis caps, in addition to existing
V3/V4 streaming buffers and output trajectories. There is no future-frame
smoothing or learned calibration. This method does not guarantee improved
accuracy on every video: static distractors detected consistently by multiple
models can still win, and abstention trades recall for fewer false detections.
The shipped V4 weights originated in the sibling `TrackNetV4/new_tennis` folder;
their accuracy on this project's badminton videos has not been established.

Validation must distinguish runtime success from accuracy. Synthetic regression
tests cover outliers, multiple YOLO boxes, ambiguity, short gaps, sharp turns,
scaling, invalid values, and reset. Quantitative accuracy requires held-out
badminton videos with per-frame visibility and pixel ground truth, reporting
precision, recall, localization error, and track breaks against each individual
model and the old fusion under the same evaluation tolerance.
