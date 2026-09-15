"""Run the actual three-model analysis and verify its output contract.

Uses a fresh directory to avoid cached tracks and overwriting user analyses.
This is a smoke test, not an accuracy benchmark against labeled ground truth.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies


def main():
    output = Path(tempfile.mkdtemp(prefix="gb-fusion-v2-"))
    clip = output / "clip.mp4"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i",
        str(ROOT / "uploads/bb948abdcd47/input.mp4"), "-frames:v", "48", "-an", str(clip),
    ], check=True)
    load_runtime_dependencies()
    system = BadmintonAnalysisSystem(
        str(clip), show_display=False, output_dir=str(output / "result"),
        court_corners=[(437, 271), (846, 270), (980, 646), (306, 646)],
        shuttle_model="ensemble", pose_family="yolo-pose",
        yolo_pose_model=str(ROOT / "weights/yolo11n-pose.pt"),
        ball_model_path=str(ROOT / "weights/yolo11s-ball.pt"),
        tracknet_tracker_path=str(ROOT / "weights/tracknet/tracknet_v3_tracker.pt"),
        tracknet_rectifier_path=str(ROOT / "weights/tracknet/tracknet_v3_rectifier.pt"),
        tracknet_v4_weights_path=str(ROOT / "weights/tracknet_v4/tracknet_v4_type_b.keras"),
        show_performance_stats=False,
    )
    system.keep_audio = False
    system.process_video()
    metadata = json.loads(Path(system.metadata_path).read_text(encoding="utf-8"))
    assert metadata["models"]["shuttle_fusion"] == "multi_hypothesis_kalman_v2"
    records = [json.loads(line) for line in Path(system.detections_path).read_text(encoding="utf-8").splitlines()]
    assert len(records) == 48
    assert [row["frame"] for row in records] == list(range(1, 49))
    reasons = {}
    for record in records:
        fusion = record["shuttlecock"]["fusion"]
        assert fusion["algorithm"] == "multi_hypothesis_kalman_v2"
        assert fusion["hypotheses"] <= 6
        if fusion["reason"] != "observed":
            assert record["shuttlecock"]["image"] is None
        reasons[fusion["reason"]] = reasons.get(fusion["reason"], 0) + 1
    assert Path(system.output_video_path).stat().st_size > 0
    print(json.dumps({"output": str(output), "frames": len(records), "decisions": reasons,
                      "accuracy_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
