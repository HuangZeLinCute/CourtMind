# -*- coding: utf-8 -*-
"""Analysis job execution service.

Bridges the FastAPI job system to the headless analysis pipeline. Runs in a
background thread; the computer-vision pipeline itself is untouched.
"""
import json
import threading
import traceback
from pathlib import Path

from backend.core.config import DEFAULT_OPTIONS, OPTION_KEYS
from backend.core.paths import OUTPUTS_DIR, UPLOADS_DIR
from backend.services import job_service
from backend.services.video_service import get_video_path
from backend.services.analysis_pipeline import imread_safe, run_analysis

# --------------------------------------------------------------------------- #
# Option normalization
# --------------------------------------------------------------------------- #
def normalize_options(raw: dict | None) -> dict:
    options = dict(DEFAULT_OPTIONS)
    if raw:
        for key, value in raw.items():
            if key in OPTION_KEYS and value is not None:
                options[key] = value
    return options


# --------------------------------------------------------------------------- #
# Background worker
# --------------------------------------------------------------------------- #
def run_analysis_job(job_id: str, video_id: str, corners, options: dict,
                     legacy_template_id: str | None = None,
                     analysis_name: str | None = None) -> None:
    """Execute the analysis pipeline for a job (runs in a background thread)."""
    from backend.services.court_service import load_court_state  # noqa: PLC0415

    try:
        job_service.set_stage(job_id, "Preparation")
        video_path = get_video_path(video_id)

        prev = load_court_state(video_id)
        if corners is None:
            corners = prev.get("corners")
        if not corners or len(corners) != 4:
            raise ValueError(
                "Court corners are required — run court detection and confirm first."
            )
        corners = [[int(x), int(y)] for x, y in corners]

        auto_template_frame = None
        template_path = None
        if legacy_template_id:
            template_path = UPLOADS_DIR / legacy_template_id / "template.png"
            if not template_path.is_file():
                raise FileNotFoundError("Legacy court template not found.")
        else:
            best_frame_path = prev.get("best_frame_path")
            if best_frame_path and Path(best_frame_path).is_file():
                auto_template_frame = imread_safe(best_frame_path)

        job_service.set_stage(job_id, "Analysis")

        def progress_cb(frame_count: int, total_frames: int) -> None:
            job_service.update_progress(job_id, frame_count, total_frames)

        output_dir = OUTPUTS_DIR / job_id
        result = run_analysis(
            video_path=str(video_path),
            template_path=template_path,
            corners=corners,
            options=options,
            progress_cb=progress_cb,
            auto_template_frame=auto_template_frame,
            output_dir=str(output_dir),
        )

        metadata_path = Path(result.get("metadata") or output_dir / "metadata.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            job = job_service.get(job_id) or {}
            metadata["analysis"] = {
                "name": job.get("name") or analysis_name or video_path.stem,
                "job_id": job_id,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        job_service.complete(job_id, result)
    except Exception as exc:  # noqa: BLE001 - job must capture and record the error
        job_service.fail(job_id, exc, traceback.format_exc())


def spawn_job(job_id: str, video_id: str, corners, options: dict,
              legacy_template_id: str | None = None,
              analysis_name: str | None = None) -> None:
    """Start the analysis in a daemon thread (non-blocking)."""
    thread = threading.Thread(
        target=run_analysis_job,
        args=(job_id, video_id, corners, options, legacy_template_id, analysis_name),
        daemon=True,
        name=f"analysis-{job_id}",
    )
    thread.start()
