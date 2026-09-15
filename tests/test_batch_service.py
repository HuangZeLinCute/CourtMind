"""Batch scheduling tests; no video models are loaded."""

import unittest
from unittest.mock import patch

from backend.services import batch_service, job_service


class ImmediateThread:
    def __init__(self, target, args=(), **_kwargs):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class BatchServiceTests(unittest.TestCase):
    def setUp(self):
        with job_service._lock:
            job_service._jobs.clear()
        with batch_service._lock:
            batch_service._batches.clear()

    def test_batch_runs_videos_sequentially(self):
        analyzed = []

        def finish(job_id, video_id, corners, options, **_kwargs):
            analyzed.append((video_id, corners, options["shuttle_model"]))
            job_service.complete(job_id, {"output_dir": job_id})

        with (
            patch.object(batch_service.threading, "Thread", ImmediateThread),
            patch.object(
                batch_service.video_service,
                "get_info",
                side_effect=lambda video_id: {"filename": f"{video_id}.mp4"},
            ),
            patch.object(batch_service.court_service, "load_court_state", return_value={}),
            patch.object(
                batch_service.court_service,
                "detect_court",
                side_effect=lambda _video_id: {
                    "success": True,
                    "corners": [[0, 0], [10, 0], [10, 10], [0, 10]],
                },
            ),
            patch.object(batch_service.analysis_service, "run_analysis_job", side_effect=finish),
        ):
            result = batch_service.create(["video-a", "video-b"], {"shuttle_model": "ensemble"})

        self.assertEqual([item[0] for item in analyzed], ["video-a", "video-b"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["progress"], 100)

    def test_one_court_failure_does_not_stop_remaining_videos(self):
        analyzed = []

        def detect(video_id):
            if video_id == "bad":
                return {"success": False, "message": "court missing"}
            return {"success": True, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}

        def finish(job_id, video_id, **_kwargs):
            analyzed.append(video_id)
            job_service.complete(job_id, {})

        with (
            patch.object(batch_service.threading, "Thread", ImmediateThread),
            patch.object(
                batch_service.video_service,
                "get_info",
                side_effect=lambda video_id: {"filename": f"{video_id}.mp4"},
            ),
            patch.object(batch_service.court_service, "load_court_state", return_value={}),
            patch.object(batch_service.court_service, "detect_court", side_effect=detect),
            patch.object(batch_service.analysis_service, "run_analysis_job", side_effect=finish),
        ):
            result = batch_service.create(["bad", "good"], {"shuttle_model": "tracknet"})

        self.assertEqual(analyzed, ["good"])
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["failed"], 1)


if __name__ == "__main__":
    unittest.main()

