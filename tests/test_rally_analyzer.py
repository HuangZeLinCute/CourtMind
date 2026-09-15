"""Synthetic behavioral tests for hit/rally analysis and report output."""

import json
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.events import analyze_match, generate_match_report


def synthetic_exchange(scale=1.0):
    contacts = [(1, 250, 250), (30, 500, 100), (60, 500, 400), (90, 500, 100), (120, 500, 400)]
    records = []
    for frame in range(1, 121):
        for (f1, x1, y1), (f2, x2, y2) in zip(contacts, contacts[1:]):
            if f1 <= frame <= f2:
                ratio = (frame - f1) / (f2 - f1)
                x = x1 + (x2 - x1) * ratio
                y = y1 + (y2 - y1) * ratio
                break
        records.append({
            "frame": frame,
            "time_sec": frame / 30,
            "players": {
                "upper": {
                    "image": [500 * scale, 150 * scale],
                    "court": [3.05, 3.0],
                    "hands": {"left": [500 * scale, 100 * scale], "right": None},
                },
                "lower": {
                    "image": [500 * scale, 350 * scale],
                    "court": [3.05, 10.4],
                    "hands": {"left": [500 * scale, 400 * scale], "right": None},
                },
            },
            "shuttlecock": {"image": [x * scale, y * scale]},
        })
    return records


class RallyAnalyzerTests(unittest.TestCase):
    def _analyze(self, scale=1.0):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in synthetic_exchange(scale)),
                encoding="utf-8",
            )
            metadata = {"video": {"fps": 30, "width": int(1000 * scale), "height": int(500 * scale), "total_frames": 120}}
            return analyze_match(path, metadata)

    def test_alternating_contacts_form_one_rally(self):
        result = self._analyze()
        self.assertEqual(result["summary"]["rally_count"], 1)
        self.assertGreaterEqual(result["summary"]["total_hits"], 3)
        players = [hit["player"] for hit in result["rallies"][0]["hits"]]
        self.assertTrue(all(left != right for left, right in zip(players, players[1:])))

    def test_detection_is_resolution_independent(self):
        normal = self._analyze(1.0)["summary"]
        doubled = self._analyze(2.0)["summary"]
        self.assertEqual(normal["rally_count"], doubled["rally_count"])
        self.assertEqual(normal["total_hits"], doubled["total_hits"])

    def test_long_pause_splits_two_rallies(self):
        first = synthetic_exchange()
        second = synthetic_exchange()
        for record in second:
            record["frame"] += 160
            record["time_sec"] = record["frame"] / 30
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in first + second),
                encoding="utf-8",
            )
            result = analyze_match(
                path,
                {"video": {"fps": 30, "width": 1000, "height": 500, "total_frames": 280}},
            )
        self.assertEqual(result["summary"]["rally_count"], 2)

    def test_report_is_written_and_metadata_enriched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detections = root / "detections.jsonl"
            metadata = root / "metadata.json"
            report_path = root / "match_report.json"
            detections.write_text(
                "\n".join(json.dumps(record) for record in synthetic_exchange()),
                encoding="utf-8",
            )
            metadata.write_text(
                json.dumps({"video": {"fps": 30, "width": 1000, "height": 500, "total_frames": 120}}),
                encoding="utf-8",
            )
            report = generate_match_report(detections, metadata, report_path, language="zh")
            enriched = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertTrue(report_path.is_file())
            self.assertIn("AI", report["title"])
            self.assertEqual(enriched["analytics"]["total_hits"], report["summary"]["total_hits"])


if __name__ == "__main__":
    unittest.main()
