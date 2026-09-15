# -*- coding: utf-8 -*-
"""Build a compact playback timeline from frame-level detector output."""
from __future__ import annotations

import json
import math
from bisect import bisect_right
from pathlib import Path

from backend.services import job_service


def _finite_pair(value):
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        pair = [round(float(value[0]), 3), round(float(value[1]), 3)]
    except (TypeError, ValueError):
        return None
    return pair if all(math.isfinite(item) for item in pair) else None


def _rally_at(rallies: list[dict], time_sec: float) -> dict | None:
    return next(
        (item for item in rallies if float(item.get("start_sec", 0)) <= time_sec <= float(item.get("end_sec", 0))),
        None,
    )


def build_timeline(job_id: str, sample_hz: float = 10.0) -> dict:
    result = job_service.job_result(job_id)
    if not result or result.get("status") != "completed":
        raise FileNotFoundError(f"Completed analysis not found: {job_id}")
    output_dir = Path(result.get("output_dir") or "")
    detections_path = output_dir / "detections.jsonl"
    if not detections_path.is_file():
        raise FileNotFoundError("Detection timeline is unavailable.")

    report = result.get("report") or {}
    rallies = report.get("rallies") or []
    interval = 1.0 / max(1.0, min(float(sample_hz), 20.0))
    next_sample = 0.0
    points: list[dict] = []
    previous = {"upper": None, "lower": None}
    match_distance = {"upper": 0.0, "lower": 0.0}
    rally_distance: dict[tuple[int, str], float] = {}
    speed_sum = {"upper": 0.0, "lower": 0.0}
    speed_count = {"upper": 0, "lower": 0}
    speed_max = {"upper": 0.0, "lower": 0.0}
    rally_speed_sum: dict[tuple[int, str], float] = {}
    rally_speed_count: dict[tuple[int, str], int] = {}
    rally_speed_max: dict[tuple[int, str], float] = {}

    with detections_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                time_sec = float(record.get("time_sec", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            rally = _rally_at(rallies, time_sec)
            rally_id = int(rally["rally_id"]) if rally else None
            players_out = {}
            for side in ("upper", "lower"):
                player = (record.get("players") or {}).get(side) or {}
                court = _finite_pair(player.get("court"))
                previous_item = previous[side]
                if court and previous_item:
                    old_court, old_time = previous_item
                    dt = max(0.0, time_sec - old_time)
                    step = math.dist(court, old_court)
                    # Ignore identity jumps / broken homographies rather than
                    # inflating movement totals shown to the user.
                    if dt > 0 and step <= 12.0 * dt + 0.15:
                        match_distance[side] += step
                        if rally_id is not None:
                            key = (rally_id, side)
                            rally_distance[key] = rally_distance.get(key, 0.0) + step
                if court:
                    previous[side] = (court, time_sec)
                try:
                    speed = max(0.0, float(player.get("speed") or 0.0))
                except (TypeError, ValueError):
                    speed = 0.0
                speed_sum[side] += speed
                speed_count[side] += 1
                speed_max[side] = max(speed_max[side], speed)
                if rally_id is not None:
                    key = (rally_id, side)
                    rally_speed_sum[key] = rally_speed_sum.get(key, 0.0) + speed
                    rally_speed_count[key] = rally_speed_count.get(key, 0) + 1
                    rally_speed_max[key] = max(rally_speed_max.get(key, 0.0), speed)
                rally_key = (rally_id, side)
                players_out[side] = {
                    "court": court,
                    "speed_mps": round(speed, 2),
                    "match_distance_m": round(match_distance[side], 2),
                    "rally_distance_m": round(rally_distance.get((rally_id, side), 0.0), 2) if rally_id else 0.0,
                    "match_average_speed_mps": round(speed_sum[side] / speed_count[side], 2),
                    "match_max_speed_mps": round(speed_max[side], 2),
                    "rally_average_speed_mps": round(
                        rally_speed_sum.get(rally_key, 0.0) / max(rally_speed_count.get(rally_key, 0), 1), 2
                    ) if rally_id else 0.0,
                    "rally_max_speed_mps": round(rally_speed_max.get(rally_key, 0.0), 2) if rally_id else 0.0,
                }

            if time_sec + 1e-6 < next_sample:
                continue
            next_sample = time_sec + interval
            hits = (rally or {}).get("hits") or []
            hit_times = [float(hit.get("time_sec", 0)) for hit in hits]
            shuttle = record.get("shuttlecock") or {}
            fusion = shuttle.get("fusion") or {}
            points.append({
                "frame": int(record.get("frame") or 0),
                "time_sec": round(time_sec, 3),
                "rally_id": rally_id,
                "rally_hit_count": bisect_right(hit_times, time_sec),
                "rally_total_hits": len(hits),
                "players": players_out,
                "shuttlecock": {
                    "image": _finite_pair(shuttle.get("image")),
                    "visible": _finite_pair(shuttle.get("image")) is not None,
                    "sources": list(fusion.get("sources") or []),
                },
            })

    video = (result.get("metadata") or {}).get("video") or {}
    return {
        "job_id": job_id,
        "sample_hz": sample_hz,
        "duration_sec": video.get("duration_sec"),
        "total_frames": video.get("total_frames"),
        "points": points,
    }
