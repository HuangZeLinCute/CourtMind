"""Explainable rally segmentation, stroke counting, and match reporting.

The detector deliberately combines three signals instead of treating every
trajectory turn as a stroke: shuttle velocity impulse, direction change, and
proximity to a tracked player's hands/body.  All distances are normalized by
the video diagonal, so thresholds are resolution independent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from badminton_analysis.data.writer import write_json


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return np.asarray([x, y], dtype=np.float64)


def _load_records(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record.get("frame"), int):
                records.append(record)
    records.sort(key=lambda item: item["frame"])
    return records


def _trajectory_segments(records: list[dict], fps: float):
    observed = []
    for record in records:
        position = _point((record.get("shuttlecock") or {}).get("image"))
        if position is not None:
            observed.append((int(record["frame"]), position))
    if not observed:
        return [], 0

    max_bridge = max(2, int(round(fps * 0.22)))
    raw_segments, current = [], [observed[0]]
    for sample in observed[1:]:
        if sample[0] - current[-1][0] > max_bridge:
            raw_segments.append(current)
            current = []
        current.append(sample)
    raw_segments.append(current)

    segments = []
    for raw in raw_segments:
        if len(raw) < 5:
            continue
        source_frames = np.asarray([item[0] for item in raw], dtype=np.int32)
        source_points = np.asarray([item[1] for item in raw], dtype=np.float64)
        frames = np.arange(source_frames[0], source_frames[-1] + 1, dtype=np.int32)
        points = np.column_stack([
            np.interp(frames, source_frames, source_points[:, axis]) for axis in (0, 1)
        ])
        if len(points) >= 5:
            kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float64) / 9.0
            for axis in (0, 1):
                points[:, axis] = np.convolve(
                    np.pad(points[:, axis], (2, 2), mode="edge"), kernel, mode="valid"
                )
        segments.append((frames, points))
    return segments, len(observed)


def _nearest_player(record: dict, shuttle: np.ndarray, diagonal: float):
    best = None
    for side in ("upper", "lower"):
        player = ((record.get("players") or {}).get(side) or {})
        candidates = []
        hands = player.get("hands") or {}
        for hand in (hands.get("left"), hands.get("right")):
            point = _point(hand)
            if point is not None:
                candidates.append((point, True))
        body = _point(player.get("image"))
        if body is not None:
            candidates.append((body, False))
        for point, is_hand in candidates:
            distance = float(np.linalg.norm(shuttle - point) / diagonal)
            # Body centroids are farther from racket contact than hand keypoints.
            scale = 0.085 if is_hand else 0.13
            proximity = math.exp(-((distance / scale) ** 2))
            if not is_hand:
                proximity *= 0.68
            candidate = (proximity, distance, side, "hand" if is_hand else "body")
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def _detect_hits(records: list[dict], fps: float, width: int, height: int) -> list[dict]:
    by_frame = {int(record["frame"]): record for record in records}
    segments, _ = _trajectory_segments(records, fps)
    diagonal = max(1.0, math.hypot(width, height))
    window = max(1, int(round(fps * 0.08)))
    candidates = []

    for frames, points in segments:
        if len(frames) < window * 2 + 1:
            continue
        for index in range(window, len(frames) - window):
            dt_before = (frames[index] - frames[index - window]) / fps
            dt_after = (frames[index + window] - frames[index]) / fps
            if dt_before <= 0 or dt_after <= 0:
                continue
            before = (points[index] - points[index - window]) / dt_before / diagonal
            after = (points[index + window] - points[index]) / dt_after / diagonal
            before_speed = float(np.linalg.norm(before))
            after_speed = float(np.linalg.norm(after))
            impulse = float(np.linalg.norm(after - before))
            cosine = float(np.dot(before, after) / max(1e-6, before_speed * after_speed))
            direction_change = float(np.clip((1.0 - cosine) / 2.0, 0.0, 1.0))
            speed_change = abs(after_speed - before_speed) / max(0.08, before_speed, after_speed)
            kinematic = (
                0.65 * float(np.clip(impulse / 2.0, 0.0, 1.0))
                + 0.20 * direction_change
                + 0.15 * float(np.clip(speed_change, 0.0, 1.0))
            )

            frame = int(frames[index])
            player = _nearest_player(by_frame.get(frame, {}), points[index], diagonal)
            if player is None:
                continue
            proximity, distance, side, contact_source = player
            score = 0.62 * kinematic + 0.38 * proximity
            if kinematic < 0.16 or proximity < 0.055 or score < 0.34:
                continue
            candidates.append({
                "frame": frame,
                "time_sec": round(frame / fps, 3),
                "player": side,
                "position": [round(float(v), 2) for v in points[index]],
                "confidence": round(float(np.clip(score, 0.0, 0.99)), 3),
                "contact_source": contact_source,
                "hand_distance_norm": round(distance, 4),
                "kinematic_score": round(kinematic, 3),
            })

    # Time-domain non-maximum suppression prevents one contact from being
    # counted on several adjacent frames.
    radius = max(2, int(round(fps * 0.28)))
    accepted = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if all(abs(candidate["frame"] - hit["frame"]) > radius for hit in accepted):
            accepted.append(candidate)
    accepted.sort(key=lambda item: item["frame"])
    # Consecutive contacts by the same side inside a very short interval are
    # normally multiple peaks from one swing. Keep only the stronger peak.
    deduplicated = []
    same_side_window = max(2, int(round(fps * 0.70)))
    for hit in accepted:
        if (
            deduplicated
            and hit["player"] == deduplicated[-1]["player"]
            and hit["frame"] - deduplicated[-1]["frame"] <= same_side_window
        ):
            if hit["confidence"] > deduplicated[-1]["confidence"]:
                deduplicated[-1] = hit
        else:
            deduplicated.append(hit)
    return deduplicated


def _movement(records: list[dict], start: int, end: int, fps: float) -> dict:
    result = {}
    for side in ("upper", "lower"):
        samples = []
        for record in records:
            frame = int(record["frame"])
            if start <= frame <= end:
                court = _point((((record.get("players") or {}).get(side) or {}).get("court")))
                if court is not None:
                    samples.append((frame, court))
        distance = 0.0
        speeds = []
        for (f1, p1), (f2, p2) in zip(samples, samples[1:]):
            dt = (f2 - f1) / fps
            step = float(np.linalg.norm(p2 - p1))
            if dt > 0 and step / dt <= 8.5:
                distance += step
                speeds.append(step / dt)
        result[side] = {
            "distance_m": round(distance, 2),
            "average_speed_mps": round(mean(speeds), 2) if speeds else 0.0,
            "max_speed_mps": round(max(speeds), 2) if speeds else 0.0,
        }
    return result


def _segment_rallies(hits: list[dict], records: list[dict], fps: float) -> list[dict]:
    if not hits:
        return []
    max_gap = max(1, int(round(fps * 3.2)))
    groups, current = [], [hits[0]]
    for hit in hits[1:]:
        if hit["frame"] - current[-1]["frame"] > max_gap:
            groups.append(current)
            current = []
        current.append(hit)
    groups.append(current)

    minimum_frame = min(int(record["frame"]) for record in records)
    maximum_frame = max(int(record["frame"]) for record in records)
    rallies = []
    for rally_id, group in enumerate(groups, 1):
        # A weak isolated candidate is reported as an event but not promoted to
        # a rally, reducing false rally counts in low-quality footage.
        if len(group) == 1 and group[0]["confidence"] < 0.52:
            continue
        start = max(minimum_frame, group[0]["frame"] - int(round(0.6 * fps)))
        end = min(maximum_frame, group[-1]["frame"] + int(round(0.8 * fps)))
        per_player = {
            side: sum(hit["player"] == side for hit in group) for side in ("upper", "lower")
        }
        intervals = [
            (right["frame"] - left["frame"]) / fps for left, right in zip(group, group[1:])
        ]
        confidence = mean(hit["confidence"] for hit in group)
        rallies.append({
            "rally_id": rally_id,
            "start_frame": start,
            "end_frame": end,
            "start_sec": round(start / fps, 3),
            "end_sec": round(end / fps, 3),
            "duration_sec": round((end - start) / fps, 3),
            "hit_count": len(group),
            "hits_by_player": per_player,
            "average_hit_interval_sec": round(mean(intervals), 3) if intervals else None,
            "confidence": round(confidence, 3),
            "intensity": "high" if len(group) >= 10 else "medium" if len(group) >= 5 else "low",
            "movement": _movement(records, start, end, fps),
            "hits": group,
        })
    return rallies


def analyze_match(detections_path: str | Path, metadata: dict) -> dict:
    records = _load_records(detections_path)
    video = metadata.get("video", {})
    fps = float(video.get("fps") or 30.0)
    width = int(video.get("width") or 1920)
    height = int(video.get("height") or 1080)
    total_frames = int(video.get("total_frames") or (records[-1]["frame"] if records else 0))
    hits = _detect_hits(records, fps, width, height)
    rallies = _segment_rallies(hits, records, fps)
    confirmed_hits = [hit for rally in rallies for hit in rally["hits"]]
    observed_points = sum(
        _point((record.get("shuttlecock") or {}).get("image")) is not None for record in records
    )
    hit_counts = {
        side: sum(hit["player"] == side for hit in confirmed_hits) for side in ("upper", "lower")
    }
    if records:
        match_movement = _movement(
            records, int(records[0]["frame"]), int(records[-1]["frame"]), fps
        )
    else:
        match_movement = _movement([], 0, 0, fps)
    players = {
        side: {
            "hit_count": hit_counts[side],
            "hit_share": round(hit_counts[side] / max(1, len(confirmed_hits)), 3),
            **match_movement[side],
        }
        for side in ("upper", "lower")
    }
    rally_hit_counts = [rally["hit_count"] for rally in rallies]
    active_time = sum(rally["duration_sec"] for rally in rallies)
    longest = max(rallies, key=lambda rally: rally["hit_count"], default=None)
    return {
        "schema_version": "1.0",
        "algorithm": "kinematic_proximity_rally_v1",
        "summary": {
            "rally_count": len(rallies),
            "total_hits": len(confirmed_hits),
            "hits_by_player": hit_counts,
            "average_hits_per_rally": round(mean(rally_hit_counts), 2) if rallies else 0.0,
            "median_hits_per_rally": round(median(rally_hit_counts), 2) if rallies else 0.0,
            "longest_rally_id": longest["rally_id"] if longest else None,
            "longest_rally_hits": longest["hit_count"] if longest else 0,
            "active_rally_time_sec": round(active_time, 2),
            "hits_per_active_minute": round(len(confirmed_hits) / active_time * 60, 1) if active_time else 0.0,
        },
        "data_quality": {
            "processed_records": len(records),
            "video_total_frames": total_frames,
            "shuttle_observed_frames": observed_points,
            "shuttle_coverage": round(observed_points / max(1, total_frames), 3),
            "average_hit_confidence": round(mean(hit["confidence"] for hit in confirmed_hits), 3) if confirmed_hits else 0.0,
        },
        "players": players,
        "rallies": rallies,
    }


def _narrative(analysis: dict, language: str) -> tuple[str, list[dict]]:
    summary = analysis["summary"]
    quality = analysis["data_quality"]
    players = analysis["players"]
    rallies = analysis["rallies"]
    zh = language != "en"
    insights = []

    if not rallies:
        overview = (
            "当前轨迹中没有达到联合置信门槛的完整回合。建议使用三模型融合并确认球场画面稳定。"
            if zh else
            "No complete rally passed the combined confidence threshold. Use the three-model ensemble and verify a stable court view."
        )
    else:
        overview = (
            f"识别到 {summary['rally_count']} 个回合、{summary['total_hits']} 次可信击球，"
            f"平均每回合 {summary['average_hits_per_rally']} 拍。"
            if zh else
            f"Detected {summary['rally_count']} rallies and {summary['total_hits']} reliable hits, averaging {summary['average_hits_per_rally']} shots per rally."
        )

        avg_hits = summary["average_hits_per_rally"]
        if avg_hits >= 8:
            message = "多拍相持能力较强；训练重点可转向相持后的主动变速和终结。" if zh else "Sustained exchanges are strong; focus training on changing pace and finishing after long exchanges."
        elif avg_hits >= 4:
            message = "回合持续性中等；建议加强前三拍衔接以及由防守转主动的稳定性。" if zh else "Rally persistence is moderate; improve the first three shots and the transition from defense to attack."
        else:
            message = "回合普遍较短；优先训练发接发质量、回球稳定性和减少非受迫中断。" if zh else "Rallies are generally short; prioritize serve/return quality, consistency, and fewer unforced interruptions."
        insights.append({"kind": "action", "title": "训练重点" if zh else "Training priority", "message": message})

        upper = summary["hits_by_player"]["upper"]
        lower = summary["hits_by_player"]["lower"]
        total = max(1, upper + lower)
        imbalance = abs(upper - lower) / total
        if imbalance > 0.25:
            more = "上方球员" if upper > lower else "下方球员"
            if not zh:
                more = "upper player" if upper > lower else "lower player"
            message = (
                f"{more}承担了更多可识别击球（{max(upper, lower)}/{total}）；可结合关键回合检查是否存在持续受压或检测遮挡。"
                if zh else
                f"The {more} accounts for more detected contacts ({max(upper, lower)}/{total}); review key rallies for sustained pressure or occlusion bias."
            )
            insights.append({"kind": "balance", "title": "参与度差异" if zh else "Participation balance", "message": message})

        longest = max(rallies, key=lambda rally: rally["hit_count"])
        insights.append({
            "kind": "key_moment",
            "title": "关键回合" if zh else "Key rally",
            "message": (
                f"第 {longest['rally_id']} 回合最长，共 {longest['hit_count']} 拍，发生在 {longest['start_sec']:.1f}–{longest['end_sec']:.1f} 秒。"
                if zh else
                f"Rally {longest['rally_id']} is the longest at {longest['hit_count']} shots, from {longest['start_sec']:.1f}s to {longest['end_sec']:.1f}s."
            ),
            "rally_id": longest["rally_id"],
        })

        upper_distance = players["upper"]["distance_m"]
        lower_distance = players["lower"]["distance_m"]
        if max(upper_distance, lower_distance) > 0:
            heavier = "上方球员" if upper_distance > lower_distance else "下方球员"
            if not zh:
                heavier = "upper player" if upper_distance > lower_distance else "lower player"
            insights.append({
                "kind": "workload",
                "title": "移动负荷" if zh else "Movement load",
                "message": (
                    f"{heavier}的有效移动距离更高（上方 {upper_distance:.1f}m / 下方 {lower_distance:.1f}m），可优先复盘其被调动路线与回位效率。"
                    if zh else
                    f"The {heavier} covered more valid distance (upper {upper_distance:.1f}m / lower {lower_distance:.1f}m); review movement pressure and recovery efficiency."
                ),
            })

    if quality["shuttle_coverage"] < 0.45:
        insights.append({
            "kind": "quality",
            "title": "数据质量提醒" if zh else "Data quality note",
            "message": (
                f"羽毛球轨迹覆盖率为 {quality['shuttle_coverage']:.0%}，击球与回合统计可能偏低。"
                if zh else
                f"Shuttle trajectory coverage is {quality['shuttle_coverage']:.0%}; hit and rally totals may be under-counted."
            ),
        })
    return overview, insights


def generate_match_report(
    detections_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    language: str = "zh",
) -> dict:
    metadata_file = Path(metadata_path)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    analysis = analyze_match(detections_path, metadata)
    overview, insights = _narrative(analysis, language)
    report = {
        **analysis,
        "title": "AI 羽毛球比赛报告" if language != "en" else "AI Badminton Match Report",
        "overview": overview,
        "insights": insights,
        "limitations": (
            "击球与回合来自轨迹和人体关键点的自动估计，不代表正式裁判比分；遮挡、镜头切换和漏检会使统计偏低。"
            if language != "en" else
            "Hits and rallies are estimated from trajectory and pose signals, not official scoring. Occlusion, cuts, and missed detections can lower counts."
        ),
    }
    write_json(str(output_path), report)

    metadata["analytics"] = {
        "algorithm": report["algorithm"],
        **report["summary"],
        "data_quality": report["data_quality"],
    }
    metadata.setdefault("outputs", {})["match_report"] = str(Path(output_path))
    write_json(str(metadata_file), metadata)
    return report
