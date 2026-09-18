# -*- coding: utf-8 -*-
"""Resolve grounded AI frame references and create shareable still images."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

from backend.services import job_service

_FRAME_MARKER = re.compile(r"\[\[FRAME:(\d+(?:\.\d+)?)\|([^\]\r\n]{1,160})\]\]")
_VISUAL_TERMS = (
    "移动", "负荷", "站位", "回位", "步法", "下一步", "怎么打", "如何打", "如何改进",
    "这个环节", "这一拍", "关键帧", "截图", "画面", "进攻", "防守", "表现", "节奏",
    "训练", "策略", "建议", "落点", "相持", "回合", "movement", "position", "footwork",
    "attack", "defense", "performance", "rally", "training", "strategy", "recover", "next step",
    "frame", "screenshot",
)


def needs_frame(message: str) -> bool:
    lowered = message.lower()
    return any(term in lowered for term in _VISUAL_TERMS)


def detection_states(output_dir: Path, frames: set[int]) -> dict[int, dict]:
    """Read only requested frames, keeping LLM context bounded."""
    path = output_dir / "detections.jsonl"
    found: dict[int, dict] = {}
    if not path.is_file() or not frames:
        return found
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                frame = int(item.get("frame") or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if frame not in frames:
                continue
            players = item.get("players") or {}
            found[frame] = {
                "upper": {key: (players.get("upper") or {}).get(key) for key in ("court", "image", "speed")},
                "lower": {key: (players.get("lower") or {}).get(key) for key in ("court", "image", "speed")},
                "shuttlecock_image": (item.get("shuttlecock") or {}).get("image"),
            }
            if len(found) == len(frames):
                break
    return found


def _extract_frame(job_id: str, time_sec: float) -> tuple[str, int]:
    result = job_service.job_result(job_id) or {}
    output_dir = Path(result.get("output_dir") or "")
    metadata = result.get("metadata") or {}
    original = Path((metadata.get("video") or {}).get("path") or "")
    video_path = original if original.is_file() else output_dir / Path(result.get("video_url") or "").name
    if not video_path.is_file():
        raise FileNotFoundError("Source video is unavailable for frame capture.")

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_number = max(0, int(round(time_sec * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Unable to read the requested video frame.")
    finally:
        capture.release()

    target_dir = output_dir / "chat_frames"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"frame_{frame_number:08d}.jpg"
    if not target.is_file() and not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError("Unable to save the requested video frame.")
    return f"/outputs/{job_id}/chat_frames/{target.name}", frame_number


def _select_evidence_hit(job_id: str, valid_hits: list[dict], report: dict) -> tuple[dict | None, dict | None]:
    result = job_service.job_result(job_id) or {}
    output_dir = Path(result.get("output_dir") or "")
    states = detection_states(output_dir, {int(hit.get("frame") or 0) for hit in valid_hits})
    players = report.get("players") or {}
    target = max(
        ("upper", "lower"),
        key=lambda side: float((players.get(side) or {}).get("distance_m") or 0),
    )
    candidates = []
    for hit in valid_hits:
        state = states.get(int(hit.get("frame") or 0))
        if not state:
            continue
        try:
            target_speed = float((state.get(target) or {}).get("speed") or 0)
            other_speed = float((state.get("lower" if target == "upper" else "upper") or {}).get("speed") or 0)
        except (TypeError, ValueError):
            target_speed = other_speed = 0.0
        candidates.append((target_speed + other_speed * 0.15, hit, state))
    if not candidates:
        return (valid_hits[len(valid_hits) // 2], None) if valid_hits else (None, None)
    _, hit, state = max(candidates, key=lambda item: item[0])
    return hit, state


def _movement_guidance(hit: dict, state: dict | None, report: dict,
                       language: str = "zh") -> tuple[str, str]:
    """Build the frame caption plus a tactical note in the reply language."""
    time_sec = float(hit.get("time_sec") or 0)
    player = hit.get("player")
    players = report.get("players") or {}
    upper_distance = float((players.get("upper") or {}).get("distance_m") or 0)
    lower_distance = float((players.get("lower") or {}).get("distance_m") or 0)
    # Chinese only when the reply language is known to be Chinese; any other
    # (or unknown) language gets the English copy rather than stray Chinese.
    zh = language == "zh"

    if zh:
        heavier = "上方" if upper_distance >= lower_distance else "下方"
        caption = f"{time_sec:.2f} 秒击球时刻；用于复盘{heavier}球员的站位与回位路线"
    else:
        heavier = "upper" if upper_distance >= lower_distance else "lower"
        caption = (
            f"Contact at {time_sec:.2f}s; review the {heavier} player's "
            "positioning and recovery path"
        )
    if not state:
        return caption, (
            "该帧用于定位真实击球时刻；当前坐标数据不足，不能判断具体横移方向。"
            if zh else
            "This frame locates a real contact; the coordinate data is not sufficient to judge a sideways direction."
        )

    details = []
    advice = []
    for side in ("upper", "lower"):
        item = state.get(side) or {}
        court = item.get("court")
        speed = item.get("speed")
        if not isinstance(court, list) or len(court) < 2:
            continue
        x, y = float(court[0]), float(court[1])
        label = ("上方" if side == "upper" else "下方") if zh else side
        if zh:
            details.append(f"{label}约在 ({x:.2f}, {y:.2f})m、速度 {float(speed or 0):.2f}m/s")
        else:
            details.append(f"{label} around ({x:.2f}, {y:.2f}) m at {float(speed or 0):.2f} m/s")
        if x < 2.65:
            direction = "向右侧中路回收" if zh else "recover toward the middle on the right side"
        elif x > 3.45:
            direction = "向左侧中路回收" if zh else "recover toward the middle on the left side"
        else:
            direction = "保持中路并做分腿垫步" if zh else "hold the middle and split-step"
        if zh:
            role = "击球后" if side == player else "准备下一拍时"
            advice.append(f"{label}球员{role}建议先{direction}，再根据来球方向启动")
        else:
            role = "after the shot" if side == player else "while preparing for the next shot"
            advice.append(f"the {label} player should {direction} {role}, then react to the next shuttle")

    if zh:
        evidence = "；".join(details) if details else "当前帧未获得完整球员坐标"
        guidance = (
            f"\n\n**画面复盘（战术推断）**\n"
            f"已附上 {time_sec:.2f} 秒的真实击球帧。检测数据：{evidence}。"
            f"{'；'.join(advice)}。场地坐标只能支持站位与移动方向建议，不能据此判断具体挥拍动作。"
        )
    else:
        evidence = "; ".join(details) if details else "no complete player coordinates for this frame"
        guidance = (
            f"\n\n**Frame review (tactical inference)**\n"
            f"A real contact frame at {time_sec:.2f}s is attached. Detected data: {evidence}. "
            f"{'; '.join(advice)}. Court coordinates only support positioning and movement advice; "
            f"they cannot establish the actual swing."
        )
    return caption, guidance


def resolve_frame_markers(job_id: str, answer: str, user_message: str = "",
                          language: str = "zh") -> tuple[str, list[dict]]:
    """Turn at most one model marker into a validated hit-frame attachment."""
    result = job_service.job_result(job_id) or {}
    report = result.get("report") or {}
    valid_hits = [
        hit for rally in (report.get("rallies") or [])
        for hit in (rally.get("hits") or [])
    ]
    attachments = []
    fallback_guidance = ""
    match = _FRAME_MARKER.search(answer)
    selected_hit = None
    selected_caption = ""
    if match and valid_hits:
        requested = float(match.group(1))
        nearest = min(valid_hits, key=lambda hit: abs(float(hit.get("time_sec", 0)) - requested))
        hit_time = float(nearest.get("time_sec", 0))
        # Never permit an arbitrary timestamp invented by the model.
        if abs(hit_time - requested) <= 0.2:
            selected_hit = nearest
            selected_caption = match.group(2).strip()
    if selected_hit is None and needs_frame(user_message) and valid_hits:
        selected_hit, state = _select_evidence_hit(job_id, valid_hits, report)
        if selected_hit:
            selected_caption, fallback_guidance = _movement_guidance(
                selected_hit, state, report, language
            )
    if selected_hit:
        hit_time = float(selected_hit.get("time_sec", 0))
        try:
            image_url, actual_frame = _extract_frame(job_id, hit_time)
            attachments.append({
                "image_url": image_url,
                "frame": int(selected_hit.get("frame") or actual_frame),
                "time_sec": hit_time,
                "caption": selected_caption,
                "player": selected_hit.get("player"),
                "confidence": selected_hit.get("confidence"),
            })
        except (FileNotFoundError, RuntimeError):
            fallback_guidance = ""
    clean = _FRAME_MARKER.sub("", answer)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip() + fallback_guidance
    return clean, attachments
