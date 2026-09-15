# -*- coding: utf-8 -*-
"""End-to-end API test for the FastAPI backend.

Run:  python test_e2e_api.py
Covers: upload -> detect-court -> PUT court -> analyze -> poll -> results.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api"
VIDEO = r"C:\Users\14181\Desktop\GraduationProject\Good-Badminton\videos\demo.mp4"


def req(method, path, data=None, headers=None, timeout=600):
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers=headers or {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body) if body else None


def main():
    print("== 1. upload ==")
    with open(VIDEO, "rb") as f:
        content = f.read()
    boundary = "----e2e"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="demo.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    status, info = req("POST", "/videos", body,
                       {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert status == 201, (status, info)
    vid = info["video_id"]
    print(f"  video_id={vid} fps={info['fps']} {info['width']}x{info['height']} "
          f"duration={info['duration']}")

    print("== 2. detect-court (CourtKeyNet) ==")
    t0 = time.time()
    status, court = req("POST", f"/videos/{vid}/detect-court")
    print(f"  status={status} detector={court.get('detector')} "
          f"corners={court.get('corners')} preview={court.get('preview_url')} "
          f"({time.time()-t0:.1f}s)")
    assert status == 200 and court["success"], court

    print("== 3. PUT court (manual corners, shifted slightly) ==")
    corners = court["corners"]
    shifted = [[c[0] + 2, c[1] + 2] for c in corners]
    status, saved = req("PUT", f"/videos/{vid}/court",
                        json.dumps({"corners": shifted}).encode(),
                        {"Content-Type": "application/json"})
    print(f"  status={status} saved_corners={saved['corners']}")
    assert status == 200 and saved["corners"] == shifted

    print("== 4. GET court state ==")
    status, state = req("GET", f"/videos/{vid}/court")
    assert status == 200 and state["corners"] == shifted
    print("  ok")

    print("== 5. start analysis ==")
    payload = json.dumps({
        "corners": shifted,
        "options": {
            "pose_family": "yolo-pose",
            "pose_mode": "balanced",
            "show_skeletons": True,
            "show_player_trajectories": True,
            "show_court_trajectory": True,
            "show_shuttlecock_trajectory": True,
            "show_player_stats": True,
            "show_pose_roi": True,
            "audio": True,
        },
    }).encode()
    status, job = req("POST", f"/videos/{vid}/analyze", payload,
                      {"Content-Type": "application/json"})
    print(f"  status={status} job_id={job['job_id']}")
    assert status == 202, (status, job)
    jid = job["job_id"]

    print("== 6. poll job ==")
    last = None
    for i in range(240):
        status, st = req("GET", f"/jobs/{jid}")
        if st["status"] in ("completed", "failed"):
            last = st
            break
        if i % 10 == 0:
            print(f"  [{i:3d}] {st['status']} {st['progress']}% "
                  f"{st.get('current_frame')}/{st.get('total_frames')} stage={st.get('stage')}")
        time.sleep(1)
    assert last is not None, "job did not finish in 240s"
    print(f"  final: {last['status']} progress={last['progress']}% "
          f"stage={last.get('stage')} error={last.get('error')}")
    assert last["status"] == "completed", last

    print("== 7. fetch results ==")
    status, result = req("GET", f"/jobs/{jid}/results")
    print(f"  video_url={result.get('video_url')}")
    print(f"  metadata_url={result.get('metadata_url')}")
    print(f"  detections_url={result.get('detections_url')}")
    print(f"  visualizations={len(result.get('visualizations', []))}")
    meta = result.get("metadata") or {}
    print(f"  metadata.video={meta.get('video')}")
    print(f"  metadata.court.detector={ (meta.get('court') or {}).get('detector') }")
    assert result["status"] == "completed"
    assert result.get("video_url")
    assert result.get("metadata_url")

    print("== 8. static file check ==")
    for u in [result["video_url"], result["metadata_url"]]:
        url = "http://127.0.0.1:8000" + u
        req2 = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req2, timeout=30) as resp:
            print(f"  {u} -> {resp.status} ({resp.headers.get('Content-Length')} bytes)")

    print("== 9. history ==")
    status, hist = req("GET", "/jobs/history")
    print(f"  history items: {len(hist)}; first: {hist[0]['job_id'] if hist else None}")
    assert any(h["job_id"] == jid for h in hist)

    print("\nALL E2E TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"E2E FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
