import json
import os
import subprocess
import time

import cv2


def encode_vscode_compatible_mp4(input_video_path, output_path, audio_source_path=None):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final_output_path = output_path
    if os.path.abspath(input_video_path) == os.path.abspath(output_path):
        final_output_path = f"{output_path}.h264.tmp.mp4"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
    ]
    if audio_source_path:
        command.extend(["-i", audio_source_path])

    command.extend(
        [
            "-map",
            "0:v:0",
        ]
    )
    if audio_source_path:
        command.extend(["-map", "1:a:0?", "-shortest"])
    else:
        command.append("-an")

    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            final_output_path,
        ]
    )

    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not os.path.exists(final_output_path) or os.path.getsize(final_output_path) == 0:
        message = result.stderr.strip()[-1000:] if result.stderr else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg H.264 export failed: {message}")

    if final_output_path != output_path:
        os.replace(final_output_path, output_path)

    return True


def has_audio_track(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            return any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))

        try:
            from moviepy.editor import VideoFileClip

            video = VideoFileClip(video_path)
            has_audio = video.audio is not None
            video.close()
            return has_audio
        except Exception:
            return False
    except Exception as exc:
        print(f"Error checking audio track: {exc}")
        return True


def process_video_with_audio(video_path, temp_video_path, output_path, save_dir):
    try:
        print("\nProcessing video audio...")

        if not has_audio_track(video_path):
            print("No audio track detected; exporting video without audio.")
            return process_video_without_audio(temp_video_path, output_path)

        if not os.path.exists(temp_video_path):
            raise FileNotFoundError(f"Temporary video not found: {temp_video_path}")

        encode_vscode_compatible_mp4(temp_video_path, output_path, audio_source_path=video_path)

        print(f"Video with audio saved to: {output_path}")
        cleanup_temp_files([temp_video_path])
        return True

    except Exception as exc:
        print(f"Audio merge failed: {exc}")
        print("Falling back to video without audio.")
        return process_video_without_audio(temp_video_path, output_path)


def process_video_without_audio(temp_video_path, output_path):
    try:
        print("\nProcessing video without audio...")
        if not os.path.exists(temp_video_path):
            raise FileNotFoundError(f"Temporary video not found: {temp_video_path}")

        encode_vscode_compatible_mp4(temp_video_path, output_path)

        print(f"Video saved to: {output_path}")
        cleanup_temp_files([temp_video_path])
        return True
    except Exception as exc:
        print(f"Video processing failed: {exc}")
        return False


def setup_video_writer(frame_width, frame_height, fps, temp_output_path):
    os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
    # OpenCV writes a temporary mp4v file; final export is transcoded to H.264.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video writer: {temp_output_path}")
    return writer


def cleanup_temp_files(file_list, keep_temp_video=False):
    for file_path in file_list:
        if keep_temp_video and file_path and "temp_detect_" in os.path.basename(file_path):
            continue
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as exc:
            print(f"Failed to remove temporary file {file_path}: {exc}")

    time.sleep(0.1)

