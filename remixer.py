"""Places every dubbed segment clip at its original timestamp to build one
continuous English audio track, then swaps it into the source video
without touching the video stream.
"""

import subprocess

from pydub import AudioSegment

from utils import log


def max_clip_duration_ms(segments, index, total_duration_ms, clip_duration_ms):
    """Limit a speech clip to the video end and the next dubbed segment start.

    This avoids layering a long translated clip over the next segment. It is a
    conservative mitigation; heavily compressed dialogue may still sound rushed
    and should be checked by listening to the exported video.
    """
    if index < 0 or index >= len(segments):
        raise IndexError("Segment index is out of range.")

    segment = segments[index]
    start_seconds = float(segment["start"])
    start_ms = max(0, int(start_seconds * 1000))
    total_ms = max(0, int(total_duration_ms))
    available = min(max(0, int(clip_duration_ms)), max(0, total_ms - start_ms))

    next_starts = [
        max(0, int(float(other["start"]) * 1000))
        for other_index, other in enumerate(segments)
        if other_index != index
        and other.get("audio_path")
        and float(other["start"]) > start_seconds + 0.001
    ]
    if next_starts:
        available = min(available, max(0, min(next_starts) - start_ms))
    return available


def build_dubbed_track(segments, total_duration_ms, out_path):
    """Overlay each segment's synthesized clip onto a silent base track at
    its original start time, so the dub lines up with the video timeline.
    """
    log("remix", "assembling full audio track")
    track = AudioSegment.silent(duration=int(total_duration_ms))

    audible_segments = sorted(
        (seg for seg in segments if seg.get("audio_path")),
        key=lambda item: float(item["start"]),
    )
    for index, seg in enumerate(audible_segments):
        clip = AudioSegment.from_file(seg["audio_path"])
        start_ms = max(0, int(float(seg["start"]) * 1000))
        allowed_ms = max_clip_duration_ms(
            audible_segments, index, total_duration_ms, len(clip)
        )
        if allowed_ms <= 0:
            log("remix", f"segment {index + 1}/{len(audible_segments)} skipped (outside audio timeline)")
            continue
        if allowed_ms < len(clip):
            log(
                "remix",
                f"segment {index + 1}/{len(audible_segments)} tail clipped by "
                f"{len(clip) - allowed_ms} ms to avoid overlapping the next segment",
            )
            clip = clip[:allowed_ms]
        track = track.overlay(clip, position=start_ms)

    track.export(out_path, format="wav")
    return out_path


def mux_audio_into_video(video_path, audio_path, output_path):
    """Replace the video's audio track with the dubbed one. The video
    stream is copied as-is (-c:v copy), so there's no re-encode and no
    quality loss on the picture.
    """
    log("remix", "muxing dubbed audio into video")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("remix", f"final video saved to {output_path}")
