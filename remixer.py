"""Places every dubbed segment clip at its original timestamp to build one
continuous English audio track, then swaps it into the source video
without touching the video stream.
"""

import subprocess

from pydub import AudioSegment

from utils import log


def build_dubbed_track(segments, total_duration_ms, out_path):
    """Overlay each segment's synthesized clip onto a silent base track at
    its original start time, so the dub lines up with the video timeline.
    """
    log("remix", "assembling full audio track")
    track = AudioSegment.silent(duration=int(total_duration_ms))

    for seg in segments:
        if not seg["audio_path"]:
            continue
        clip = AudioSegment.from_file(seg["audio_path"])
        start_ms = int(seg["start"] * 1000)
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
