"""Downloads a YouTube video and pulls out a clean audio track for
transcription. Two separate steps because Whisper wants a plain wav,
not the mp4 container.
"""

import os
import subprocess

from utils import log


def download_video(url, out_dir):
    """Download the best available mp4 for the given URL using yt-dlp.

    Returns the path to the downloaded video file.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "source.%(ext)s")

    log("download", f"fetching {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ]
    subprocess.run(cmd, check=True)

    video_path = os.path.join(out_dir, "source.mp4")
    if not os.path.exists(video_path):
        # yt-dlp keeps the original extension when no merge was needed
        for fname in os.listdir(out_dir):
            if fname.startswith("source."):
                video_path = os.path.join(out_dir, fname)
                break

    log("download", f"saved to {video_path}")
    return video_path


def extract_audio(video_path, out_dir):
    """Pull a 16kHz mono wav out of the video — the format Whisper expects."""
    audio_path = os.path.join(out_dir, "source_audio.wav")
    log("download", "extracting audio track")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        audio_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return audio_path
