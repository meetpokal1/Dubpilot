"""Turns each translated segment into English speech with edge-tts, then
speeds it up or slows it down slightly so it fits the original segment's
timing. That's what keeps the dub roughly lined up with the video instead
of drifting more and more out of sync as it goes.
"""

import asyncio
import os
import subprocess

import edge_tts
from pydub import AudioSegment

from utils import log

DEFAULT_VOICE = "en-US-GuyNeural"


async def _synthesize_text(text, voice, out_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def _speed_adjust(in_path, out_path, factor):
    """Change playback speed without changing pitch, via ffmpeg's atempo.
    atempo only accepts 0.5-2.0 per call, so this clamps to that range —
    good enough here since we only nudge speed by a small amount.
    """
    factor = max(0.5, min(factor, 2.0))
    cmd = ["ffmpeg", "-y", "-i", in_path, "-filter:a", f"atempo={factor:.3f}", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def synthesize_segments(segments, work_dir, voice=DEFAULT_VOICE):
    """Generate one audio clip per segment, time-stretched to fit the
    original segment's duration. Adds an 'audio_path' key to each segment.
    Segments can carry their own 'voice' key (set by diarization) which
    overrides the default — that's how multi-speaker dubbing plugs in.
    """
    clips_dir = os.path.join(work_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        text = seg["english"].strip()
        if not text:
            seg["audio_path"] = None
            continue

        raw_path = os.path.join(clips_dir, f"seg_{i:04d}_raw.mp3")
        final_path = os.path.join(clips_dir, f"seg_{i:04d}.mp3")
        seg_voice = seg.get("voice", voice)

        asyncio.run(_synthesize_text(text, seg_voice, raw_path))

        target_duration = seg["end"] - seg["start"]
        actual_duration = len(AudioSegment.from_file(raw_path)) / 1000.0

        if actual_duration > 0 and target_duration > 0:
            factor = actual_duration / target_duration
            if abs(factor - 1.0) > 0.05:
                _speed_adjust(raw_path, final_path, factor)
            else:
                final_path = raw_path
        else:
            final_path = raw_path

        seg["audio_path"] = final_path
        log("synthesize", f"segment {i + 1}/{len(segments)} done")

    return segments
