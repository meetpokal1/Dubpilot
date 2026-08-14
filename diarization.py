"""Optional stretch goal: tell speakers apart and dub each one in a
different voice. Only activates with --diarize, and only runs if
pyannote.audio is installed and a Hugging Face token is available (the
pyannote diarization model is gated — accept its terms on HF and set
HF_TOKEN as an environment variable to use this).

If either piece is missing, diarize() just returns None and the caller
falls back to a single voice for the whole video, which is the required
part of the assignment working as normal.
"""

import os

from utils import log

SPEAKER_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
]


def diarize(audio_path):
    """Returns a list of (start, end, speaker_label) turns, or None if
    diarization isn't available.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log("diarize", "pyannote.audio not installed, skipping speaker detection")
        return None

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log("diarize", "no HF_TOKEN set, skipping speaker detection")
        return None

    log("diarize", "running speaker diarization")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    result = pipeline(audio_path)

    turns = [(turn.start, turn.end, speaker) for turn, _, speaker in result.itertracks(yield_label=True)]
    log("diarize", f"found {len(set(t[2] for t in turns))} speakers across {len(turns)} turns")
    return turns


def assign_speakers_to_segments(segments, turns):
    """Tag each transcript segment with the speaker whose turn overlaps its
    midpoint, and map each distinct speaker to its own TTS voice.
    """
    if not turns:
        for seg in segments:
            seg["voice"] = SPEAKER_VOICES[0]
        return segments

    speaker_voice_map = {}
    next_voice = 0

    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        speaker = next((s for start, end, s in turns if start <= mid <= end), "SPEAKER_00")

        if speaker not in speaker_voice_map:
            speaker_voice_map[speaker] = SPEAKER_VOICES[next_voice % len(SPEAKER_VOICES)]
            next_voice += 1

        seg["voice"] = speaker_voice_map[speaker]

    log("diarize", f"mapped {len(speaker_voice_map)} speakers to distinct voices")
    return segments
