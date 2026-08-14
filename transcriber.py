"""Speech-to-text using Whisper. Produces timestamped segments in the
video's original language — the translator module converts these to
English afterward. Keeping transcription and translation as separate
stages (instead of Whisper's built-in translate mode) is what lets us
also feed IndicTrans2 for Indian-language sources.
"""

import whisper

from utils import log


def transcribe(audio_path, model_size="small"):
    """Run Whisper on the extracted audio.

    Returns (language, segments) where segments is a list of dicts with
    start, end and text keys, in the source language.
    """
    log("transcribe", f"loading whisper '{model_size}' model")
    model = whisper.load_model(model_size)

    log("transcribe", "running speech recognition")
    result = model.transcribe(audio_path, task="transcribe", verbose=False)

    language = result.get("language", "unknown")
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
    ]

    log("transcribe", f"detected language: {language}, {len(segments)} segments")
    return language, segments
