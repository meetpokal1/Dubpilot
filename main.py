"""Automated Video Dubbing System.

Takes a YouTube URL in any language, downloads the video, transcribes and
translates its speech to English, synthesizes a natural English voiceover
timed to match the original, and outputs a dubbed video.

Usage:
    python main.py "https://youtube.com/watch?v=..." --out dubbed.mp4
    python main.py "https://youtube.com/watch?v=..." --diarize   # stretch goal
"""

import argparse
import os
import time

from diarization import assign_speakers_to_segments, diarize
from downloader import download_video, extract_audio
from pydub.utils import mediainfo
from remixer import build_dubbed_track, mux_audio_into_video
from synthesizer import DEFAULT_VOICE, synthesize_segments
from transcriber import transcribe
from translator import translate_segments
from utils import Timer, log


def parse_args():
    parser = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--out", default="dubbed_output.mp4", help="Path for the final dubbed video")
    parser.add_argument("--work-dir", default="work", help="Scratch folder for intermediate files")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size: tiny/base/small/medium/large")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts voice for the English dub")
    parser.add_argument("--diarize", action="store_true", help="Stretch goal: detect speakers and use a different voice per speaker")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    pipeline_start = time.time()

    with Timer("download"):
        video_path = download_video(args.url, args.work_dir)
        audio_path = extract_audio(video_path, args.work_dir)

    with Timer("transcribe"):
        language, segments = transcribe(audio_path, model_size=args.whisper_model)

    with Timer("translate"):
        segments = translate_segments(segments, source_lang=language)

    if args.diarize:
        with Timer("diarize"):
            turns = diarize(audio_path)
            segments = assign_speakers_to_segments(segments, turns)

    with Timer("synthesize"):
        segments = synthesize_segments(segments, args.work_dir, voice=args.voice)

    with Timer("remix"):
        total_duration_ms = float(mediainfo(video_path)["duration"]) * 1000
        dubbed_audio_path = os.path.join(args.work_dir, "dubbed_audio.wav")
        dubbed_audio = build_dubbed_track(segments, total_duration_ms, dubbed_audio_path)
        mux_audio_into_video(video_path, dubbed_audio, args.out)

    total_elapsed = time.time() - pipeline_start
    log("done", f"finished in {total_elapsed / 60:.1f} minutes, output at {args.out}")


if __name__ == "__main__":
    main()
