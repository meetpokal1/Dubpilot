"""Automated Video Dubbing System with resumable pipeline stages.

Usage:
    python main.py "https://youtube.com/watch?v=..." --out dubbed.mp4
    python main.py "https://youtube.com/watch?v=..." --resume
"""
import argparse
import json
import os
import time
from pathlib import Path

from diarization import assign_speakers_to_segments, diarize
from downloader import download_video, extract_audio
from checkpoint import PipelineCheckpoint
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
    parser.add_argument("--work-dir", default="work", help="Scratch folder and checkpoint directory")
    parser.add_argument("--whisper-model", default="small", choices=("tiny", "base", "small", "medium", "large"))
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts voice for the English dub")
    parser.add_argument("--diarize", action="store_true", help="Detect speakers and use a different voice per speaker")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed stages and valid segment clips from the same work directory",
    )
    return parser.parse_args()


def _write_json(path: str, payload) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return str(target)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _valid_file(path) -> bool:
    return isinstance(path, str) and os.path.isfile(path) and os.path.getsize(path) > 0


def _valid_segment_file(path) -> bool:
    if not _valid_file(path):
        return False
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, list)


def main():
    args = parse_args()
    work_dir = os.path.abspath(args.work_dir)
    output_path = os.path.abspath(args.out)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    config = {
        "url": args.url,
        "whisper_model": args.whisper_model,
        "voice": args.voice,
        "diarize": bool(args.diarize),
        "output_path": output_path,
        "work_dir": work_dir,
    }
    try:
        checkpoint = PipelineCheckpoint(
            os.path.join(work_dir, "pipeline_state.json"),
            config=config,
            resume=args.resume,
        )
    except ValueError as exc:
        raise SystemExit(f"Cannot resume this job: {exc}") from exc

    if args.resume and checkpoint.is_complete("complete") and _valid_file(output_path):
        log("resume", f"all stages already completed; reusing {output_path}")
        return

    pipeline_start = time.time()

    # Stage 1: download and extract audio.
    download_state = checkpoint.artifact("download")
    if (
        args.resume
        and checkpoint.is_complete("download")
        and _valid_file(download_state.get("video_path"))
        and _valid_file(download_state.get("audio_path"))
    ):
        video_path = download_state["video_path"]
        audio_path = download_state["audio_path"]
        log("resume", "reusing downloaded video and extracted audio")
    else:
        checkpoint.invalidate_from("download")
        with Timer("download"):
            video_path = download_video(args.url, work_dir)
            audio_path = extract_audio(video_path, work_dir)
        video_path = os.path.abspath(video_path)
        audio_path = os.path.abspath(audio_path)
        if not _valid_file(video_path) or not _valid_file(audio_path):
            raise RuntimeError("Download stage did not produce a non-empty video and audio file.")
        checkpoint.mark_complete("download", video_path=video_path, audio_path=audio_path)

    # Stage 2: transcription.
    transcription_path = os.path.join(work_dir, "transcription.json")
    transcribe_state = checkpoint.artifact("transcribe")
    if (
        args.resume
        and checkpoint.is_complete("transcribe")
        and _valid_file(transcribe_state.get("result_path"))
    ):
        transcript = _read_json(transcribe_state["result_path"])
        language = transcript["language"]
        segments = transcript["segments"]
        log("resume", f"reusing transcript ({len(segments)} segments, language={language})")
    else:
        checkpoint.invalidate_from("transcribe")
        with Timer("transcribe"):
            language, segments = transcribe(audio_path, model_size=args.whisper_model)
        transcription_path = _write_json(
            transcription_path,
            {"language": language, "segments": segments},
        )
        checkpoint.mark_complete("transcribe", result_path=transcription_path, language=language)

    # Stage 3: translation.
    translated_path = os.path.join(work_dir, "translated_segments.json")
    translation_state = checkpoint.artifact("translate")
    if (
        args.resume
        and checkpoint.is_complete("translate")
        and _valid_segment_file(translation_state.get("segments_path"))
    ):
        segments = _read_json(translation_state["segments_path"])
        log("resume", f"reusing translated transcript ({len(segments)} segments)")
    else:
        checkpoint.invalidate_from("translate")
        with Timer("translate"):
            segments = translate_segments(segments, source_lang=language)
        translated_path = _write_json(translated_path, segments)
        checkpoint.mark_complete("translate", segments_path=translated_path)

    # Stage 4: optional speaker diarization.
    diarized_path = os.path.join(work_dir, "speaker_segments.json")
    diarize_state = checkpoint.artifact("diarize")
    if args.diarize:
        if (
            args.resume
            and checkpoint.is_complete("diarize")
            and _valid_segment_file(diarize_state.get("segments_path"))
        ):
            segments = _read_json(diarize_state["segments_path"])
            log("resume", "reusing speaker assignments")
        else:
            checkpoint.invalidate_from("diarize")
            with Timer("diarize"):
                turns = diarize(audio_path)
                segments = assign_speakers_to_segments(segments, turns)
            diarized_path = _write_json(diarized_path, segments)
            checkpoint.mark_complete("diarize", segments_path=diarized_path, skipped=False)
    else:
        if not checkpoint.is_complete("diarize") or not diarize_state.get("skipped", False):
            checkpoint.invalidate_from("diarize")
            checkpoint.mark_complete(
                "diarize",
                segments_path=translated_state_path(checkpoint),
                skipped=True,
            )
        log("diarize", "skipped (single voice mode)")

    # Stage 5: speech synthesis. The synthesizer also caches individual clips
    # by text/voice/timing when --resume is used, so a failed long run need not
    # repeat all completed TTS calls.
    synthesized_path = os.path.join(work_dir, "synthesized_segments.json")
    synth_state = checkpoint.artifact("synthesize")
    if (
        args.resume
        and checkpoint.is_complete("synthesize")
        and _valid_segment_file(synth_state.get("segments_path"))
    ):
        cached_segments = _read_json(synth_state["segments_path"])
        if all(not seg.get("audio_path") or _valid_file(seg.get("audio_path")) for seg in cached_segments):
            segments = cached_segments
            log("resume", f"reusing synthesized clips ({len(segments)} segments)")
        else:
            segments = None
    else:
        segments = None

    if segments is None or not all(
        not seg.get("audio_path") or _valid_file(seg.get("audio_path")) for seg in segments
    ):
        checkpoint.invalidate_from("synthesize")
        with Timer("synthesize"):
            segments = synthesize_segments(
                segments if segments is not None else _load_segments_for_synthesis(checkpoint, args.diarize),
                work_dir,
                voice=args.voice,
                resume=args.resume,
            )
        synthesized_path = _write_json(synthesized_path, segments)
        checkpoint.mark_complete("synthesize", segments_path=synthesized_path)

    # Stage 6: mix timestamped clips and mux the audio back into the source video.
    remix_state = checkpoint.artifact("remix")
    if args.resume and checkpoint.is_complete("remix") and _valid_file(output_path):
        log("resume", f"reusing completed output {output_path}")
    else:
        checkpoint.invalidate_from("remix")
        with Timer("remix"):
            total_duration_ms = float(mediainfo(video_path)["duration"]) * 1000
            dubbed_audio_path = os.path.join(work_dir, "dubbed_audio.wav")
            dubbed_audio = build_dubbed_track(segments, total_duration_ms, dubbed_audio_path)
            mux_audio_into_video(video_path, dubbed_audio, output_path)
        checkpoint.mark_complete(
            "remix",
            dubbed_audio_path=os.path.abspath(dubbed_audio_path),
            output_path=output_path,
        )

    total_elapsed = time.time() - pipeline_start
    checkpoint.mark_complete("complete", output_path=output_path, elapsed_seconds=round(total_elapsed, 2))
    log("done", f"finished in {total_elapsed / 60:.1f} minutes, output at {output_path}")


def translated_state_path(checkpoint):
    """Path to the translated segments for single-voice mode."""
    return checkpoint.artifact("translate").get("segments_path", "")


def _load_segments_for_synthesis(checkpoint, diarize_enabled):
    """Load the latest stage's segment list when a synthesis stage needs rebuilding."""
    stage = "diarize" if diarize_enabled else "translate"
    path = checkpoint.artifact(stage).get("segments_path")
    if not _valid_segment_file(path):
        raise RuntimeError(
            f"Cannot resume synthesis because {stage} segment data is missing. "
            "Retry without --resume to rebuild the work directory."
        )
    return _read_json(path)


if __name__ == "__main__":
    main()
