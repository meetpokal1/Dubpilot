# Dubpilot — Automated Video Dubbing System

Dubpilot takes a video in another language and produces an English-dubbed copy while preserving the original video stream. It demonstrates a staged media pipeline: download, speech recognition, translation, text-to-speech, timestamped audio assembly, and final muxing.

## Interface Preview (Illustrative)

![Dubpilot processing console interface preview](screenshots/dubpilot-processing-console.svg)

> Illustrative console preview based on the documented stages and resume safeguards. The displayed lines are examples, not logs captured from an actual execution.


## The operational problem addressed

Speech recognition, translation and voice synthesis can take a long time—especially for lengthy videos or CPU-only systems. If a run is interrupted, a naive workflow may repeat expensive stages and TTS calls. Separately, translated phrases may be longer than the source speech and can overlap the next segment on the audio timeline.

## Pipeline

1. **Download / audio extraction:** yt-dlp downloads the video; FFmpeg extracts a 16 kHz mono WAV for Whisper.
2. **Transcription:** Whisper returns the source language and timestamped text segments.
3. **Translation:** Indian-language sources can use optional IndicTrans2; otherwise Google Translate is used as a fallback.
4. **Optional speaker diarization:** with the relevant dependencies and Hugging Face token, segments can be assigned different voices.
5. **Speech synthesis:** edge-tts creates a clip for each English segment and FFmpeg can time-adjust it to the source segment duration.
6. **Remix / mux:** clips are placed on a timeline and the dubbed audio replaces the source audio while the video stream is copied.

## Improvements in this version

- **Stage-level checkpointing:** after a stage finishes, the job stores a JSON checkpoint in the work directory, including the completed stages and artifact paths.
- **Resume a long run:** `--resume` reloads completed download, transcription, translation, diarization, synthesis and remix stages when their saved artifacts are still valid.
- **Per-segment synthesis cache:** a completed clip is reused only when the English text, selected voice and original segment timing match a stored cache key. This is especially useful if a long TTS stage is interrupted midway.
- **Settings safety:** the checkpoint includes the source URL, voice, Whisper model, diarization option, output path and work directory. A resume request with different settings is rejected instead of silently reusing incompatible artifacts.
- **Overlap mitigation:** the remixer clips the tail of a dubbed segment at the next dubbed segment's start (or at the end of the video), preventing two generated voice clips from being layered over each other on the timeline.
- **Regression tests and CI:** checkpoint round-trips, config mismatch handling, downstream-stage invalidation and clip timing are tested.

## Setup

### System dependency

Install FFmpeg and make sure `ffmpeg` is available on your PATH.

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: install FFmpeg and add its `bin` directory to PATH

### Python dependencies

    python -m pip install -r requirements.txt

For the optional IndicTrans2 path, accept the model terms at Hugging Face and follow AI4Bharat's current installation instructions. Without those optional dependencies, the general translation fallback remains available.

For optional multi-speaker dubbing, install `pyannote.audio`, accept the model terms, and set `HF_TOKEN` in your environment.

## Usage

Start a new job:

    python main.py "https://youtube.com/watch?v=XXXXXXXX" --out dubbed.mp4

Resume or continue an interrupted job using its original work directory and the same settings:

    python main.py "https://youtube.com/watch?v=XXXXXXXX" --out dubbed.mp4 --work-dir work --resume

Useful options:

- `--whisper-model` — `tiny`, `base`, `small`, `medium`, or `large` (default: `small`).
- `--voice` — any edge-tts voice name (default: `en-US-GuyNeural`).
- `--diarize` — optional speaker-aware voices.
- `--work-dir` — location for source media, checkpoints and intermediate artifacts.
- `--resume` — reuse completed stages and valid cached clips for the same job.

**Important:** use the same URL, model, voice, output and work directory with `--resume`. To deliberately start fresh, omit `--resume` or choose a new work directory. Keep the work directory until the final output has been checked.

The checkpoint file is `work/pipeline_state.json`. Transcription, translation and synthesis artifacts are stored separately in that folder so they can be inspected and reused.

## Clip alignment: what the overlap fix does—and does not do

The remixer prevents generated segments from playing over the next generated segment. If English audio is longer than the available gap, the tail is trimmed. This avoids overlapping generated dialogue, but a trimmed phrase can lose a final word. Listen to the output, especially dense conversations, before relying on it. A more advanced future version could use better speech-rate control, pause-aware segmentation and audio quality checks instead of relying on trimming alone.

## Design decisions

- **Keep transcript segments timestamped.** Transcribing and translating one huge text block would lose the time ranges needed to align the dub with the original video.
- **Separate transcription from translation.** This allows optional IndicTrans2 handling for Indian languages rather than relying exclusively on an all-purpose translator.
- **Cache individual TTS clips by input key.** A resume should not reuse a clip if its wording, voice or timing has changed.
- **Copy the video stream during muxing.** Only the audio changes, so FFmpeg can use `-c:v copy` rather than re-encoding the visual stream.

## Tests

    python -m pip install pytest pydub
    python -m pytest tests/test_checkpoint_and_remix.py -q

GitHub Actions runs the resume-state and clip-timing tests, plus a Python syntax compilation check.

## Project layout

- `main.py` — CLI orchestration and resume decisions
- `checkpoint.py` — durable checkpoint state and downstream invalidation
- `downloader.py` — video download and audio extraction
- `transcriber.py` — timestamped speech-to-text
- `translator.py` — IndicTrans2 / Google translation fallback
- `synthesizer.py` — voice generation, time adjustment and input-keyed clip caching
- `remixer.py` — clip placement, overlap mitigation and audio/video mux
- `diarization.py` — optional speaker-to-voice mapping
- `utils.py` — timestamped stage logs and timers

## Responsible use

Only process content you are authorised to use. Respect platform terms, copyright, privacy, and applicable rules about synthetic or translated speech.
