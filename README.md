# Dubpilot
Takes a YouTube video in any language and produces an English-dubbed version: same video, same speaker energy, new audio track.

## Pipeline

```
YouTube URL
    |
    v
[1] download.py     -> yt-dlp fetches the video, ffmpeg pulls a 16kHz wav
    |
    v
[2] transcriber.py  -> Whisper transcribes speech IN THE ORIGINAL LANGUAGE,
    |                   with a start/end timestamp per segment
    v
[3] translator.py   -> each segment's text is translated to English
    |                   (IndicTrans2 for Indian languages, Google Translate
    |                   fallback for everything else)
    v
[4] synthesizer.py  -> edge-tts speaks each English segment, then each
    |                   clip is time-stretched to fit its original
    |                   segment's duration
    v
[5] remixer.py      -> clips are placed at their original timestamps to
    |                   build one audio track, which replaces the video's
    |                   original audio (video stream untouched)
    v
dubbed_output.mp4
```

## Setup

1. Install ffmpeg (system dependency, not pip):
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: download from ffmpeg.org and add it to PATH

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional, for the IndicTrans2 path on Indian-language videos) —
   accept the model terms at
   huggingface.co/ai4bharat/indictrans2-indic-en-1B, then run
   `huggingface-cli login` once, and install:
   ```
   pip install torch transformers sentencepiece
   pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
   ```
   Without this, Hindi/Marathi/Tamil/etc. sources still work fine — the
   pipeline just falls back to Google Translate automatically.

4. (Optional, for `--diarize`, the multi-speaker stretch goal) — accept
   the model terms at huggingface.co/pyannote/speaker-diarization-3.1,
   get an HF access token, and:
   ```
   pip install pyannote.audio
   export HF_TOKEN=your_token_here
   ```

## Usage

```
python main.py "https://youtube.com/watch?v=XXXXXXXX" --out dubbed.mp4
```

Options:
- `--whisper-model` — tiny/base/small/medium/large (default small). Bigger
  = more accurate but slower. See the timing note below.
- `--voice` — any edge-tts voice name (default en-US-GuyNeural). Run
  `edge-tts --list-voices` to see all of them.
- `--diarize` — stretch goal, detects speakers and dubs each in a
  different voice instead of one voice for the whole video.

The terminal prints a timestamped line for every stage as it runs
(`[HH:MM:SS] [stage] message`), and each stage reports how long it took
when it finishes — that's what you'd screenshot or read out for the
processing-time part of the submission.

## Design decisions (for the walkthrough video)

**Why timestamped segments instead of one big transcript.** Whisper
already breaks speech into segments with start/end times. Keeping that
structure all the way through — translate each segment, synthesize each
segment, place each segment at its own timestamp — is what keeps the dub
roughly lined up with the video instead of the audio and video slowly
drifting apart over a long video. Translating and synthesizing the whole
transcript as one block would be simpler code but there'd be no way to
know where any given sentence should land on the timeline.

**Why speed-adjust each clip.** A translated sentence is rarely the same
length as the original — English is often longer than Hindi for the same
meaning, for instance. `ffmpeg`'s `atempo` filter stretches or compresses
each clip to fit its original segment's duration without changing pitch
(so a sped-up clip doesn't sound like a chipmunk). This is the main lever
for keeping "timed to the original," which is one of the two things
they're grading.

**Why separate transcribe and translate instead of Whisper's built-in
translate mode.** Whisper can translate directly to English in one pass,
which would be less code. Splitting it into two stages instead — Whisper
transcribes in the source language, then a separate translator converts
that to English — is what lets IndicTrans2 (trained specifically on
Indian languages) handle the translation for Hindi/Marathi/Tamil/etc,
which is meaningfully better than Whisper's own translation for those
languages. Google Translate (via deep-translator, free, no API key)
covers everything else as a fallback.

**Why -c:v copy on the final mux.** Re-encoding the video stream would
cost a lot of processing time for no quality benefit, since only the
audio track is changing. `-c:v copy` just repackages the existing video
stream into the new container instead of re-encoding it, which is why the
remix stage is fast even on a 2-hour video.

**Known limitation.** The speed-adjust approach keeps segments aligned to
their *start* time, but very dense speech translated into longer English
phrasing can still push a segment's audio slightly past its original end
before overlapping the next one — pydub's `overlay` just layers it on top
rather than clipping it. For a video with tightly-packed dialogue this can
cause minor overlap. A stricter fix would be to detect and shorten/trim
overlapping tails, which is a natural next improvement to mention if
asked about limitations.

## A note on the two test videos

For the 30-minute video, the default `small` Whisper model on CPU should
finish in a reasonable time. For the 2-hour video, transcription is the
slowest stage — on CPU, expect it to take a multiple of the video's real
length depending on your machine. Two practical options if it's too slow:
- Run with `--whisper-model base` (faster, slightly less accurate) for
  the long video specifically.
- Run on a machine/Colab with a GPU, which speeds up both Whisper and
  IndicTrans2 substantially.

Either way, note which model size and hardware you used for each of the
two submissions — that trade-off (accuracy vs. processing time) is a
natural thing to discuss in the walkthrough video.

## Project layout

```
main.py          CLI entry point, orchestrates the pipeline
downloader.py     yt-dlp download + ffmpeg audio extraction
transcriber.py    Whisper speech-to-text with timestamps
translator.py     IndicTrans2 / Google Translate to English
synthesizer.py    edge-tts speech synthesis + timing adjustment
remixer.py        builds the final audio track and muxes it into the video
diarization.py    optional stretch goal: per-speaker voices
utils.py          logging and timing helpers
```
