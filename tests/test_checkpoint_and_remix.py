from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint import PipelineCheckpoint  # noqa: E402
from remixer import max_clip_duration_ms  # noqa: E402


def test_checkpoint_round_trip_and_downstream_invalidation(tmp_path):
    path = tmp_path / "work" / "pipeline_state.json"
    config = {"url": "https://example.invalid/video", "voice": "en-US-GuyNeural"}

    state = PipelineCheckpoint(path, config, resume=False)
    state.mark_complete("download", video_path="/tmp/video.mp4")
    state.mark_complete("transcribe", result_path="/tmp/transcript.json")

    resumed = PipelineCheckpoint(path, config, resume=True)
    assert resumed.is_complete("download")
    assert resumed.is_complete("transcribe")
    assert resumed.artifact("download")["video_path"] == "/tmp/video.mp4"

    resumed.invalidate_from("transcribe")
    assert resumed.is_complete("download")
    assert not resumed.is_complete("transcribe")
    assert resumed.artifact("transcribe") == {}


def test_resume_rejects_different_job_settings(tmp_path):
    path = tmp_path / "pipeline_state.json"
    PipelineCheckpoint(path, {"url": "source-A", "voice": "voice-A"}, resume=False)

    with pytest.raises(ValueError, match="different URL/model/voice/output"):
        PipelineCheckpoint(path, {"url": "source-B", "voice": "voice-A"}, resume=True)


def test_clip_is_capped_at_next_segment_start():
    segments = [
        {"start": 0.0, "end": 2.0, "audio_path": "one.mp3"},
        {"start": 2.5, "end": 4.0, "audio_path": "two.mp3"},
        {"start": 5.0, "end": 6.0, "audio_path": "three.mp3"},
    ]

    assert max_clip_duration_ms(segments, 0, 10_000, 4_000) == 2_500
    assert max_clip_duration_ms(segments, 1, 10_000, 4_000) == 2_500
    assert max_clip_duration_ms(segments, 2, 6_000, 4_000) == 1_000


def test_clip_outside_video_timeline_is_not_played():
    segments = [{"start": 8.0, "end": 12.0, "audio_path": "one.mp3"}]
    assert max_clip_duration_ms(segments, 0, 7_000, 2_000) == 0
