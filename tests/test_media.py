from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from semantix import benchmark, media
from semantix.config import Config, ConfigError
from semantix.model import (
    CONTEXT_TOKENS,
    MAX_AUDIO_SECONDS,
    audio_segment_seconds,
    max_video_frames,
    video_segment_seconds,
)


def test_video_segments_never_exceed_the_model_context():
    for tokens in (70, 140, 280):
        assert max_video_frames(tokens) * (tokens + 4) + 64 <= CONTEXT_TOKENS
    assert video_segment_seconds(Config()) == 30
    assert video_segment_seconds(Config({"video.frame_tokens": 280})) == 28
    assert video_segment_seconds(Config({"video.segment_seconds": 3600})) == 56
    assert video_segment_seconds(Config({"video.segment_seconds": 3600, "video.fps": 0.5})) == 112
    assert video_segment_seconds(Config({"video.segment_seconds": 2, "video.fps": 0.25})) == 4


def test_audio_segment_is_capped():
    assert audio_segment_seconds(Config()) == 30
    assert audio_segment_seconds(Config({"audio.segment_seconds": 300})) == MAX_AUDIO_SECONDS
    with pytest.raises(ConfigError):
        Config({"audio.segment_seconds": 301})


def test_video_is_cut_into_full_segments(tmp_path):
    target = tmp_path / "clip.mp4"
    benchmark.write_video(target, seconds=34)
    info = media.probe(str(target))
    assert info.has_video and not info.has_audio and 33 <= info.duration <= 35
    segments = list(media.iter_video_segments(str(target), 1.0, 30))
    assert [len(frames) for _start, _end, frames in segments] == [30, 4]
    assert segments[0][:2] == (0.0, 30.0) and segments[1][0] == 30.0
    halved = list(media.iter_video_segments(str(target), 0.5, 30))
    assert [len(frames) for _start, _end, frames in halved] == [15, 2]


def test_audio_is_cut_into_segments_and_silence_is_dropped(tmp_path):
    target = tmp_path / "sweep.wav"
    benchmark.write_audio(target, seconds=20)
    segments = list(media.iter_audio_segments(str(target), 8))
    assert [(start, end) for start, end, _samples in segments] == [(0.0, 8.0), (8.0, 16.0), (16.0, 20.0)]
    assert all(samples.dtype == np.float32 for _start, _end, samples in segments)
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(media.AUDIO_RATE)
        handle.writeframes(np.zeros(media.AUDIO_RATE * 5, dtype=np.int16).tobytes())
    assert list(media.iter_audio_segments(str(silent), 8)) == []


def test_full_length_video_goes_through_the_engine(engine, tmp_path):
    from conftest import run_sync

    folder = tmp_path / "clips"
    folder.mkdir()
    benchmark.write_video(folder / "long.mp4", seconds=34)
    _plan, done = run_sync(engine, [str(folder)])
    assert done["embedded"] == 1 and done["failed"] == 0 and done["chunks"] == 2
    hit = engine.search("video", kinds=["video"], per_file=0)["hits"]
    assert sorted((part["start_sec"], part["end_sec"] > part["start_sec"]) for part in hit) == [(0.0, True), (30.0, True)]


MODELS = os.environ.get("SEMANTIX_TEST_MODELS")


@pytest.mark.skipif(not MODELS, reason="set SEMANTIX_TEST_MODELS to a models directory to run real-model checks")
def test_real_model_accepts_the_longest_segments():
    from semantix.model import Embedder

    embedder = Embedder(Config({"model.device": "cpu"}), Path(MODELS))
    embedder.load()
    try:
        limits = benchmark.check_limits(embedder)
    finally:
        embedder.unload()
    assert limits["ok"], limits
    assert [entry["frames"] for entry in limits["video"]] == [109, 56, 28]
    assert limits["audio"][-1]["seconds"] == MAX_AUDIO_SECONDS and limits["audio"][-1]["complete"]


def test_oversized_segment_is_split_instead_of_failing():
    from semantix.engine import split_segments

    pieces = list(split_segments(iter([(0.0, 10.0, list(range(10))), (10.0, 13.0, [1, 2, 3])]), 4))
    assert [(begin, end, len(frames)) for begin, end, frames in pieces] == [
        (0.0, 4.0, 4), (4.0, 8.0, 4), (8.0, 10.0, 2), (10.0, 13.0, 3),
    ]


def test_auto_number_format_follows_the_gpu_generation(monkeypatch):
    import torch

    from semantix.model import resolve_dtype

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *args, **kwargs: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (7, 5))
    assert resolve_dtype("cuda") == torch.float32
    assert resolve_dtype("cuda", "bfloat16") == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 9))
    assert resolve_dtype("cuda") == torch.bfloat16
    assert resolve_dtype("cuda", "float32") == torch.float32
    assert resolve_dtype("cpu") == torch.float32 and resolve_dtype("mps") == torch.float32
