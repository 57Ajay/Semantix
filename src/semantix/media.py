from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

AUDIO_RATE = 16000
MAX_IMAGE_SIDE = 2048
MAX_FRAME_SIDE = 896
SILENCE_RMS = 1e-4
MIN_AUDIO_SECONDS = 1.0


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_video: bool
    has_audio: bool


def probe(path: str) -> MediaInfo:
    import av

    with av.open(path) as container:
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        video = next(iter(container.streams.video), None)
        audio = next(iter(container.streams.audio), None)
        if not duration:
            stream = video or audio
            if stream is not None and stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
        return MediaInfo(max(duration, 0.0), video is not None, audio is not None)


def load_image(path: str):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


def iter_video_segments(path: str, fps: float, segment_seconds: float) -> Iterator[tuple[float, float, list]]:
    import av

    step = 1.0 / fps
    with av.open(path) as container:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            return
        stream.thread_type = "AUTO"
        origin = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        next_sample = 0.0
        segment_start = 0.0
        frames: list = []
        last_time = 0.0
        fallback_step = float(1 / stream.average_rate) if stream.average_rate else 1 / 30
        position = 0
        for frame in container.decode(stream):
            if frame.pts is not None and stream.time_base is not None:
                moment = float(frame.pts * stream.time_base) - origin
            else:
                moment = position * fallback_step
            position += 1
            if moment < 0:
                continue
            last_time = moment
            if moment + 1e-6 < next_sample:
                continue
            while moment >= segment_start + segment_seconds:
                if frames:
                    yield segment_start, segment_start + segment_seconds, frames
                    frames = []
                segment_start += segment_seconds
            image = frame.to_image()
            if max(image.size) > MAX_FRAME_SIDE:
                image.thumbnail((MAX_FRAME_SIDE, MAX_FRAME_SIDE))
            frames.append(image)
            next_sample = max(next_sample + step, moment - (moment % step) + step)
        if frames:
            yield segment_start, max(last_time, segment_start + step), frames


def iter_audio_segments(path: str, segment_seconds: float) -> Iterator[tuple[float, float, np.ndarray]]:
    import av

    window = int(segment_seconds * AUDIO_RATE)
    with av.open(path) as container:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            return
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
        pending: list[np.ndarray] = []
        buffered = 0
        offset = 0

        def emit(samples: np.ndarray, begin: int):
            start = begin / AUDIO_RATE
            return start, start + len(samples) / AUDIO_RATE, samples

        def drain(final: bool):
            nonlocal pending, buffered, offset
            while buffered >= window or (final and buffered > 0):
                merged = np.concatenate(pending) if len(pending) > 1 else pending[0]
                piece, rest = merged[:window], merged[window:]
                pending = [rest] if len(rest) else []
                buffered = len(rest)
                begin = offset
                offset += len(piece)
                if len(piece) >= MIN_AUDIO_SECONDS * AUDIO_RATE and not is_silent(piece):
                    yield emit(piece, begin)

        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                samples = converted.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                pending.append(samples)
                buffered += len(samples)
            yield from drain(final=False)
        for converted in resampler.resample(None):
            samples = converted.to_ndarray().reshape(-1).astype(np.float32, copy=False)
            pending.append(samples)
            buffered += len(samples)
        yield from drain(final=True)


def is_silent(samples: np.ndarray) -> bool:
    if not len(samples):
        return True
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) < SILENCE_RMS
