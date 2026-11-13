from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .media import AUDIO_RATE

log = logging.getLogger("semantix.model")

QUERY_PROMPTS = {"search": "SearchQuery", "code": "CodeRetrieval"}
CONTEXT_TOKENS = 8192
NATIVE_BF16_CAPABILITY = 8
CPU_BF16_FLAGS = ("avx512_bf16", "amx_bf16")
FRAME_OVERHEAD_TOKENS = 4
SEQUENCE_OVERHEAD_TOKENS = 64
MAX_AUDIO_SECONDS = 300


def max_video_frames(frame_tokens: int) -> int:
    return max((CONTEXT_TOKENS - SEQUENCE_OVERHEAD_TOKENS) // (frame_tokens + FRAME_OVERHEAD_TOKENS), 1)


def video_segment_seconds(config: Config) -> float:
    fps = config.get("video.fps")
    longest = max_video_frames(config.get("video.frame_tokens")) / fps
    return max(min(float(config.get("video.segment_seconds")), longest), 1.0 / fps)


def audio_segment_seconds(config: Config) -> float:
    return float(min(config.get("audio.segment_seconds"), MAX_AUDIO_SECONDS))


class ModelError(RuntimeError):
    pass


class ModelMissingError(ModelError):
    pass


def resolve_device(preference: str) -> str:
    import torch

    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if preference == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if preference == "cuda" and not cuda:
        raise ModelError("model.device is set to cuda but no CUDA GPU is available")
    if preference == "mps" and not mps:
        raise ModelError("model.device is set to mps but Apple Metal is not available")
    return preference


def resolve_dtype(device: str, preference: str = "auto"):
    import torch

    if preference == "float32":
        return torch.float32
    if preference == "bfloat16":
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported() and native_bf16():
        return torch.bfloat16
    if device == "cpu" and cpu_bf16():
        return torch.bfloat16
    return torch.float32


def native_bf16() -> bool:
    import torch

    return torch.cuda.get_device_capability()[0] >= NATIVE_BF16_CAPABILITY


def cpu_bf16(cpuinfo: str = "/proc/cpuinfo") -> bool:
    try:
        with open(cpuinfo) as handle:
            for line in handle:
                if line.startswith("flags"):
                    return any(flag in CPU_BF16_FLAGS for flag in line.partition(":")[2].split())
    except OSError:
        pass
    return False


def local_snapshot(model_id: str, cache_dir: Path) -> Path | None:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(model_id, cache_dir=str(cache_dir), local_files_only=True))
    except Exception:
        return None


def download(model_id: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(model_id, cache_dir=str(cache_dir)))


def normalize(vectors: np.ndarray, dimension: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    vectors = vectors[:, :dimension]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(vectors / norms)


def release_memory() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class Embedder:
    def __init__(self, config: Config, models_dir: Path):
        self.config = config
        self.models_dir = models_dir
        self.dimension: int = config.get("model.dimension")
        self.capabilities: tuple[str, ...] = tuple(config.get("model.capabilities"))
        self.device: str | None = None
        self.dtype: str | None = None
        self.loaded_at: float | None = None
        self.last_used = time.time()
        self._model: Any = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def signature(self) -> tuple:
        get = self.config.get
        return (get("model.id"), tuple(get("model.capabilities")), get("model.device"), get("model.dtype"))

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            snapshot = local_snapshot(self.config.get("model.id"), self.models_dir)
            if snapshot is None:
                raise ModelMissingError("the model is not downloaded yet, run `semantix setup`")

            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()

            device = resolve_device(self.config.get("model.device"))
            dtype = resolve_dtype(device, self.config.get("model.dtype"))
            config_kwargs: dict[str, Any] = {}
            if not ({"image", "video"} & set(self.capabilities)):
                config_kwargs["vision_config"] = None
            if "audio" not in self.capabilities:
                config_kwargs["audio_config"] = None
            started = time.time()
            self._model = SentenceTransformer(
                str(snapshot),
                device=device,
                local_files_only=True,
                model_kwargs={"torch_dtype": dtype},
                config_kwargs=config_kwargs,
                processor_kwargs={"model_max_length": CONTEXT_TOKENS},
            )
            self._model.eval()
            self.device = device
            self.dtype = str(dtype).replace("torch.", "")
            self.loaded_at = time.time()
            self.last_used = time.time()
            log.info(
                "model loaded on %s (%s) with %s in %.1fs",
                device,
                self.dtype,
                ",".join(self.capabilities),
                time.time() - started,
            )

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            self._model = None
            self.loaded_at = None
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            release_memory()
            log.info("model unloaded")

    def gpu_memory_mb(self) -> float | None:
        if self._model is None or self.device != "cuda":
            return None
        import torch

        return round(torch.cuda.memory_allocated() / 1048576, 1)

    def require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise ModelError(
                f"the {capability} capability is turned off, enable it with "
                f"`semantix config set model.capabilities {','.join([*self.capabilities, capability])}`"
            )

    def text_batch_size(self) -> int:
        configured = self.config.get("model.batch_size")
        if configured:
            return configured
        return {"cuda": 32, "mps": 16}.get(self.device or "cpu", 4)

    def media_batch_size(self) -> int:
        configured = self.config.get("model.media_batch_size")
        if configured:
            return configured
        return {"cuda": 8, "mps": 4}.get(self.device or "cpu", 1)

    def _encode(self, inputs: list, batch_size: int, **kwargs) -> np.ndarray:
        import torch

        with self._lock:
            self.load()
            self.last_used = time.time()
            size = max(1, batch_size)
            while True:
                try:
                    with torch.inference_mode():
                        vectors = self._model.encode(
                            inputs,
                            batch_size=size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            **kwargs,
                        )
                    break
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if size == 1:
                        raise ModelError("ran out of memory embedding a single item") from None
                    size = max(1, size // 2)
                    log.warning("out of memory, retrying with batch size %d", size)
            self.last_used = time.time()
        vectors = normalize(vectors, self.dimension)
        if not np.isfinite(vectors).all():
            raise ModelError("the model produced invalid numbers for this input")
        return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode([{"text": text} for text in texts], self.text_batch_size())

    def embed_query(self, text: str) -> dict[str, np.ndarray]:
        import torch

        with self._lock:
            self.load()
            prompts = self._model.prompts
            inputs = [{"text": prompts[name] + text} for name in QUERY_PROMPTS.values()]
            with torch.inference_mode():
                vectors = self._model.encode(inputs, batch_size=2, show_progress_bar=False)
            self.last_used = time.time()
        vectors = normalize(vectors, self.dimension)
        return dict(zip(QUERY_PROMPTS.keys(), vectors, strict=True))

    def embed_images(self, images: list) -> np.ndarray:
        self.require("image")
        tokens = self.config.get("image.tokens")
        return self._encode(
            [{"image": image} for image in images],
            self.media_batch_size(),
            processing_kwargs={"image": {"max_soft_tokens": tokens}},
        )

    def embed_video(self, frames: list, fps: float) -> np.ndarray:
        self.require("video")
        count = len(frames)
        tokens = self.config.get("video.frame_tokens")
        if count > max_video_frames(tokens):
            raise ModelError(f"{count} frames do not fit in one segment, the limit is {max_video_frames(tokens)}")
        metadata = {
            "fps": fps,
            "total_num_frames": count,
            "duration": count / fps,
            "frames_indices": list(range(count)),
        }
        return self._encode(
            [{"video": {"array": frames, "video_metadata": metadata}}],
            1,
            processing_kwargs={"video": {"do_sample_frames": False, "max_soft_tokens": tokens}},
        )[0]

    def video_token_count(self, frames: list, fps: float) -> int:
        self.require("video")
        with self._lock:
            self.load()
            count = len(frames)
            metadata = {
                "fps": fps, "total_num_frames": count, "duration": count / fps, "frames_indices": list(range(count)),
            }
            features = self._model.preprocess(
                [{"video": {"array": frames, "video_metadata": metadata}}],
                processing_kwargs={
                    "video": {"do_sample_frames": False, "max_soft_tokens": self.config.get("video.frame_tokens")}
                },
            )
            return int(features["input_ids"].shape[1])

    def embed_audio(self, samples: np.ndarray) -> np.ndarray:
        self.require("audio")
        if len(samples) > MAX_AUDIO_SECONDS * AUDIO_RATE:
            raise ModelError(f"audio segments longer than {MAX_AUDIO_SECONDS} seconds do not fit")
        return self._encode(
            [{"audio": {"array": samples, "sampling_rate": AUDIO_RATE}}],
            1,
            processing_kwargs={"audio": {"truncation": False}},
        )[0]

    def audio_token_count(self, samples: np.ndarray) -> int:
        self.require("audio")
        with self._lock:
            self.load()
            features = self._model.preprocess(
                [{"audio": {"array": samples, "sampling_rate": AUDIO_RATE}}],
                processing_kwargs={"audio": {"truncation": False}},
            )
            return int(features["input_ids"].shape[1])
