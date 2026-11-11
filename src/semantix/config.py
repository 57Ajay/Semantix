from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

CAPABILITIES = ("text", "image", "video", "audio")
DIMENSIONS = (768, 512, 256, 128)


@dataclass(frozen=True)
class Option:
    default: Any
    help: str
    choices: tuple | None = None
    minimum: float | None = None


OPTIONS: dict[str, Option] = {
    "model.id": Option("google/embeddinggemma-2", "Hugging Face model id"),
    "model.capabilities": Option(
        list(CAPABILITIES), "Modalities loaded into memory (text is always on)", CAPABILITIES
    ),
    "model.dimension": Option(768, "Stored vector size", DIMENSIONS),
    "model.device": Option("auto", "Compute device", ("auto", "cuda", "mps", "cpu")),
    "model.dtype": Option("auto", "Number format; auto uses bfloat16 on CUDA when supported", ("auto", "float32", "bfloat16")),
    "model.batch_size": Option(0, "Text batch size, 0 picks one for the device", minimum=0),
    "daemon.auto_start": Option(True, "Start the background service on demand"),
    "daemon.idle_unload_minutes": Option(
        0, "Unload the model after this many idle minutes, 0 keeps it loaded", minimum=0
    ),
    "index.chunk_chars": Option(1600, "Target characters per text chunk", minimum=200),
    "index.respect_gitignore": Option(True, "Apply .gitignore files"),
    "index.follow_symlinks": Option(False, "Follow symbolic links while scanning"),
    "limits.text_mb": Option(5, "Warn before embedding text files above this size", minimum=0),
    "limits.image_mb": Option(25, "Warn before embedding images above this size", minimum=0),
    "limits.video_mb": Option(100, "Warn before embedding videos above this size", minimum=0),
    "limits.audio_mb": Option(100, "Warn before embedding audio above this size", minimum=0),
    "image.tokens": Option(280, "Vision tokens per image", (70, 140, 280, 560, 1120)),
    "video.fps": Option(1.0, "Frames sampled per second of video", minimum=0.05),
    "video.segment_seconds": Option(30, "Length of each video segment", minimum=2),
    "video.frame_tokens": Option(140, "Vision tokens per video frame", (70, 140, 280)),
    "video.audio_track": Option(True, "Also embed the soundtrack of videos"),
    "audio.segment_seconds": Option(30, "Length of each audio segment", minimum=2),
    "search.limit": Option(10, "Default number of results", minimum=1),
    "search.mode": Option("hybrid", "Default search mode", ("hybrid", "vector", "keyword")),
    "search.per_file": Option(2, "Maximum results from one file, 0 for no cap", minimum=0),
}

PRESETS: dict[str, dict[str, Any]] = {
    "quality": {"image.tokens": 280, "video.frame_tokens": 140, "video.fps": 1.0},
    "balanced": {"image.tokens": 140, "video.frame_tokens": 70, "video.fps": 0.5},
    "fast": {"image.tokens": 70, "video.frame_tokens": 70, "video.fps": 0.25},
}


class ConfigError(ValueError):
    pass


def _coerce(key: str, value: Any) -> Any:
    option = OPTIONS[key]
    default = option.default
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "on", "1"):
                value = True
            elif lowered in ("false", "no", "off", "0"):
                value = False
        if not isinstance(value, bool):
            raise ConfigError(f"{key} expects true or false")
    elif isinstance(default, list):
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if not isinstance(value, list):
            raise ConfigError(f"{key} expects a comma separated list")
        unknown = [item for item in value if item not in (option.choices or ())]
        if unknown:
            raise ConfigError(f"{key}: unknown value {unknown[0]!r}, choose from {list(option.choices)}")
        ordered = [item for item in option.choices if item in value]
        if key == "model.capabilities" and "text" not in ordered:
            ordered.insert(0, "text")
        return ordered
    elif isinstance(default, int):
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{key} expects a whole number") from None
    elif isinstance(default, float):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{key} expects a number") from None
    else:
        value = str(value)
    if option.choices and not isinstance(default, list) and value not in option.choices:
        raise ConfigError(f"{key} must be one of {list(option.choices)}")
    if option.minimum is not None and not isinstance(value, bool) and value < option.minimum:
        raise ConfigError(f"{key} must be at least {option.minimum}")
    return value


class Config:
    def __init__(self, values: dict[str, Any] | None = None):
        self._values = {key: copy.deepcopy(option.default) for key, option in OPTIONS.items()}
        for key, value in (values or {}).items():
            self.set(key, value)

    def get(self, key: str) -> Any:
        if key not in OPTIONS:
            raise ConfigError(f"unknown setting {key!r}")
        return self._values[key]

    def set(self, key: str, value: Any) -> Any:
        if key not in OPTIONS:
            raise ConfigError(f"unknown setting {key!r}")
        self._values[key] = _coerce(key, value)
        return self._values[key]

    def has(self, capability: str) -> bool:
        return capability in self._values["model.capabilities"]

    def flat(self) -> dict[str, Any]:
        return dict(self._values)

    def nested(self) -> dict[str, dict[str, Any]]:
        tree: dict[str, dict[str, Any]] = {}
        for key, value in self._values.items():
            section, name = key.split(".", 1)
            tree.setdefault(section, {})[name] = value
        return tree

    @classmethod
    def load(cls, path: Path) -> Config:
        if not path.exists():
            return cls()
        with path.open("rb") as handle:
            tree = tomllib.load(handle)
        config = cls()
        for section, entries in tree.items():
            if not isinstance(entries, dict):
                continue
            for name, value in entries.items():
                key = f"{section}.{name}"
                if key in OPTIONS:
                    config.set(key, value)
        return config

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            tomli_w.dump(self.nested(), handle)
        temporary.replace(path)
