from __future__ import annotations

import pytest

from semantix.config import Config, ConfigError
from semantix.paths import Paths, display_path


def test_defaults_and_round_trip(tmp_path):
    config = Config()
    assert config.get("model.dimension") == 768
    config.set("model.capabilities", "image")
    config.set("daemon.auto_start", "off")
    config.set("video.fps", "0.5")
    config.save(tmp_path / "config.toml")
    loaded = Config.load(tmp_path / "config.toml")
    assert loaded.get("model.capabilities") == ["text", "image"]
    assert loaded.get("daemon.auto_start") is False
    assert loaded.get("video.fps") == 0.5
    assert loaded.has("image") and not loaded.has("audio")


@pytest.mark.parametrize(
    ("key", "value"),
    [("model.dimension", 300), ("model.capabilities", "text,smell"), ("search.mode", "fuzzy"),
     ("index.chunk_chars", 10), ("nope.key", 1), ("daemon.auto_start", "maybe")],
)
def test_invalid_values_are_rejected(key, value):
    with pytest.raises(ConfigError):
        Config().set(key, value)


def test_long_home_gets_a_short_socket_path(tmp_path):
    deep = tmp_path / ("x" * 90)
    paths = Paths(deep)
    assert len(str(paths.socket)) <= 100
    assert Paths(tmp_path).socket == tmp_path / "run" / "daemon.sock" or len(str(tmp_path)) > 80


def test_display_path():
    assert display_path("/work/app/src/main.py", cwd="/work/app") == "src/main.py"
    assert display_path("/other/place", cwd="/work/app") == "/other/place"
