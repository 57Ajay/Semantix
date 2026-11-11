from __future__ import annotations

import pytest
from typer.testing import CliRunner

from semantix import autostart
from semantix.cli import app, clock, estimate, human_bytes, human_time, location_of

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIX_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def test_formatting_helpers():
    assert human_bytes(512) == "512 B" and human_bytes(1536) == "1.5 KB"
    assert human_time(0.2) == "under 1s" and human_time(75) == "1m 15s" and human_time(3700) == "1h 01m"
    assert estimate(0.3) == "under 1s" and estimate(90) == "about 1m 30s"
    assert clock(75) == "01:15" and clock(3725) == "1:02:05"
    assert location_of({"start_line": 4, "end_line": 9, "start_sec": None}) == ":4-9"
    assert location_of({"start_line": None, "start_sec": 30.0, "end_sec": 60.0}) == " @ 00:30–01:00"


def test_config_commands():
    assert runner.invoke(app, ["config", "set", "search.limit", "25"]).exit_code == 0
    assert runner.invoke(app, ["config", "get", "search.limit"]).output.strip() == "25"
    result = runner.invoke(app, ["config", "set", "model.dimension", "999"])
    assert result.exit_code == 1
    assert runner.invoke(app, ["config", "preset", "fast"]).exit_code == 0
    assert runner.invoke(app, ["config", "get", "image.tokens"]).output.strip() == "70"
    assert "model.capabilities" in runner.invoke(app, ["config"]).output


def test_status_and_list_without_a_service():
    status = runner.invoke(app, ["status"])
    assert status.exit_code == 0 and "stopped" in status.output and "not downloaded" in status.output
    assert "Nothing is tracked" in runner.invoke(app, ["list"]).output


def test_search_needs_a_model_before_starting():
    result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code == 1


def test_systemd_unit_and_launchd_plist(tmp_path):
    unit = autostart.systemd_unit()
    assert "-m semantix start --foreground" in unit and "WantedBy=default.target" in unit
    assert "SEMANTIX_HOME" in unit
    plist = autostart.launchd_plist(tmp_path / "daemon.log")
    assert b"com.semantix.daemon" in plist and b"RunAtLoad" in plist


def test_benchmark_helpers():
    import numpy as np

    from semantix import benchmark

    sources = benchmark.stdlib_sources(10_000)
    assert [name for name, _text in sources[:3]] == ["json/decoder.py", "heapq.py", "textwrap.py"]
    assert benchmark.candidate_dtypes("cpu", None) == ["float32", "bfloat16"]
    assert benchmark.candidate_dtypes("mps", None) == ["float32"]
    assert benchmark.candidate_dtypes("cuda", ["float32"]) == ["float32"]
    unit = np.eye(4, dtype=np.float32)
    assert benchmark.agreement({"text": unit}, {"text": unit, "image": unit}) == {"text": 1.0}
    assert len(benchmark.tone(2)) == 32000 and benchmark.shape_image("red", "triangle").size == (1280, 720)
