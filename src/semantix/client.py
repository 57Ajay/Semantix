from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any

from .paths import Paths

START_TIMEOUT = 600


class DaemonUnavailable(RuntimeError):
    pass


class RemoteError(RuntimeError):
    pass


def _connect(paths: Paths, timeout: float | None) -> socket.socket:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        connection.connect(str(paths.socket))
    except OSError as error:
        connection.close()
        raise DaemonUnavailable("the semantix service is not running") from error
    return connection


def stream(paths: Paths, op: str, args: dict[str, Any] | None = None, timeout: float | None = None) -> Iterator[dict[str, Any]]:
    connection = _connect(paths, timeout)
    try:
        connection.sendall(json.dumps({"op": op, "args": args or {}}).encode() + b"\n")
        with connection.makefile("rb") as reader:
            for line in reader:
                message = json.loads(line)
                if "ok" in message:
                    if not message["ok"]:
                        raise RemoteError(message.get("error") or "request failed")
                    yield {"event": "result", "result": message.get("result")}
                    return
                yield message
        raise DaemonUnavailable("the semantix service closed the connection")
    finally:
        connection.close()


def request(paths: Paths, op: str, args: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
    for message in stream(paths, op, args, timeout):
        if message["event"] == "result":
            return message["result"]
    raise DaemonUnavailable("the semantix service gave no answer")


def ping(paths: Paths) -> dict[str, Any] | None:
    try:
        return request(paths, "ping", timeout=3.0)
    except (DaemonUnavailable, RemoteError, OSError, ValueError):
        return None


def spawn(paths: Paths) -> subprocess.Popen:
    paths.ensure()
    with open(paths.logs_dir / "daemon.out", "ab") as output:
        return subprocess.Popen(
            [sys.executable, "-m", "semantix", "start", "--foreground"],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
            close_fds=True,
        )


def start(paths: Paths, on_wait: Callable[[str], None] | None = None) -> dict[str, Any]:
    state = ping(paths)
    process = None
    if state is None:
        process = spawn(paths)
    deadline = time.time() + START_TIMEOUT
    while time.time() < deadline:
        state = ping(paths)
        if state is not None:
            if state["model_state"] in ("ready", "unloaded"):
                return state
            if state["model_state"] == "error":
                stop(paths)
                raise RemoteError(state.get("model_error") or "the model failed to load")
            if on_wait:
                on_wait("loading the model")
        elif process is not None and process.poll() is not None:
            raise DaemonUnavailable(f"the service exited while starting, see {paths.log}")
        elif on_wait:
            on_wait("starting")
        time.sleep(0.2)
    raise DaemonUnavailable("the service took too long to start")


def stop(paths: Paths, wait: float = 15.0) -> bool:
    state = ping(paths)
    if state is None:
        return False
    try:
        request(paths, "shutdown", timeout=5.0)
    except (DaemonUnavailable, RemoteError, OSError):
        pass
    deadline = time.time() + wait
    while time.time() < deadline:
        if ping(paths) is None:
            settle = time.time() + 5.0
            while paths.socket.exists() and time.time() < settle:
                time.sleep(0.05)
            return True
        time.sleep(0.1)
    try:
        os.kill(int(state["pid"]), signal.SIGTERM)
    except (OSError, ValueError, KeyError):
        pass
    return True
