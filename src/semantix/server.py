from __future__ import annotations

import json
import logging
import logging.handlers
import os
import select
import signal
import socket
import socketserver
import sys
import threading
import time
from typing import Any

from .engine import Engine, EngineError
from .model import ModelError
from .paths import Paths

log = logging.getLogger("semantix.server")

IDLE_CHECK_SECONDS = 30


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32


class Daemon:
    def __init__(self, paths: Paths, engine: Engine | None = None):
        self.paths = paths.ensure()
        self.engine = engine or Engine(paths)
        self.model_state = "loading"
        self.model_error: str | None = None
        self._model_ready = threading.Event()
        self._stopping = threading.Event()
        self._server: _Server | None = None

    def _load_model(self) -> None:
        self.model_state = "loading"
        self.model_error = None
        self._model_ready.clear()
        try:
            self.engine.embedder.load()
            self.engine.ensure_anchors()
            self.model_state = "ready"
        except Exception as error:
            self.model_state = "error"
            self.model_error = str(error) or type(error).__name__
            log.error("model failed to load: %s", self.model_error)
        finally:
            self._model_ready.set()

    def _need_model(self) -> None:
        self._model_ready.wait()
        if self.model_state == "error":
            raise ModelError(self.model_error or "the model failed to load")
        if self.model_state == "unloaded":
            self._load_model()
            if self.model_state == "error":
                raise ModelError(self.model_error or "the model failed to load")

    def _idle_loop(self) -> None:
        while not self._stopping.wait(IDLE_CHECK_SECONDS):
            minutes = self.engine.config.get("daemon.idle_unload_minutes")
            embedder = self.engine.embedder
            if not minutes or self.model_state != "ready" or self.engine.sync_lock.locked():
                continue
            if time.time() - getattr(embedder, "last_used", time.time()) > minutes * 60:
                embedder.unload()
                self.model_state = "unloaded"

    def status(self) -> dict[str, Any]:
        status = self.engine.status()
        status["model"]["state"] = self.model_state
        status["model"]["error"] = self.model_error
        status["socket"] = str(self.paths.socket)
        return status

    def handle(self, op: str, args: dict[str, Any], connection: socket.socket) -> Any:
        engine = self.engine
        if op == "ping":
            return {"pid": os.getpid(), "model_state": self.model_state, "model_error": self.model_error}
        if op == "status":
            return self.status()
        if op == "plan":
            return engine.plan(args.get("targets"), args.get("selection"), bool(args.get("force")))
        if op == "search":
            if args.get("like") or args.get("mode") != "keyword":
                self._need_model()
            return engine.search(
                query=args.get("query"), like=args.get("like"), kinds=args.get("kinds"), under=args.get("under"),
                limit=args.get("limit"), mode=args.get("mode"), per_file=args.get("per_file"),
                min_score=args.get("min_score"),
            )
        if op == "forget":
            return engine.forget(args["path"])
        if op == "check":
            return engine.check(args["path"])
        if op == "reload":
            result = engine.reload_config()
            if result["model_changed"]:
                threading.Thread(target=self._load_model, daemon=True).start()
            return result
        if op == "shutdown":
            threading.Thread(target=self.stop, daemon=True).start()
            return {"stopping": True}
        raise EngineError(f"unknown operation {op!r}")

    def stream_execute(self, args: dict[str, Any], connection: socket.socket, send) -> None:
        self._need_model()
        cancel = threading.Event()
        finished = threading.Event()

        def watch() -> None:
            while not finished.is_set():
                try:
                    readable, _, _ = select.select([connection], [], [], 0.5)
                    if readable and not connection.recv(1, socket.MSG_PEEK):
                        cancel.set()
                        return
                except (OSError, ValueError):
                    cancel.set()
                    return

        threading.Thread(target=watch, daemon=True).start()
        try:
            for event in self.engine.execute(args["plan_id"], args.get("groups"), args.get("skip"), cancel):
                if event["event"] == "done":
                    send({"ok": True, "result": event})
                else:
                    send(event)
        finally:
            finished.set()

    def _handler(self):
        daemon = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                def send(payload: dict[str, Any]) -> None:
                    try:
                        self.wfile.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

                try:
                    line = self.rfile.readline()
                    if not line:
                        return
                    message = json.loads(line)
                    op = message.get("op", "")
                    args = message.get("args") or {}
                    if op == "execute":
                        daemon.stream_execute(args, self.connection, send)
                    else:
                        send({"ok": True, "result": daemon.handle(op, args, self.connection)})
                except (EngineError, ModelError, ValueError, KeyError, OSError) as error:
                    send({"ok": False, "error": str(error) or type(error).__name__})
                except Exception as error:
                    log.exception("request failed")
                    send({"ok": False, "error": f"internal error: {error}"})

        return Handler

    def serve(self) -> None:
        socket_path = self.paths.socket
        if socket_path.exists():
            if _alive(socket_path):
                raise EngineError("semantix is already running")
            socket_path.unlink()
        previous = os.umask(0o077)
        try:
            self._server = _Server(str(socket_path), self._handler())
        finally:
            os.umask(previous)
        self.paths.pid.write_text(str(os.getpid()))
        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        log.info("listening on %s", socket_path)
        try:
            self._server.serve_forever(poll_interval=0.2)
        finally:
            self._cleanup()

    def stop(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        if self._server is not None:
            self._server.shutdown()

    def _cleanup(self) -> None:
        if self._server is not None:
            self._server.server_close()
        for path in (self.paths.socket, self.paths.pid):
            try:
                path.unlink()
            except OSError:
                pass
        self.engine.close()
        log.info("stopped")


def _alive(socket_path) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        probe.connect(str(socket_path))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def configure_logging(paths: Paths, foreground: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = logging.handlers.RotatingFileHandler(paths.log, maxBytes=2_000_000, backupCount=2)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    if foreground and sys.stderr and sys.stderr.isatty():
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    for noisy in ("sentence_transformers", "transformers", "huggingface_hub", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run(paths: Paths) -> None:
    paths.ensure()
    configure_logging(paths, foreground=True)
    daemon = Daemon(paths)

    def on_signal(_signum, _frame) -> None:
        threading.Thread(target=daemon.stop, daemon=True).start()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    daemon.serve()
