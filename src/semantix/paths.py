from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

SOCKET_PATH_LIMIT = 100


def home() -> Path:
    override = os.environ.get("SEMANTIX_HOME")
    return Path(override).expanduser().resolve() if override else Path.home() / ".semantix"


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def config(self) -> Path:
        return self.home / "config.toml"

    @property
    def ignore(self) -> Path:
        return self.home / "ignore"

    @property
    def index_dir(self) -> Path:
        return self.home / "index"

    @property
    def db(self) -> Path:
        return self.index_dir / "semantix.db"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def run_dir(self) -> Path:
        return self.home / "run"

    @property
    def pid(self) -> Path:
        return self.run_dir / "daemon.pid"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def log(self) -> Path:
        return self.logs_dir / "daemon.log"

    @property
    def socket(self) -> Path:
        preferred = self.run_dir / "daemon.sock"
        if len(str(preferred)) <= SOCKET_PATH_LIMIT:
            return preferred
        digest = hashlib.blake2b(str(self.home).encode(), digest_size=6).hexdigest()
        return Path(tempfile.gettempdir()) / f"semantix-{os.getuid()}-{digest}.sock"

    def ensure(self) -> Paths:
        for directory in (self.home, self.index_dir, self.models, self.run_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    return Paths(home())


def display_path(path: str, cwd: str | None = None) -> str:
    cwd = cwd or os.getcwd()
    if path != cwd and path.startswith(cwd.rstrip("/") + "/"):
        return path[len(cwd.rstrip("/")) + 1 :]
    user_home = str(Path.home())
    if path == user_home or path.startswith(user_home + "/"):
        return "~" + path[len(user_home) :]
    return path
