from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SERVICE_NAME = "semantix.service"
LAUNCHD_LABEL = "com.semantix.daemon"


class AutostartError(RuntimeError):
    pass


def command() -> list[str]:
    return [sys.executable, "-m", "semantix", "start", "--foreground"]


def environment() -> dict[str, str]:
    variables = {}
    if os.environ.get("SEMANTIX_HOME"):
        variables["SEMANTIX_HOME"] = os.environ["SEMANTIX_HOME"]
    return variables


def systemd_unit_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user" / SERVICE_NAME


def launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def systemd_unit() -> str:
    lines = [
        "[Unit]",
        "Description=Semantix local semantic search service",
        "",
        "[Service]",
        "Type=simple",
        f"ExecStart={shlex.join(command())}",
        "Restart=on-failure",
        "RestartSec=5",
    ]
    for name, value in environment().items():
        lines.append(f"Environment={shlex.quote(f'{name}={value}')}")
    lines += ["", "[Install]", "WantedBy=default.target", ""]
    return "\n".join(lines)


def launchd_plist(log_path: Path) -> bytes:
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": command(),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "ProcessType": "Background",
    }
    if environment():
        payload["EnvironmentVariables"] = environment()
    return plistlib.dumps(payload)


def _run(arguments: list[str]) -> None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AutostartError(f"could not run {' '.join(arguments)}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AutostartError(f"{' '.join(arguments)} failed: {detail}")


def platform() -> str:
    if sys.platform == "darwin":
        return "launchd"
    if sys.platform.startswith("linux"):
        return "systemd"
    raise AutostartError("autostart is supported on Linux (systemd) and macOS (launchd) only")


def target_path() -> Path:
    return launchd_plist_path() if platform() == "launchd" else systemd_unit_path()


def is_enabled() -> bool:
    return target_path().exists()


def enable(log_path: Path) -> Path:
    kind = platform()
    path = target_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "launchd":
        path.write_bytes(launchd_plist(log_path))
        subprocess.run(["launchctl", "unload", str(path)], capture_output=True)
        _run(["launchctl", "load", "-w", str(path)])
        return path
    path.write_text(systemd_unit())
    if shutil.which("systemctl") is None:
        raise AutostartError(f"wrote {path} but systemctl was not found, enable it manually")
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", SERVICE_NAME])
    return path


def disable() -> Path | None:
    kind = platform()
    path = target_path()
    if not path.exists():
        return None
    if kind == "launchd":
        subprocess.run(["launchctl", "unload", "-w", str(path)], capture_output=True)
    elif shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "disable", SERVICE_NAME], capture_output=True)
    path.unlink()
    if kind == "systemd" and shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    return path
