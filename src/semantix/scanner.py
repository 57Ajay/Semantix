from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field
from typing import Any

from .filetypes import KIND_CAPABILITY, classify, extension_of
from .ignore import IgnoreRules

FULL_HASH_LIMIT = 64 * 1024 * 1024
SAMPLE_BLOCK = 1024 * 1024


@dataclass(frozen=True)
class Candidate:
    path: str
    kind: str
    size: int
    mtime_ns: int


@dataclass
class ScanResult:
    candidates: list[Candidate] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    off_kinds: dict[str, int] = field(default_factory=dict)
    off_paths: set[str] = field(default_factory=set)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def normalize_selection(selection: dict[str, Any] | None) -> dict[str, Any]:
    selection = selection or {}
    kinds = selection.get("kinds")
    extensions = []
    for extension in selection.get("skip_ext") or []:
        extension = extension.strip().lower()
        if extension:
            extensions.append(extension if extension.startswith(".") else "." + extension)
    return {
        "kinds": sorted(set(kinds)) if kinds else None,
        "skip_ext": sorted(set(extensions)),
        "exclude": list(dict.fromkeys(selection.get("exclude") or [])),
    }


def file_hash(path: str, size: int) -> str:
    digest = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as handle:
        if size <= FULL_HASH_LIMIT:
            while block := handle.read(SAMPLE_BLOCK):
                digest.update(block)
            return digest.hexdigest()
        digest.update(str(size).encode())
        for offset in (0, size // 2, max(size - SAMPLE_BLOCK, 0)):
            handle.seek(offset)
            digest.update(handle.read(SAMPLE_BLOCK))
    return "s:" + digest.hexdigest()


class Scanner:
    def __init__(
        self,
        rules: IgnoreRules,
        selection: dict[str, Any],
        capabilities: tuple[str, ...],
        follow_symlinks: bool = False,
    ):
        self.rules = rules
        self.kinds = set(selection["kinds"]) if selection.get("kinds") else None
        self.skip_ext = set(selection.get("skip_ext") or [])
        self.capabilities = set(capabilities)
        self.follow_symlinks = follow_symlinks

    def candidate(self, path: str, result: ScanResult, explicit: bool = False) -> Candidate | None:
        try:
            info = os.stat(path)
        except OSError:
            result.skip("unreadable")
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        if not explicit and extension_of(path) in self.skip_ext:
            result.skip("skipped extension")
            return None
        kind = classify(path)
        if kind is None:
            result.skip("unsupported type")
            return None
        if not explicit and self.kinds is not None and kind not in self.kinds:
            result.skip("type not selected")
            return None
        if KIND_CAPABILITY[kind] not in self.capabilities:
            result.off_kinds[kind] = result.off_kinds.get(kind, 0) + 1
            result.off_paths.add(path)
            return None
        return Candidate(path, kind, info.st_size, info.st_mtime_ns)

    def scan(self, scope: str, pinned: set[str] | None = None) -> ScanResult:
        result = ScanResult()
        pinned = pinned or set()
        seen: set[str] = set()
        if os.path.isdir(scope):
            self._walk(scope, result, seen, pinned)
        for path in sorted(pinned):
            if path in seen or not os.path.isfile(path):
                continue
            if path == scope or path.startswith(scope.rstrip("/") + "/"):
                found = self.candidate(path, result, explicit=True)
                if found:
                    result.candidates.append(found)
                    seen.add(path)
        return result

    def _walk(self, top: str, result: ScanResult, seen: set[str], pinned: set[str]) -> None:
        stack = [top]
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            except OSError:
                result.skip("unreadable")
                continue
            for entry in entries:
                try:
                    is_link = entry.is_symlink()
                    if is_link and not self.follow_symlinks:
                        continue
                    is_dir = entry.is_dir(follow_symlinks=self.follow_symlinks)
                except OSError:
                    continue
                path = entry.path
                if is_dir:
                    if self.rules.check(path, True).ignored:
                        result.skip("ignored directory")
                    else:
                        stack.append(path)
                    continue
                explicit = path in pinned
                if not explicit and self.rules.check(path, False).ignored:
                    result.skip("ignored")
                    continue
                found = self.candidate(path, result, explicit=explicit)
                if found:
                    result.candidates.append(found)
                    seen.add(path)
