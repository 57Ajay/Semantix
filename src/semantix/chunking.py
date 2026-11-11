from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

REPLACEMENT_LIMIT = 0.02


@dataclass(frozen=True)
class TextChunk:
    start_line: int
    end_line: int
    text: str

    @property
    def digest(self) -> str:
        return hashlib.blake2b(self.text.encode("utf-8", "replace"), digest_size=16).hexdigest()


class NotTextError(ValueError):
    pass


def read_text(path: str) -> str:
    with open(path, "rb") as handle:
        raw = handle.read()
    if b"\x00" in raw[:8192]:
        raise NotTextError("file looks binary")
    text = raw.decode("utf-8", errors="replace")
    if text and text.count("�") / len(text) > REPLACEMENT_LIMIT:
        raise NotTextError("file is not valid UTF-8 text")
    if path.lower().endswith(".ipynb"):
        return _notebook_source(text)
    return text


def _notebook_source(text: str) -> str:
    try:
        notebook = json.loads(text)
    except ValueError:
        return text
    parts = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            parts.append(source.rstrip())
    return "\n\n".join(parts)


def _is_boundary(lines: list[str], index: int) -> bool:
    line = lines[index]
    if not line or line[0] in " \t":
        return False
    if line.startswith("#") and len(line) > 1 and line.lstrip("#")[:1] == " ":
        return True
    if line[0] in ")}]":
        return False
    return index == 0 or not lines[index - 1].strip()


def _split_long_line(line: str, budget: int) -> list[str]:
    return [line[start : start + budget] for start in range(0, len(line), budget)]


def chunk_text(text: str, budget: int = 1600) -> list[TextChunk]:
    lines = text.splitlines()
    if not lines:
        return []
    minimum = budget // 2
    chunks: list[TextChunk] = []
    current: list[str] = []
    current_size = 0
    start = 0

    def flush(end: int) -> None:
        nonlocal current, current_size
        body = "\n".join(current).strip("\n")
        if body.strip():
            chunks.append(TextChunk(start + 1, end, body))
        current = []
        current_size = 0

    for index, line in enumerate(lines):
        if current and current_size >= minimum and _is_boundary(lines, index):
            flush(index)
        if len(line) > budget:
            if current:
                flush(index)
            for piece in _split_long_line(line, budget):
                if piece.strip():
                    chunks.append(TextChunk(index + 1, index + 1, piece))
            start = index + 1
            continue
        if current and current_size + len(line) + 1 > budget:
            cut = _soft_cut(current, minimum)
            if cut:
                tail = current[cut:]
                current = current[:cut]
                flush(index - len(tail))
                start = index - len(tail)
                current = tail
                current_size = sum(len(item) + 1 for item in tail)
            else:
                flush(index)
        if not current:
            start = index
        current.append(line)
        current_size += len(line) + 1
    flush(len(lines))
    return chunks


def _soft_cut(lines: list[str], minimum: int) -> int:
    size = 0
    best = 0
    for position, line in enumerate(lines):
        size += len(line) + 1
        if size >= minimum and not line.strip() and position + 1 < len(lines):
            best = position + 1
    return best


def document_prompt(title: str, body: str) -> str:
    return f"title: {title or 'none'} | text: {body}"
