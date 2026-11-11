from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS roots (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    is_dir INTEGER NOT NULL,
    selection TEXT NOT NULL DEFAULT '{}',
    added_at REAL NOT NULL,
    synced_at REAL
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    root_id INTEGER NOT NULL REFERENCES roots(id),
    path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    hash TEXT,
    status TEXT NOT NULL,
    error TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL
);
CREATE INDEX IF NOT EXISTS files_root ON files(root_id);
CREATE INDEX IF NOT EXISTS files_hash ON files(hash);
CREATE INDEX IF NOT EXISTS files_status ON files(status);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    ord INTEGER NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    start_sec REAL,
    end_sec REAL,
    hash TEXT,
    text TEXT,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(path, body, tokenize = 'unicode61');
"""

STATUS_INDEXED = "indexed"
STATUS_PENDING = "pending"
STATUS_STALE = "stale"
STATUS_ERROR = "error"

WORD = re.compile(r"[\w]+", re.UNICODE)


@dataclass
class Root:
    id: int
    path: str
    is_dir: bool
    selection: dict[str, Any]
    added_at: float
    synced_at: float | None


@dataclass
class FileRow:
    id: int
    root_id: int
    path: str
    kind: str
    size: int
    mtime_ns: int
    hash: str | None
    status: str
    error: str | None
    pinned: bool


@dataclass
class ChunkRecord:
    kind: str
    vector: np.ndarray
    ord: int = 0
    start_line: int | None = None
    end_line: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    hash: str | None = None
    text: str | None = None


def path_prefix_clause(column: str) -> str:
    return f"({column} = ? OR substr({column}, 1, ?) = ?)"


def path_prefix_args(path: str) -> tuple:
    prefix = path.rstrip("/") + "/"
    return (path, len(prefix), prefix)


def fts_query(text: str) -> str:
    words = [word for word in WORD.findall(text) if len(word) > 1]
    seen = []
    for word in words:
        lowered = word.lower()
        if lowered not in seen:
            seen.append(lowered)
    return " OR ".join(f'"{word}"' for word in seen[:32])


class Store:
    def __init__(self, path: Path, readonly: bool = False):
        self.path = path
        self._lock = threading.RLock()
        if readonly:
            self._db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = NORMAL")
            with self._db:
                self._db.executescript(SCHEMA)
            if self.get_meta("schema_version") is None:
                self.set_meta("schema_version", str(SCHEMA_VERSION))
        self._db.row_factory = sqlite3.Row

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(sql, args).fetchone()

    def _run(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock, self._db:
            return self._db.execute(sql, args)

    def get_meta(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key = ?", (key,))
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._run("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def get_stat(self, key: str) -> float | None:
        row = self._one("SELECT value FROM stats WHERE key = ?", (key,))
        return float(row[0]) if row else None

    def update_stat(self, key: str, value: float, weight: float = 0.5) -> None:
        current = self.get_stat(key)
        blended = value if current is None else current * (1 - weight) + value * weight
        self._run("INSERT INTO stats(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, blended))

    @staticmethod
    def _root(row: sqlite3.Row) -> Root:
        return Root(row["id"], row["path"], bool(row["is_dir"]), json.loads(row["selection"]), row["added_at"], row["synced_at"])

    def roots(self) -> list[Root]:
        return [self._root(row) for row in self._all("SELECT * FROM roots ORDER BY path")]

    def root_by_path(self, path: str) -> Root | None:
        row = self._one("SELECT * FROM roots WHERE path = ?", (path,))
        return self._root(row) if row else None

    def root_containing(self, path: str) -> Root | None:
        best = None
        for root in self.roots():
            inside = path == root.path or (root.is_dir and path.startswith(root.path.rstrip("/") + "/"))
            if inside and (best is None or len(root.path) > len(best.path)):
                best = root
        return best

    def roots_under(self, path: str) -> list[Root]:
        prefix = path.rstrip("/") + "/"
        return [root for root in self.roots() if root.path.startswith(prefix)]

    def add_root(self, path: str, is_dir: bool, selection: dict[str, Any]) -> Root:
        self._run(
            "INSERT INTO roots(path, is_dir, selection, added_at) VALUES(?, ?, ?, ?)",
            (path, int(is_dir), json.dumps(selection), time.time()),
        )
        return self.root_by_path(path)

    def update_root_selection(self, root_id: int, selection: dict[str, Any]) -> None:
        self._run("UPDATE roots SET selection = ? WHERE id = ?", (json.dumps(selection), root_id))

    def touch_root(self, root_id: int) -> None:
        self._run("UPDATE roots SET synced_at = ? WHERE id = ?", (time.time(), root_id))

    def merge_root(self, child_id: int, parent_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE files SET root_id = ? WHERE root_id = ?", (parent_id, child_id))
            self._db.execute("DELETE FROM roots WHERE id = ?", (child_id,))

    def delete_root(self, root_id: int) -> list[int]:
        file_ids = [row[0] for row in self._all("SELECT id FROM files WHERE root_id = ?", (root_id,))]
        self.delete_files(file_ids)
        self._run("DELETE FROM roots WHERE id = ?", (root_id,))
        return file_ids

    @staticmethod
    def _file(row: sqlite3.Row) -> FileRow:
        return FileRow(
            row["id"], row["root_id"], row["path"], row["kind"], row["size"], row["mtime_ns"],
            row["hash"], row["status"], row["error"], bool(row["pinned"]),
        )

    def file_by_path(self, path: str) -> FileRow | None:
        row = self._one("SELECT * FROM files WHERE path = ?", (path,))
        return self._file(row) if row else None

    def files_under(self, path: str) -> list[FileRow]:
        rows = self._all(f"SELECT * FROM files WHERE {path_prefix_clause('path')}", path_prefix_args(path))
        return [self._file(row) for row in rows]

    def file_ids_under(self, path: str) -> np.ndarray:
        rows = self._all(f"SELECT id FROM files WHERE {path_prefix_clause('path')}", path_prefix_args(path))
        return np.asarray([row[0] for row in rows], dtype=np.int64)

    def files_with_status(self, statuses: tuple[str, ...]) -> list[FileRow]:
        marks = ",".join("?" * len(statuses))
        return [self._file(row) for row in self._all(f"SELECT * FROM files WHERE status IN ({marks})", statuses)]

    def upsert_file(
        self, root_id: int, path: str, kind: str, size: int, mtime_ns: int, hash: str | None,
        status: str, error: str | None = None, pinned: bool | None = None,
    ) -> int:
        with self._lock, self._db:
            existing = self._db.execute("SELECT id, pinned FROM files WHERE path = ?", (path,)).fetchone()
            indexed_at = time.time() if status == STATUS_INDEXED else None
            if existing:
                keep_pinned = existing["pinned"] if pinned is None else int(pinned)
                self._db.execute(
                    "UPDATE files SET root_id = ?, kind = ?, size = ?, mtime_ns = ?, hash = COALESCE(?, hash), "
                    "status = ?, error = ?, pinned = ?, indexed_at = COALESCE(?, indexed_at) WHERE id = ?",
                    (root_id, kind, size, mtime_ns, hash, status, error, keep_pinned, indexed_at, existing["id"]),
                )
                return existing["id"]
            cursor = self._db.execute(
                "INSERT INTO files(root_id, path, kind, size, mtime_ns, hash, status, error, pinned, indexed_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (root_id, path, kind, size, mtime_ns, hash, status, error, int(bool(pinned)), indexed_at),
            )
            return cursor.lastrowid

    def set_file_status(self, file_id: int, status: str, error: str | None = None) -> None:
        self._run("UPDATE files SET status = ?, error = ? WHERE id = ?", (status, error, file_id))

    def touch_file(self, file_id: int, size: int, mtime_ns: int) -> None:
        self._run("UPDATE files SET size = ?, mtime_ns = ? WHERE id = ?", (size, mtime_ns, file_id))

    def move_file(self, file_id: int, root_id: int, path: str, size: int, mtime_ns: int) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE files SET root_id = ?, path = ?, size = ?, mtime_ns = ? WHERE id = ?",
                (root_id, path, size, mtime_ns, file_id),
            )
            self._db.execute(
                "UPDATE chunks_fts SET path = ? WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)",
                (path, file_id),
            )

    def delete_files(self, file_ids: list[int]) -> None:
        with self._lock, self._db:
            for file_id in file_ids:
                self._delete_chunks(file_id)
                self._db.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def _delete_chunks(self, file_id: int) -> None:
        self._db.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)", (file_id,))
        self._db.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    def chunk_vectors_by_hash(self, file_id: int) -> dict[str, np.ndarray]:
        rows = self._all("SELECT hash, vec FROM chunks WHERE file_id = ? AND hash IS NOT NULL", (file_id,))
        return {row["hash"]: np.frombuffer(row["vec"], dtype=np.float32) for row in rows}

    def replace_chunks(self, file_id: int, path: str, records: list[ChunkRecord]) -> list[int]:
        with self._lock, self._db:
            self._delete_chunks(file_id)
            ids = []
            for record in records:
                cursor = self._db.execute(
                    "INSERT INTO chunks(file_id, ord, kind, start_line, end_line, start_sec, end_sec, hash, text, vec) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        file_id, record.ord, record.kind, record.start_line, record.end_line,
                        record.start_sec, record.end_sec, record.hash, record.text,
                        np.asarray(record.vector, dtype=np.float32).tobytes(),
                    ),
                )
                ids.append(cursor.lastrowid)
                self._db.execute(
                    "INSERT INTO chunks_fts(rowid, path, body) VALUES(?, ?, ?)",
                    (cursor.lastrowid, path, record.text or ""),
                )
            return ids

    def load_vectors(self, dimension: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        with self._lock:
            count = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            ids = np.zeros(count, dtype=np.int64)
            files = np.zeros(count, dtype=np.int64)
            kinds: list[str] = []
            vectors = np.zeros((count, dimension), dtype=np.float32)
            cursor = self._db.execute(
                "SELECT c.id, c.file_id, f.kind, c.vec FROM chunks c JOIN files f ON f.id = c.file_id ORDER BY c.id"
            )
            for position, row in enumerate(cursor):
                ids[position] = row[0]
                files[position] = row[1]
                kinds.append(row[2])
                vectors[position] = np.frombuffer(row[3], dtype=np.float32)[:dimension]
            return ids, files, kinds, vectors

    def stored_dimension(self) -> int | None:
        row = self._one("SELECT length(vec) FROM chunks LIMIT 1")
        return row[0] // 4 if row else None

    def truncate_vectors(self, dimension: int) -> None:
        with self._lock, self._db:
            rows = self._db.execute("SELECT id, vec FROM chunks").fetchall()
            for row in rows:
                vector = np.frombuffer(row[1], dtype=np.float32)[:dimension].copy()
                norm = float(np.linalg.norm(vector))
                if norm:
                    vector /= norm
                self._db.execute("UPDATE chunks SET vec = ? WHERE id = ?", (vector.tobytes(), row[0]))

    def reset_embeddings(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks_fts")
            self._db.execute("DELETE FROM chunks")
            self._db.execute("UPDATE files SET status = ?, error = NULL", (STATUS_PENDING,))

    def chunk_details(self, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self._all(
            "SELECT c.id, c.kind, c.start_line, c.end_line, c.start_sec, c.end_sec, c.text, c.file_id, "
            "f.path, f.kind AS file_kind, f.status FROM chunks c JOIN files f ON f.id = c.file_id "
            f"WHERE c.id IN ({marks})",
            tuple(chunk_ids),
        )
        return {row["id"]: dict(row) for row in rows}

    def keyword_search(
        self, text: str, limit: int, kinds: list[str] | None = None, under: str | None = None
    ) -> list[tuple[int, float]]:
        query = fts_query(text)
        if not query:
            return []
        sql = (
            "SELECT chunks_fts.rowid AS id, bm25(chunks_fts, 3.0, 1.0) AS rank FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid JOIN files f ON f.id = c.file_id "
            "WHERE chunks_fts MATCH ?"
        )
        args: list[Any] = [query]
        if kinds is not None:
            sql += f" AND f.kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        if under is not None:
            sql += f" AND {path_prefix_clause('f.path')}"
            args.extend(path_prefix_args(under))
        sql += " ORDER BY rank LIMIT ?"
        args.append(limit)
        try:
            return [(row["id"], float(row["rank"])) for row in self._all(sql, tuple(args))]
        except sqlite3.OperationalError:
            return []

    def counts(self) -> dict[str, Any]:
        files = {}
        for row in self._all("SELECT kind, status, COUNT(*) AS n, SUM(size) AS bytes FROM files GROUP BY kind, status"):
            entry = files.setdefault(row["kind"], {"files": 0, "bytes": 0, "indexed": 0, "pending": 0, "stale": 0, "error": 0})
            entry["files"] += row["n"]
            entry["bytes"] += row["bytes"] or 0
            entry[row["status"]] += row["n"]
        chunks = {row["kind"]: row["n"] for row in self._all("SELECT kind, COUNT(*) AS n FROM chunks GROUP BY kind")}
        return {"files": files, "chunks": chunks}

    def root_summaries(self) -> list[dict[str, Any]]:
        summaries = []
        for root in self.roots():
            rows = self._all(
                "SELECT kind, status, COUNT(*) AS n FROM files WHERE root_id = ? GROUP BY kind, status", (root.id,)
            )
            kinds: dict[str, int] = {}
            waiting = 0
            errors = 0
            for row in rows:
                kinds[row["kind"]] = kinds.get(row["kind"], 0) + row["n"]
                if row["status"] in (STATUS_PENDING, STATUS_STALE):
                    waiting += row["n"]
                elif row["status"] == STATUS_ERROR:
                    errors += row["n"]
            summaries.append(
                {
                    "path": root.path, "is_dir": root.is_dir, "selection": root.selection,
                    "added_at": root.added_at, "synced_at": root.synced_at,
                    "kinds": kinds, "files": sum(kinds.values()), "waiting": waiting, "errors": errors,
                }
            )
        return summaries

    def error_files(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._all("SELECT path, error FROM files WHERE status = ? ORDER BY path LIMIT ?", (STATUS_ERROR, limit))
        return [dict(row) for row in rows]
