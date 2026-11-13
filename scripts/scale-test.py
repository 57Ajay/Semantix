"""Measure index size, start-up time, memory and search latency for large collections.

Builds stores of synthetic parts (random unit vectors, Zipf-distributed text) without the
model, then opens them with the engine and times searches in every mode.

    uv run python scripts/scale-test.py 10000 50000 100000
"""

from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from semantix.config import Config
from semantix.engine import Engine
from semantix.paths import Paths
from semantix.store import ChunkRecord, Store

DIMENSION = 768
PARTS_PER_FILE = 10
WORDS_PER_PART = 220
VOCABULARY = 30000
QUERIES = 40


class RandomEmbedder:
    device = "cpu"
    dtype = "float32"
    loaded = True

    def __init__(self, seed: int):
        self.random = np.random.default_rng(seed)
        self.dimension = DIMENSION

    def _vector(self) -> np.ndarray:
        vector = self.random.standard_normal(DIMENSION).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def embed_query(self, text: str) -> dict[str, np.ndarray]:
        return {"search": self._vector(), "code": self._vector()}

    def text_batch_size(self) -> int:
        return 4

    def media_batch_size(self) -> int:
        return 2

    def unload(self) -> None:
        pass


def rss_mb() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build(home: Path, parts: int, random: np.random.Generator, words: list[str], weights: np.ndarray) -> float:
    paths = Paths(home).ensure()
    store = Store(paths.db)
    root = store.add_root("/data", True, {})
    started = time.perf_counter()
    kinds = ("code", "text", "code", "text", "image")
    for file_index in range(parts // PARTS_PER_FILE):
        kind = kinds[file_index % len(kinds)]
        path = f"/data/dir{file_index % 200}/file{file_index}.{'py' if kind == 'code' else 'md' if kind == 'text' else 'jpg'}"
        file_id = store.upsert_file(root.id, path, kind, 16000, file_index, f"h{file_index}", "indexed")
        vectors = random.standard_normal((PARTS_PER_FILE, DIMENSION)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        records = []
        for ordinal in range(PARTS_PER_FILE):
            text = None
            if kind != "image":
                text = " ".join(random.choice(words, WORDS_PER_PART, p=weights))
            records.append(ChunkRecord(kind, vectors[ordinal], ordinal, ordinal * 40 + 1, ordinal * 40 + 40, text=text))
        store.replace_chunks(file_id, path, records)
    store.set_meta("model_id", Config().get("model.id"))
    store.close()
    return time.perf_counter() - started


def measure(home: Path, words: list[str]) -> dict:
    before = rss_mb()
    started = time.perf_counter()
    engine = Engine(Paths(home), Config(), RandomEmbedder(1))
    engine.ensure_anchors()
    load_seconds = time.perf_counter() - started
    loaded = rss_mb()
    common = words[:3]
    rare = words[-200:]
    random = np.random.default_rng(2)
    queries = {
        "vector": lambda i: ("an unrelated question about something", "vector"),
        "hybrid, common words": lambda i: (" ".join(common), "hybrid"),
        "hybrid, rare words": lambda i: (" ".join(random.choice(rare, 3)), "hybrid"),
        "keyword, rare words": lambda i: (" ".join(random.choice(rare, 3)), "keyword"),
    }
    timings = {}
    for name, make in queries.items():
        samples = []
        for index in range(QUERIES):
            text, mode = make(index)
            began = time.perf_counter()
            engine.search(text, mode=mode, limit=10)
            samples.append((time.perf_counter() - began) * 1000)
        timings[name] = round(statistics.median(samples), 1)
    began = time.perf_counter()
    for _ in range(QUERIES):
        engine.search("an unrelated question", mode="vector", limit=10, under="/data/dir7")
    timings["vector, inside one folder"] = round((time.perf_counter() - began) * 1000 / QUERIES, 1)
    engine.close()
    return {
        "load_seconds": round(load_seconds, 2),
        "rss_after_load_mb": round(loaded - before, 1),
        "search_ms": timings,
    }


def main() -> None:
    sizes = [int(value) for value in sys.argv[1:]] or [10000, 50000, 100000]
    random = np.random.default_rng(0)
    words = [f"w{index}" for index in range(VOCABULARY)]
    weights = 1.0 / np.arange(1, VOCABULARY + 1)
    weights /= weights.sum()
    results = []
    for parts in sizes:
        with tempfile.TemporaryDirectory(prefix="semantix-scale-") as folder:
            home = Path(folder)
            build_seconds = build(home, parts, random, words, weights)
            db = Paths(home).db
            size_mb = sum(os.path.getsize(str(db) + suffix) for suffix in ("", "-wal") if os.path.exists(str(db) + suffix))
            result = {"parts": parts, "build_seconds": round(build_seconds, 1), "db_mb": round(size_mb / 1048576, 1)}
            result.update(measure(home, words))
            result["peak_rss_mb"] = round(peak_mb(), 1)
            results.append(result)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
