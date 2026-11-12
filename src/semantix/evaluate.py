from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import shutil
import statistics
import tarfile
import tempfile
import time
import tomllib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import Config
from .filetypes import KINDS, classify

SET_ARCHIVE = "https://codeload.github.com/57Ajay/Semantix/tar.gz/refs/heads/eval"
SET_SOURCE = "github.com/57Ajay/Semantix, branch eval"
SPLITS = ("dev", "test")
MODES = ("hybrid", "vector", "keyword")
METRICS = ("hit@1", "recall@5", "recall@10", "mrr", "ndcg@10")
GRADES = {"relevant": 1.0, "acceptable": 0.5, "not_relevant": 0.0}
DEPTH = 50
CUTOFF = 10
UNJUDGED_DEPTH = 5
BASE_SETTINGS = ("model.id", "model.dimension", "model.capabilities", "model.device", "model.dtype")
KIND_SETTINGS = {
    "index.chunk_chars": ("code", "text"),
    "image.tokens": ("image",),
    "video.fps": ("video",),
    "video.segment_seconds": ("video",),
    "video.frame_tokens": ("video",),
    "video.audio_track": ("video",),
    "audio.segment_seconds": ("audio",),
}
EMBEDDING_SETTINGS = (*BASE_SETTINGS[:3], *KIND_SETTINGS)


class EvalError(RuntimeError):
    pass


@dataclass(frozen=True)
class Judgment:
    path: str
    grade: float
    lines: tuple[int, int] | None = None
    seconds: tuple[float, float] | None = None

    def label(self) -> str:
        return self.path + span_label(self.lines, self.seconds)


@dataclass
class Query:
    id: str
    category: str
    split: str
    text: str | None = None
    like: str | None = None
    kinds: list[str] | None = None
    judgments: tuple[Judgment, ...] = ()

    @property
    def relevant(self) -> list[Judgment]:
        return [judgment for judgment in self.judgments if judgment.grade == GRADES["relevant"]]

    @property
    def answered(self) -> bool:
        return bool(self.relevant)


@dataclass
class EvalSet:
    root: Path
    name: str
    version: str
    queries: list[Query]

    @property
    def corpus(self) -> Path:
        return self.root / "corpus"


def span_label(lines: tuple[int, int] | None, seconds: tuple[float, float] | None) -> str:
    if lines:
        return f":{lines[0]}-{lines[1]}"
    if seconds:
        return f"@{seconds[0]:g}-{seconds[1]:g}s"
    return ""


def _span(value: Any, where: str, kind: type) -> tuple | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, int | float) for item in value):
        raise EvalError(f"{where}: a range must be two numbers")
    start, end = (kind(item) for item in value)
    if end < start:
        raise EvalError(f"{where}: the range ends before it starts")
    return start, end


def _judgments(entry: dict[str, Any], where: str, corpus: Path) -> tuple[Judgment, ...]:
    result = []
    seen = set()
    for field, grade in GRADES.items():
        items = entry.get(field, [])
        if not isinstance(items, list):
            raise EvalError(f"{where}: {field} must be a list")
        for position, item in enumerate(items):
            spot = f"{where} {field}[{position}]"
            if isinstance(item, str):
                item = {"path": item}
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise EvalError(f"{spot}: needs a path")
            unknown = set(item) - {"path", "lines", "seconds"}
            if unknown:
                raise EvalError(f"{spot}: unknown field {sorted(unknown)[0]!r}")
            if not (corpus / item["path"]).is_file():
                raise EvalError(f"{spot}: {item['path']} is not in the collection")
            judgment = Judgment(
                item["path"], grade, _span(item.get("lines"), spot, int), _span(item.get("seconds"), spot, float)
            )
            if judgment.label() in seen:
                raise EvalError(f"{spot}: {judgment.label()} is graded twice")
            seen.add(judgment.label())
            result.append(judgment)
    return tuple(result)


def load_set(root: Path) -> EvalSet:
    root = Path(root)
    manifest = root / "set.toml"
    if not manifest.is_file():
        raise EvalError(f"{root} is not an eval set: set.toml is missing")
    if not (root / "corpus").is_dir():
        raise EvalError(f"{root} is not an eval set: the corpus folder is missing")
    try:
        meta = tomllib.loads(manifest.read_text())
        entries = tomllib.loads((root / "queries.toml").read_text()).get("query", [])
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise EvalError(f"cannot read the eval set: {error}") from None
    corpus = root / "corpus"
    queries: list[Query] = []
    ids: set[str] = set()
    for position, entry in enumerate(entries):
        where = f"query {entry.get('id') or position + 1}"
        unknown = set(entry) - {"id", "text", "like", "category", "split", "kinds", *GRADES}
        if unknown:
            raise EvalError(f"{where}: unknown field {sorted(unknown)[0]!r}")
        query_id = entry.get("id")
        if not isinstance(query_id, str) or not query_id:
            raise EvalError(f"{where}: needs an id")
        if query_id in ids:
            raise EvalError(f"{where}: the id is used twice")
        ids.add(query_id)
        text, like = entry.get("text"), entry.get("like")
        if bool(text) == bool(like):
            raise EvalError(f"{where}: give either text or like")
        if like and not (corpus / like).is_file():
            raise EvalError(f"{where}: {like} is not in the collection")
        split = entry.get("split", "dev")
        if split not in SPLITS:
            raise EvalError(f"{where}: split must be one of {', '.join(SPLITS)}")
        kinds = entry.get("kinds")
        if kinds is not None and (not isinstance(kinds, list) or any(kind not in KINDS for kind in kinds)):
            raise EvalError(f"{where}: kinds must be a list of {', '.join(KINDS)}")
        if not isinstance(entry.get("category"), str):
            raise EvalError(f"{where}: needs a category")
        queries.append(
            Query(query_id, entry["category"], split, text, like, kinds, _judgments(entry, where, corpus))
        )
    if not queries:
        raise EvalError("the eval set has no queries")
    return EvalSet(root, str(meta.get("name", root.name)), str(meta.get("version", "0")), queries)


def fetch_set(target: Path, report: Callable[[int, int], None] = lambda done, total: None) -> EvalSet:
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=".download-", dir=target.parent))
    try:
        archive = workspace / "set.tar.gz"
        try:
            with urllib.request.urlopen(SET_ARCHIVE, timeout=60) as response, archive.open("wb") as handle:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while block := response.read(1 << 20):
                    handle.write(block)
                    done += len(block)
                    report(done, total)
        except OSError as error:
            raise EvalError(f"could not download the eval set from {SET_SOURCE}: {error}") from None
        unpacked = workspace / "set"
        _extract(archive, unpacked)
        load_set(unpacked)
        if target.exists():
            target.rename(workspace / "previous")
        unpacked.rename(target)
        return load_set(target)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                parts = Path(member.name).parts[1:]
                if not parts or not (member.isfile() or member.isdir()):
                    continue
                if member.name.startswith("/") or ".." in parts:
                    raise EvalError(f"the eval set archive contains an unsafe path: {member.name}")
                target = destination.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
    except (tarfile.TarError, OSError) as error:
        raise EvalError(f"the downloaded eval set is damaged: {error}") from None


def _feed(digest: Any, value: Any) -> None:
    if isinstance(value, str):
        digest.update(b"s" + value.encode())
    elif isinstance(value, bytes):
        digest.update(b"b" + value)
    elif isinstance(value, np.ndarray):
        digest.update(f"a{value.shape}{value.dtype}".encode() + np.ascontiguousarray(value).tobytes())
    elif isinstance(value, list | tuple):
        digest.update(f"l{len(value)}".encode())
        for item in value:
            _feed(digest, item)
    elif hasattr(value, "tobytes"):
        digest.update(f"i{getattr(value, 'size', '')}{getattr(value, 'mode', '')}".encode() + value.tobytes())
    else:
        digest.update(repr(value).encode())


def fingerprint(*parts: Any) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        _feed(digest, part)
    return digest.hexdigest()


class CachingEmbedder:
    def __init__(self, inner: Any, path: Path):
        self.inner = inner
        self.path = path
        self.active = False
        self._cache: dict[str, np.ndarray] = {}
        self._dirty = False
        if path.exists():
            try:
                with np.load(path) as stored:
                    self._cache = {key: stored[key] for key in stored.files}
            except (OSError, ValueError):
                self._cache = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _remember(self, key: str, compute: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = np.asarray(compute(), dtype=np.float32)
            self._dirty = True
        return self._cache[key]

    def embed_query(self, text: str) -> dict[str, np.ndarray]:
        key = fingerprint("query", text)
        if f"{key}.search" not in self._cache or f"{key}.code" not in self._cache:
            vectors = self.inner.embed_query(text)
            for name, vector in vectors.items():
                self._cache[f"{key}.{name}"] = np.asarray(vector, dtype=np.float32)
            self._dirty = True
        return {name: self._cache[f"{key}.{name}"] for name in ("search", "code")}

    def _maybe(self, name: str, compute: Callable[[], Any], *payload: Any) -> Any:
        if not self.active:
            return compute()
        return self._remember(fingerprint(name, *payload), compute)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._maybe("documents", lambda: self.inner.embed_documents(texts), texts)

    def embed_images(self, images: list) -> np.ndarray:
        return self._maybe("images", lambda: self.inner.embed_images(images), images)

    def embed_video(self, frames: list, fps: float) -> np.ndarray:
        return self._maybe("video", lambda: self.inner.embed_video(frames, fps), frames, fps)

    def embed_audio(self, samples: np.ndarray) -> np.ndarray:
        return self._maybe("audio", lambda: self.inner.embed_audio(samples), samples)

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(temporary, **self._cache)
        temporary.replace(self.path)
        self._dirty = False


def _overlaps(start: float | None, end: float | None, span: tuple[float, float], inclusive: bool) -> bool:
    if start is None:
        return True
    end = start if end is None else end
    if inclusive:
        return start <= span[1] and end >= span[0]
    return start < span[1] and end > span[0]


def matches(hit: dict[str, Any], judgment: Judgment, corpus: str) -> bool:
    if hit["path"] != os.path.join(corpus, judgment.path):
        return False
    if judgment.lines:
        return _overlaps(hit.get("start_line"), hit.get("end_line"), judgment.lines, True)
    if judgment.seconds:
        return _overlaps(hit.get("start_sec"), hit.get("end_sec"), judgment.seconds, False)
    return True


def relative(path: str, corpus: str) -> str:
    prefix = corpus.rstrip("/") + "/"
    return path[len(prefix) :] if path.startswith(prefix) else path


def hit_label(hit: dict[str, Any], corpus: str) -> str:
    lines = (hit["start_line"], hit["end_line"]) if hit.get("start_line") is not None else None
    seconds = (hit["start_sec"], hit["end_sec"]) if hit.get("start_sec") is not None else None
    label = relative(hit["path"], corpus) + span_label(lines, seconds)
    if hit.get("kind") == "video" and hit.get("part") == "audio":
        label += " (soundtrack)"
    return label


def score_query(query: Query, hits: list[dict[str, Any]], corpus: str) -> dict[str, Any]:
    credited: set[int] = set()
    gains: list[float] = []
    unjudged: list[dict[str, Any]] = []
    rank = None
    for position, hit in enumerate(hits, 1):
        matched = [index for index, judgment in enumerate(query.judgments) if matches(hit, judgment, corpus)]
        fresh = [index for index in matched if index not in credited and query.judgments[index].grade > 0]
        gain = 0.0
        if fresh:
            best = max(fresh, key=lambda index: query.judgments[index].grade)
            credited.add(best)
            gain = query.judgments[best].grade
            if rank is None and gain == GRADES["relevant"]:
                rank = position
        elif not matched and position <= UNJUDGED_DEPTH:
            unjudged.append({"rank": position, "hit": hit_label(hit, corpus), "text": (hit.get("text") or "")[:200]})
        gains.append(gain)
    relevant = [index for index, judgment in enumerate(query.judgments) if judgment.grade == GRADES["relevant"]]

    def recall(cutoff: int) -> float:
        if not relevant:
            return 0.0
        found = sum(
            any(matches(hit, query.judgments[index], corpus) for hit in hits[:cutoff]) for index in relevant
        )
        return found / len(relevant)

    ideal = sorted((judgment.grade for judgment in query.judgments if judgment.grade > 0), reverse=True)[:CUTOFF]
    dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains[:CUTOFF], 1))
    best_dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(ideal, 1))
    located = None
    ranged = [judgment for judgment in query.relevant if judgment.lines or judgment.seconds]
    if ranged:
        paths = {os.path.join(corpus, judgment.path) for judgment in ranged}
        first = next((hit for hit in hits[:CUTOFF] if hit["path"] in paths), None)
        if first is not None:
            located = any(matches(first, judgment, corpus) for judgment in ranged)
    return {
        "id": query.id,
        "category": query.category,
        "split": query.split,
        "answered": query.answered,
        "like": bool(query.like),
        "rank": rank,
        "hit@1": float(rank == 1),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr": 1.0 / rank if rank and rank <= CUTOFF else 0.0,
        "ndcg@10": dcg / best_dcg if best_dcg else 0.0,
        "located": located,
        "top_score": hits[0]["score"] if hits else None,
        "top": [hit_label(hit, corpus) for hit in hits[:3]],
        "expected": [judgment.label() for judgment in query.relevant],
        "unjudged": unjudged,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [record for record in records if record["answered"]]
    result: dict[str, Any] = {"queries": len(answered)}
    for name in METRICS:
        result[name] = round(statistics.fmean(record[name] for record in answered), 4) if answered else None
    located = [record["located"] for record in answered if record["located"] is not None]
    result["located"] = round(sum(located) / len(located), 4) if located else None
    result["located_queries"] = len(located)
    return result


def separation(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    texts = [record for record in records if not record.get("like")]
    positives = [record["top_score"] for record in texts if record["answered"] and record["rank"] == 1]
    negatives = [record["top_score"] for record in texts if not record["answered"]]
    if not negatives or not positives or None in positives or None in negatives:
        return None
    values = sorted(set(positives + negatives))
    candidates = [values[0] - 1e-6] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)] + [values[-1] + 1e-6]

    def correct(threshold: float) -> int:
        return sum(score >= threshold for score in positives) + sum(score < threshold for score in negatives)

    best = max(candidates, key=lambda threshold: (correct(threshold), -abs(threshold)))
    return {
        "threshold": round(best, 4),
        "correct": correct(best),
        "total": len(positives) + len(negatives),
        "answered_top_scores": round(statistics.median(positives), 4),
        "no_answer_top_scores": round(statistics.median(negatives), 4),
        "no_answer_queries": len(negatives),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        categories.setdefault(record["category"], []).append(record)
    return {
        "overall": aggregate(records),
        "categories": {name: aggregate(group) for name, group in categories.items() if any(r["answered"] for r in group)},
        "splits": {split: aggregate([r for r in records if r["split"] == split]) for split in SPLITS},
        "no_answer": separation(records),
    }


def single_kind(query: Query) -> str | None:
    kinds = {classify(judgment.path, sniff=False) for judgment in query.relevant}
    return kinds.pop() if len(kinds) == 1 and None not in kinds else None


def resolve_settings(overrides: dict[str, Any], device: str) -> tuple[Config, str, str]:
    from .model import resolve_device, resolve_dtype

    config = Config(overrides)
    resolved = resolve_device(device if device != "auto" else config.get("model.device"))
    dtype = str(resolve_dtype(resolved, config.get("model.dtype"))).replace("torch.", "")
    config.set("model.device", resolved)
    config.set("model.dtype", dtype)
    return config, resolved, dtype


def work_settings(config: Config) -> dict[str, Any]:
    return {key: config.get(key) for key in (*BASE_SETTINGS, *KIND_SETTINGS)}


def work_key(eval_set: EvalSet, config: Config) -> str:
    encoded = json.dumps(work_settings(config), sort_keys=True).encode()
    return f"{eval_set.name}-" + hashlib.blake2b(encoded, digest_size=6).hexdigest()


def seed_work(target: Path, eval_set: EvalSet, settings: dict[str, Any], corpus: str) -> str | None:
    if (target / "index" / "semantix.db").exists():
        return None
    best: tuple[int, Path, set[str]] | None = None
    for candidate in sorted(target.parent.glob(f"{eval_set.name}-*")):
        try:
            other = json.loads((candidate / "settings.json").read_text())
        except (OSError, ValueError):
            continue
        if candidate == target or any(other.get(key) != settings[key] for key in BASE_SETTINGS):
            continue
        stale = {kind for key, kinds in KIND_SETTINGS.items() if other.get(key) != settings[key] for kind in kinds}
        if best is None or len(stale) < best[0]:
            best = (len(stale), candidate, stale)
    if best is None or not (best[1] / "index" / "semantix.db").exists():
        return None
    import sqlite3

    from .store import Store

    _count, source, stale = best
    (target / "index").mkdir(parents=True, exist_ok=True)
    origin = sqlite3.connect(source / "index" / "semantix.db")
    copy = sqlite3.connect(target / "index" / "semantix.db")
    with copy:
        origin.backup(copy)
    origin.close()
    copy.close()
    if (source / "embeddings.npz").exists():
        shutil.copyfile(source / "embeddings.npz", target / "embeddings.npz")
    store = Store(target / "index" / "semantix.db")
    try:
        store.delete_files([row.id for row in store.files_under(corpus) if row.kind in stale])
    finally:
        store.close()
    return source.name


def run(
    eval_set: EvalSet,
    home: Path,
    models_dir: Path,
    overrides: dict[str, Any] | None = None,
    device: str = "auto",
    modes: tuple[str, ...] = ("hybrid",),
    split: str = "all",
    keyword_bonus: float | None = None,
    calibrate: bool = True,
    embedder: Any = None,
    say: Callable[[str], None] = lambda message: None,
    progress: Callable[[dict[str, Any]], None] = lambda event: None,
) -> dict[str, Any]:
    from .engine import KEYWORD_BONUS, Engine
    from .paths import Paths

    queries = [query for query in eval_set.queries if split == "all" or query.split == split]
    if not queries:
        raise EvalError(f"the eval set has no {split} queries")
    config, resolved, dtype = resolve_settings(overrides or {}, device)
    work = Paths(home / "work" / work_key(eval_set, config))
    corpus = os.path.realpath(eval_set.corpus)
    settings = work_settings(config)
    seeded = seed_work(work.home, eval_set, settings, corpus)
    work.home.mkdir(parents=True, exist_ok=True)
    (work.home / "settings.json").write_text(json.dumps(settings, indent=1))
    if embedder is None:
        from .model import Embedder

        embedder = Embedder(config, models_dir)
    cached = CachingEmbedder(embedder, work.home / "embeddings.npz")
    engine = Engine(work, config, cached)
    engine.keyword_bonus = KEYWORD_BONUS if keyword_bonus is None else keyword_bonus
    engine.calibrate = calibrate
    try:
        for root in engine.store.roots():
            if root.path != corpus:
                engine.forget(root.path)
        say("checking the collection")
        plan = engine.plan([corpus])
        started = time.perf_counter()
        done: dict[str, Any] = {}
        for event in engine.execute(plan["id"]):
            if event["event"] == "progress":
                progress(event)
            done = event
        sync = {
            "files": plan["totals"]["files"] + plan["totals"]["unchanged"],
            "embedded": done.get("embedded", 0),
            "failed": done.get("failed", 0),
            "seconds": round(time.perf_counter() - started, 2),
            "parts": len(engine.index),
            "reused_from": seeded,
            "errors": engine.store.error_files(),
        }
        cached.active = True
        say("searching")
        if calibrate:
            engine.ensure_anchors()
        report_modes: dict[str, Any] = {}
        per_query: dict[str, list[dict[str, Any]]] = {}
        for mode in modes:
            records = []
            timings = []
            restricted: list[tuple[str, int | None, int | None]] = []
            for query in queries:
                if query.like and mode == "keyword":
                    continue
                like = os.path.join(corpus, query.like) if query.like else None
                began = time.perf_counter()
                hits = engine.search(query.text, like, query.kinds, limit=DEPTH, mode=mode)["hits"]
                timings.append(time.perf_counter() - began)
                record = score_query(query, hits, corpus)
                records.append(record)
                kind = single_kind(query) if query.answered and not query.kinds else None
                if kind is not None:
                    own = engine.search(query.text, like, [kind], limit=DEPTH, mode=mode)["hits"]
                    restricted.append((query.id, record["rank"], score_query(query, own, corpus)["rank"]))
            summary = summarize(records)
            summary["search_median_ms"] = round(statistics.median(timings) * 1000, 1) if timings else None
            summary["mixed_types"] = mixed_types(restricted)
            report_modes[mode] = summary
            per_query[mode] = records
        cached.save()
    finally:
        engine.close()
        if hasattr(embedder, "unload"):
            embedder.unload()
    return {
        "semantix": __version__,
        "when": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "set": {"name": eval_set.name, "version": eval_set.version},
        "split": split,
        "device": resolved,
        "dtype": dtype,
        "settings": {key: config.get(key) for key in (*EMBEDDING_SETTINGS, "search.per_file")},
        "keyword_bonus": engine.keyword_bonus,
        "calibration": calibrate,
        "sync": sync,
        "modes": report_modes,
        "queries": per_query,
    }


def mixed_types(pairs: list[tuple[str, int | None, int | None]]) -> dict[str, Any] | None:
    if not pairs:
        return None

    def reciprocal(rank: int | None) -> float:
        return 1.0 / rank if rank and rank <= CUTOFF else 0.0

    worse = [
        {"id": query_id, "all_types": everywhere, "own_type": own}
        for query_id, everywhere, own in pairs
        if (everywhere or DEPTH + 1) > (own or DEPTH + 1)
    ]
    return {
        "queries": len(pairs),
        "mrr_all_types": round(statistics.fmean(reciprocal(everywhere) for _id, everywhere, _own in pairs), 4),
        "mrr_own_type": round(statistics.fmean(reciprocal(own) for _id, _everywhere, own in pairs), 4),
        "worse": worse,
    }


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    notes = []
    if old.get("set", {}).get("version") != new["set"]["version"]:
        notes.append(f"the eval set changed from version {old.get('set', {}).get('version')} to {new['set']['version']}")
    for key in ("split", "device", "dtype", "keyword_bonus", "calibration"):
        if old.get(key) != new.get(key):
            notes.append(f"{key}: {old.get(key)} → {new.get(key)}")
    for key, value in new["settings"].items():
        if old.get("settings", {}).get(key) != value:
            notes.append(f"{key}: {old.get('settings', {}).get(key)} → {value}")
    modes = {}
    for mode, summary in new["modes"].items():
        before = old.get("modes", {}).get(mode)
        if before is None:
            continue
        rows = {}
        for name, current in [("overall", summary["overall"]), *summary["categories"].items()]:
            previous = before["overall"] if name == "overall" else before.get("categories", {}).get(name)
            if previous is None:
                continue
            rows[name] = {
                metric: None if current[metric] is None or previous[metric] is None else round(current[metric] - previous[metric], 4)
                for metric in METRICS
            }
        earlier = {record["id"]: record for record in old.get("queries", {}).get(mode, [])}
        moved = []
        for record in new["queries"][mode]:
            previous = earlier.get(record["id"])
            if previous is None or not record["answered"] or previous["rank"] == record["rank"]:
                continue
            moved.append({"id": record["id"], "before": previous["rank"], "after": record["rank"]})
        modes[mode] = {"deltas": rows, "moved": moved}
    return {"notes": notes, "modes": modes}


def better(entry: dict[str, Any]) -> bool:
    return (entry["after"] or DEPTH + 1) < (entry["before"] or DEPTH + 1)


def latest_report(reports: Path, set_name: str, exclude: Path | None = None) -> Path | None:
    candidates = [path for path in reports.glob(f"{set_name}-*.json") if path != exclude] if reports.is_dir() else []
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def save_report(report: dict[str, Any], reports: Path) -> Path:
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = reports / f"{report['set']['name']}-{stamp}.json"
    counter = 1
    while target.exists():
        counter += 1
        target = reports / f"{report['set']['name']}-{stamp}-{counter}.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    return target
