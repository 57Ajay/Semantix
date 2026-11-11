from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import media
from .chunking import NotTextError, chunk_text, document_prompt, read_text
from .config import Config
from .filetypes import KIND_CAPABILITY, KINDS, TEXT_KINDS, classify, extension_of
from .ignore import IgnoreRules
from .model import Embedder
from .paths import Paths
from .scanner import Candidate, Scanner, ScanResult, file_hash, normalize_selection
from .store import (
    STATUS_ERROR,
    STATUS_INDEXED,
    STATUS_PENDING,
    STATUS_STALE,
    ChunkRecord,
    FileRow,
    Store,
    query_terms,
    tokenize,
)
from .vindex import VectorIndex

log = logging.getLogger("semantix.engine")

MB = 1024 * 1024
PLAN_TTL = 1800
KEYWORD_BONUS = 0.08
ANCHOR_QUERIES = (
    "a photo of a mountain landscape", "how to cook rice", "function that sorts a list",
    "meeting notes about the quarterly budget", "sound of heavy rain", "error handling in javascript",
    "a person riding a bicycle", "database migration script", "recipe for chocolate cake",
    "people talking in a meeting", "configuration file for a web server", "a dog playing in the park",
    "unit tests for the parser", "music with guitar and drums", "travel itinerary for a trip",
    "react component for a login form", "a diagram of system architecture", "shell script to back up files",
    "birthday party video", "notes on machine learning", "a city street at night",
    "sql query joining two tables", "a child laughing", "instructions to install a package",
    "a chart showing sales", "docker compose file", "a bird singing", "email about a job offer",
    "css styles for a button", "a football match", "kubernetes deployment manifest", "a bowl of fruit",
)
EXECUTION_ORDER = {"code": 0, "text": 0, "image": 1, "audio": 2, "video": 3}
CPU_RATES = {"text": 3.3e-4, "vision": 0.0145, "audio": 0.1}
DEVICE_SPEEDUP = {"cuda": 40.0, "mps": 8.0, "cpu": 1.0}


class EngineError(RuntimeError):
    pass


@dataclass
class PlanItem:
    path: str
    kind: str
    size: int
    mtime_ns: int
    action: str
    root: str
    file_id: int | None = None
    hash: str | None = None
    large: bool = False
    cost: float = 0.0
    duration: float | None = None
    pin: bool = False


@dataclass
class PlanScope:
    root: str
    is_dir: bool
    root_id: int | None
    selection: dict[str, Any]
    update_selection: bool = False
    merge_ids: list[int] = field(default_factory=list)
    unchanged: int = 0
    failed: int = 0
    inventory: dict[str, Any] | None = None


@dataclass
class Plan:
    id: str
    created: float
    scopes: list[PlanScope] = field(default_factory=list)
    items: list[PlanItem] = field(default_factory=list)
    deletes: list[tuple[int, str, str]] = field(default_factory=list)
    moves: list[tuple[int, str, Candidate, str]] = field(default_factory=list)
    touches: list[tuple[int, int, int]] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    off_kinds: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Run:
    cancel: threading.Event
    total_cost: float
    total_files: int
    started: float = field(default_factory=time.time)
    done_cost: float = 0.0
    files_done: int = 0
    embedded: int = 0
    failed: int = 0
    chunks: int = 0
    finished: set[str] = field(default_factory=set)
    measured: dict[str, list[float]] = field(default_factory=lambda: {"text": [0, 0], "vision": [0, 0], "audio": [0, 0]})

    def measure(self, name: str, units: float, seconds: float) -> None:
        self.measured[name][0] += units
        self.measured[name][1] += seconds

    def progress(self, path: str | None = None, partial: float = 0.0) -> dict[str, Any]:
        done = self.done_cost + partial
        elapsed = time.time() - self.started
        remaining = max(self.total_cost - done, 0.0)
        eta = remaining
        if done > 0 and elapsed > 2:
            eta = elapsed * remaining / done
        return {
            "event": "progress",
            "done": round(done, 3),
            "total": round(self.total_cost, 3),
            "files_done": self.files_done,
            "files_total": self.total_files,
            "path": path,
            "eta": round(eta, 1),
            "elapsed": round(elapsed, 1),
        }


def _escape_pattern(relative: str) -> str:
    return "".join("\\" + char if char in "*?[]\\" else char for char in relative)


class Engine:
    def __init__(self, paths: Paths, config: Config | None = None, embedder: Any = None):
        self.paths = paths.ensure()
        self.config = config or Config.load(paths.config)
        self.store = Store(paths.db)
        self._custom_embedder = embedder is not None
        self.embedder = embedder or Embedder(self.config, paths.models)
        self.index = VectorIndex(self.config.get("model.dimension"))
        self.plans: dict[str, Plan] = {}
        self.sync_lock = threading.Lock()
        self.started = time.time()
        self.job: dict[str, Any] | None = None
        self._reconcile_index()

    def close(self) -> None:
        self.store.close()

    def _reconcile_index(self) -> None:
        dimension = self.config.get("model.dimension")
        model_id = self.config.get("model.id")
        stored_model = self.store.get_meta("model_id")
        stored_dimension = self.store.stored_dimension()
        if stored_model and stored_model != model_id:
            log.warning("model changed from %s to %s, all files need embedding again", stored_model, model_id)
            self.store.reset_embeddings()
        elif stored_dimension and stored_dimension > dimension:
            log.info("shortening stored vectors from %d to %d", stored_dimension, dimension)
            self.store.truncate_vectors(dimension)
        elif stored_dimension and stored_dimension < dimension:
            log.warning("dimension raised from %d to %d, all files need embedding again", stored_dimension, dimension)
            self.store.reset_embeddings()
        self.store.set_meta("model_id", model_id)
        self.index = VectorIndex(dimension)
        ids, files, kinds, vectors = self.store.load_vectors(dimension)
        self.index.add(ids, files, kinds, vectors)
        self._load_anchors()

    def _anchor_key(self) -> str:
        digest = hashlib.blake2b("|".join(ANCHOR_QUERIES).encode(), digest_size=6).hexdigest()
        return f"anchors:{self.config.get('model.id')}:{self.config.get('model.dimension')}:{digest}"

    def _load_anchors(self) -> bool:
        blob = self.store.get_blob(self._anchor_key())
        if blob is None:
            return False
        dimension = self.config.get("model.dimension")
        anchors = np.frombuffer(blob, dtype=np.float32).reshape(2, len(ANCHOR_QUERIES), dimension)
        self.index.set_anchors(anchors[0], anchors[1])
        return True

    def ensure_anchors(self) -> None:
        if self.index.calibrated or self._load_anchors():
            return
        embedded = [self.embedder.embed_query(text) for text in ANCHOR_QUERIES]
        anchors = np.stack(
            [np.stack([item["search"] for item in embedded]), np.stack([item["code"] for item in embedded])]
        ).astype(np.float32)
        self.store.set_blob(self._anchor_key(), anchors.tobytes())
        self.index.set_anchors(anchors[0], anchors[1])

    def reload_config(self) -> dict[str, Any]:
        fresh = Config.load(self.paths.config)
        previous = self.config
        self.config = fresh
        model_changed = False
        if not self._custom_embedder:
            old = self.embedder
            candidate = Embedder(fresh, self.paths.models)
            model_changed = candidate.signature() != old.signature()
            if model_changed and self.sync_lock.locked():
                self.config = previous
                raise EngineError("saved, but a sync is running: run `semantix restart` when it finishes to apply it")
            if model_changed:
                old.unload()
                self.embedder = candidate
            else:
                old.config = fresh
                old.dimension = fresh.get("model.dimension")
        else:
            self.embedder.dimension = fresh.get("model.dimension")
        if fresh.get("model.dimension") != previous.get("model.dimension") or fresh.get("model.id") != previous.get("model.id"):
            self._reconcile_index()
        return {"model_changed": model_changed}

    def device(self) -> str:
        return getattr(self.embedder, "device", None) or "cpu"

    def _rate(self, name: str) -> float:
        device = self.device()
        measured = self.store.get_stat(f"rate.{device}.{name}")
        if measured:
            return measured
        return CPU_RATES[name] / DEVICE_SPEEDUP.get(device, 1.0)

    def _estimate(self, kind: str, size: int, duration: float | None) -> float:
        if kind in TEXT_KINDS:
            return size * self._rate("text")
        if kind == "image":
            return self.config.get("image.tokens") * self._rate("vision")
        seconds = duration or 0.0
        if kind == "audio":
            return seconds * self._rate("audio")
        frames = seconds * self.config.get("video.fps")
        cost = frames * self.config.get("video.frame_tokens") * self._rate("vision")
        if self.config.get("video.audio_track") and self.config.has("audio"):
            cost += seconds * self._rate("audio")
        return cost

    def _is_large(self, kind: str, size: int) -> bool:
        key = "text" if kind in TEXT_KINDS else kind
        limit = self.config.get(f"limits.{key}_mb")
        return bool(limit) and size > limit * MB

    def _rules(self, root: str, selection: dict[str, Any]) -> IgnoreRules:
        return IgnoreRules(
            root,
            global_ignore=self.paths.ignore,
            respect_gitignore=self.config.get("index.respect_gitignore"),
            exclude=selection.get("exclude"),
        )

    def _scanner(self, root: str, selection: dict[str, Any]) -> Scanner:
        return Scanner(
            self._rules(root, selection),
            selection,
            tuple(self.config.get("model.capabilities")),
            self.config.get("index.follow_symlinks"),
        )

    def _prune_plans(self) -> None:
        cutoff = time.time() - PLAN_TTL
        for plan_id in [key for key, plan in self.plans.items() if plan.created < cutoff]:
            del self.plans[plan_id]

    def plan(
        self,
        targets: list[str] | None = None,
        selection: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        self._prune_plans()
        plan = Plan(uuid.uuid4().hex, time.time())
        override = normalize_selection(selection) if selection else None
        seen: set[str] = set()
        if targets:
            ordered = sorted(dict.fromkeys(targets), key=len)
            claimed: list[str] = []
            for target in ordered:
                if any(target == parent or target.startswith(parent.rstrip("/") + "/") for parent in claimed):
                    continue
                if os.path.isdir(target):
                    claimed.append(target)
                self._plan_target(plan, target, override, force, seen)
        else:
            roots = self.store.roots()
            if not roots:
                plan.notes.append("Nothing is tracked yet. Run `semantix sync <path>` to add a directory or file.")
            for root in roots:
                self._plan_target(plan, root.path, override, force, seen)
        self.plans[plan.id] = plan
        return self._summarize(plan)

    def _plan_target(
        self, plan: Plan, target: str, override: dict[str, Any] | None, force: bool, seen: set[str]
    ) -> None:
        root = self.store.root_containing(target)
        exists = os.path.exists(target)
        if root is None and not exists:
            plan.notes.append(f"{target} does not exist.")
            return
        if root is not None and not os.path.exists(root.path):
            plan.notes.append(
                f"{root.path} is not available right now and was skipped. "
                f"If it is gone for good, run `semantix forget {root.path}`."
            )
            return
        if root is None:
            is_dir = os.path.isdir(target)
            scope = PlanScope(target, is_dir, None, override or normalize_selection(None))
            if is_dir:
                scope.merge_ids = [child.id for child in self.store.roots_under(target)]
        else:
            chosen = normalize_selection(root.selection)
            update = False
            if override and target == root.path:
                merged = dict(chosen)
                if override["kinds"] is not None:
                    merged["kinds"] = override["kinds"]
                if override["skip_ext"]:
                    merged["skip_ext"] = sorted(set(chosen["skip_ext"]) | set(override["skip_ext"]))
                if override["exclude"]:
                    merged["exclude"] = list(dict.fromkeys([*chosen["exclude"], *override["exclude"]]))
                update = merged != chosen
                chosen = merged
            scope = PlanScope(root.path, root.is_dir, root.id, chosen, update_selection=update)
        plan.scopes.append(scope)

        scanner = self._scanner(scope.root, scope.selection)
        known = {row.path: row for row in self.store.files_under(target)}
        if not exists and not known:
            plan.notes.append(f"{target} does not exist.")
            plan.scopes.remove(scope)
            return
        pinned = {path for path, row in known.items() if row.pinned}
        pin_target = False
        if exists and os.path.isfile(target):
            result = ScanResult()
            found = scanner.candidate(target, result, explicit=True)
            if found:
                result.candidates.append(found)
                if scope.is_dir:
                    verdict = scanner.rules.explain(target)
                    deselected = (
                        extension_of(target) in scanner.skip_ext
                        or (scanner.kinds is not None and found.kind not in scanner.kinds)
                    )
                    pin_target = verdict.ignored or deselected
        elif exists:
            if scope.root_id is not None and target != scope.root and scanner.rules.explain(target).ignored:
                source = scanner.rules.explain(target).source
                plan.notes.append(f"{target} is ignored by {source}. Use !pattern in .semantixignore to bring it back.")
                return
            result = scanner.scan(target, pinned)
        else:
            result = ScanResult()
        for reason, count in result.skipped.items():
            plan.skipped[reason] = plan.skipped.get(reason, 0) + count
        for kind, count in result.off_kinds.items():
            plan.off_kinds[kind] = plan.off_kinds.get(kind, 0) + count

        fresh: list[Candidate] = []
        for candidate in result.candidates:
            if candidate.path in seen:
                known.pop(candidate.path, None)
                continue
            seen.add(candidate.path)
            row = known.pop(candidate.path, None)
            if row is None:
                fresh.append(candidate)
                continue
            self._plan_known(plan, scope, candidate, row, force, pin_target)

        missing = [row for path, row in known.items() if path not in result.off_paths]
        by_hash: dict[str, FileRow] = {}
        sizes: set[int] = set()
        for row in missing:
            if row.hash and row.status == STATUS_INDEXED:
                by_hash.setdefault(row.hash, row)
                sizes.add(row.size)
        moved: set[int] = set()
        for candidate in fresh:
            digest = None
            if candidate.size in sizes:
                try:
                    digest = file_hash(candidate.path, candidate.size)
                except OSError:
                    digest = None
                row = by_hash.get(digest) if digest else None
                if row is not None and row.id not in moved and row.kind == candidate.kind:
                    moved.add(row.id)
                    plan.moves.append((row.id, row.path, candidate, scope.root))
                    continue
            self._add_item(plan, scope, candidate, "new", None, digest, pin_target)
        for row in missing:
            if row.id not in moved:
                plan.deletes.append((row.id, row.path, scope.root))
        if scope.root_id is None and scope.is_dir:
            scope.inventory = self._inventory(result.candidates)

    def _plan_known(
        self, plan: Plan, scope: PlanScope, candidate: Candidate, row: FileRow, force: bool, pin: bool
    ) -> None:
        if force:
            self._add_item(plan, scope, candidate, "changed", row.id, None, pin)
            return
        same_stat = row.size == candidate.size and row.mtime_ns == candidate.mtime_ns
        if row.status == STATUS_ERROR and same_stat:
            scope.failed += 1
            return
        if row.status in (STATUS_PENDING, STATUS_ERROR):
            self._add_item(plan, scope, candidate, "new" if row.status == STATUS_PENDING else "retry", row.id, None, pin)
            return
        if same_stat and row.status == STATUS_INDEXED:
            scope.unchanged += 1
            return
        try:
            digest = file_hash(candidate.path, candidate.size)
        except OSError:
            digest = None
        if digest and digest == row.hash and row.status == STATUS_INDEXED:
            plan.touches.append((row.id, candidate.size, candidate.mtime_ns))
            scope.unchanged += 1
            return
        self._add_item(plan, scope, candidate, "changed", row.id, digest, pin)

    def _add_item(
        self, plan: Plan, scope: PlanScope, candidate: Candidate, action: str,
        file_id: int | None, digest: str | None, pin: bool,
    ) -> None:
        duration = None
        if candidate.kind in ("video", "audio"):
            try:
                duration = media.probe(candidate.path).duration
            except Exception:
                duration = None
        plan.items.append(
            PlanItem(
                candidate.path, candidate.kind, candidate.size, candidate.mtime_ns, action, scope.root,
                file_id=file_id, hash=digest, large=self._is_large(candidate.kind, candidate.size),
                cost=self._estimate(candidate.kind, candidate.size, duration), duration=duration, pin=pin,
            )
        )

    @staticmethod
    def _inventory(candidates: list[Candidate]) -> dict[str, Any]:
        kinds: dict[str, dict[str, int]] = {}
        extensions: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            entry = kinds.setdefault(candidate.kind, {"files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += candidate.size
            extension = extension_of(candidate.path) or os.path.basename(candidate.path)
            entry = extensions.setdefault(extension, {"ext": extension, "kind": candidate.kind, "files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += candidate.size
        ranked = sorted(extensions.values(), key=lambda entry: (-entry["files"], entry["ext"]))
        return {"kinds": kinds, "extensions": ranked[:40]}

    def _summarize(self, plan: Plan) -> dict[str, Any]:
        roots: dict[str, dict[str, Any]] = {}
        for scope in plan.scopes:
            roots[scope.root] = {
                "path": scope.root,
                "new": scope.root_id is None,
                "is_dir": scope.is_dir,
                "selection": scope.selection,
                "groups": {},
                "deletes": 0,
                "moves": 0,
                "unchanged": scope.unchanged,
                "failed": scope.failed,
                "inventory": scope.inventory,
            }
        for item in plan.items:
            group = roots[item.root]["groups"].setdefault(
                item.kind, {"kind": item.kind, "files": 0, "bytes": 0, "cost": 0.0, "new": 0, "changed": 0, "retry": 0}
            )
            group["files"] += 1
            group["bytes"] += item.size
            group["cost"] += item.cost
            group[item.action] += 1
        for _file_id, _path, root in plan.deletes:
            roots[root]["deletes"] += 1
        for _file_id, _old, _candidate, root in plan.moves:
            roots[root]["moves"] += 1
        for entry in roots.values():
            entry["groups"] = sorted(entry["groups"].values(), key=lambda group: KINDS.index(group["kind"]))
        large = [
            {"path": item.path, "kind": item.kind, "size": item.size, "cost": item.cost, "root": item.root, "duration": item.duration}
            for item in plan.items
            if item.large
        ]
        return {
            "id": plan.id,
            "roots": list(roots.values()),
            "large": sorted(large, key=lambda entry: -entry["size"]),
            "totals": {
                "files": len(plan.items),
                "bytes": sum(item.size for item in plan.items),
                "cost": sum(item.cost for item in plan.items),
                "deletes": len(plan.deletes),
                "moves": len(plan.moves),
                "unchanged": sum(scope.unchanged for scope in plan.scopes),
            },
            "skipped": plan.skipped,
            "off_kinds": plan.off_kinds,
            "notes": plan.notes,
            "device": self.device(),
            "measured": self.store.get_stat(f"rate.{self.device()}.text") is not None,
        }

    def execute(
        self,
        plan_id: str,
        groups: list[list[str]] | None = None,
        skip: list[str] | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        plan = self.plans.pop(plan_id, None)
        if plan is None:
            raise EngineError("this sync plan has expired, run the command again")
        if not self.sync_lock.acquire(blocking=False):
            raise EngineError("another sync is already running")
        try:
            yield from self._execute(plan, groups, set(skip or []), cancel or threading.Event())
        finally:
            self.job = None
            self.sync_lock.release()

    def _execute(
        self, plan: Plan, groups: list[list[str]] | None, skip: set[str], cancel: threading.Event
    ) -> Iterator[dict[str, Any]]:
        root_ids: dict[str, int] = {}
        for scope in plan.scopes:
            if scope.root_id is None:
                existing = self.store.root_by_path(scope.root)
                root = existing or self.store.add_root(scope.root, scope.is_dir, scope.selection)
                scope.root_id = root.id
                for child_id in scope.merge_ids:
                    self.store.merge_root(child_id, root.id)
            elif scope.update_selection:
                self.store.update_root_selection(scope.root_id, scope.selection)
            root_ids[scope.root] = scope.root_id

        if plan.deletes:
            file_ids = [file_id for file_id, _path, _root in plan.deletes]
            self.store.delete_files(file_ids)
            self.index.remove_files(file_ids)
        for file_id, _old, candidate, root in plan.moves:
            self.store.move_file(file_id, root_ids[root], candidate.path, candidate.size, candidate.mtime_ns)
        for file_id, size, mtime_ns in plan.touches:
            self.store.touch_file(file_id, size, mtime_ns)

        allowed = None if groups is None else {(root, kind) for root, kind in groups}
        selected: list[PlanItem] = []
        deferred: list[PlanItem] = []
        for item in plan.items:
            chosen = (allowed is None or (item.root, item.kind) in allowed) and item.path not in skip
            (selected if chosen else deferred).append(item)
        selected.sort(key=lambda item: (EXECUTION_ORDER[item.kind], item.path))

        run = Run(cancel, sum(item.cost for item in selected) or 1.0, len(selected))
        self.job = {"started": run.started, "files_total": run.total_files, "run": run}
        yield {"event": "start", "files_total": run.total_files, "total": run.total_cost}

        text_items = [item for item in selected if item.kind in TEXT_KINDS]
        image_items = [item for item in selected if item.kind == "image"]
        other_items = [item for item in selected if item.kind in ("audio", "video")]
        if text_items:
            yield from self._run_text(text_items, root_ids, run)
        if image_items and not cancel.is_set():
            yield from self._run_images(image_items, root_ids, run)
        for item in other_items:
            if cancel.is_set():
                break
            yield from self._run_media(item, root_ids, run)

        unfinished = [item for item in selected if item.path not in run.finished]
        for item in [*deferred, *unfinished]:
            self._record_waiting(item, root_ids[item.root])
        for root_id in set(root_ids.values()):
            self.store.touch_root(root_id)
        self._save_rates(run)
        yield {
            "event": "done",
            "embedded": run.embedded,
            "failed": run.failed,
            "chunks": run.chunks,
            "deleted": len(plan.deletes),
            "moved": len(plan.moves),
            "deferred": len(deferred) + len(unfinished),
            "cancelled": cancel.is_set(),
            "seconds": round(time.time() - run.started, 1),
        }

    def _record_waiting(self, item: PlanItem, root_id: int) -> None:
        row = self.store.file_by_path(item.path)
        if row is None:
            self.store.upsert_file(
                root_id, item.path, item.kind, item.size, item.mtime_ns, None, STATUS_PENDING,
                pinned=True if item.pin else None,
            )
        elif row.status == STATUS_INDEXED:
            self.store.set_file_status(row.id, STATUS_STALE)

    def _save_rates(self, run: Run) -> None:
        minimums = {"text": 2000, "vision": 280, "audio": 5}
        device = self.device()
        for name, (units, seconds) in run.measured.items():
            if units >= minimums[name] and seconds > 0:
                self.store.update_stat(f"rate.{device}.{name}", seconds / units)

    def _fail(self, item: PlanItem, root_id: int, error: Exception | str, run: Run) -> dict[str, Any]:
        message = str(error) or type(error).__name__
        message = message.replace(f": '{item.path}'", "").replace(f"'{item.path}'", "").strip()
        log.warning("could not embed %s: %s", item.path, message)
        self.store.upsert_file(
            root_id, item.path, item.kind, item.size, item.mtime_ns, item.hash, STATUS_ERROR,
            error=message[:500], pinned=True if item.pin else None,
        )
        run.failed += 1
        run.files_done += 1
        run.done_cost += item.cost
        run.finished.add(item.path)
        return {"event": "file", "path": item.path, "status": "error", "error": message[:500]}

    def _commit(self, item: PlanItem, root_id: int, records: list[ChunkRecord], run: Run) -> dict[str, Any]:
        digest = item.hash
        if digest is None:
            try:
                digest = file_hash(item.path, item.size)
            except OSError:
                digest = None
        file_id = self.store.upsert_file(
            root_id, item.path, item.kind, item.size, item.mtime_ns, digest, STATUS_INDEXED,
            pinned=True if item.pin else None,
        )
        self.index.remove_files([file_id])
        ids = self.store.replace_chunks(file_id, item.path, records)
        if records:
            vectors = np.stack([record.vector for record in records])
            self.index.add(ids, [file_id] * len(ids), [item.kind] * len(ids), vectors)
        run.embedded += 1
        run.chunks += len(records)
        run.files_done += 1
        run.done_cost += item.cost
        run.finished.add(item.path)
        return {"event": "file", "path": item.path, "status": "ok", "chunks": len(records)}

    def _reuse_twin(self, item: PlanItem, root_id: int, run: Run) -> dict[str, Any] | None:
        if item.kind in TEXT_KINDS:
            return None
        if item.hash is None:
            try:
                item.hash = file_hash(item.path, item.size)
            except OSError:
                return None
        twin = self.store.indexed_twin(item.hash, item.kind, item.path)
        if twin is None:
            return None
        records = self.store.chunk_records(twin)
        if any(len(record.vector) != self.index.dimension for record in records):
            return None
        event = self._commit(item, root_id, records, run)
        event["reused"] = True
        return event

    @staticmethod
    def _title(item: PlanItem) -> str:
        if item.path == item.root:
            return os.path.basename(item.path)
        base = os.path.basename(item.root.rstrip("/"))
        relative = item.path[len(item.root.rstrip("/")) + 1 :]
        return f"{base}/{relative}" if base else relative

    def _run_text(self, items: list[PlanItem], root_ids: dict[str, int], run: Run) -> Iterator[dict[str, Any]]:
        budget = self.config.get("index.chunk_chars")
        window = max(self.embedder.text_batch_size() * 4, 8)
        queue: list[dict[str, Any]] = []
        queued = 0

        def flush() -> Iterator[dict[str, Any]]:
            nonlocal queue, queued
            texts: list[str] = []
            owners: list[tuple[dict[str, Any], int]] = []
            for entry in queue:
                for position in entry["need"]:
                    texts.append(document_prompt(entry["title"], entry["chunks"][position].text))
                    owners.append((entry, position))
            failed: set[int] = set()
            for start in range(0, len(texts), window):
                if run.cancel.is_set():
                    queue, queued = [], 0
                    return
                batch = texts[start : start + window]
                began = time.time()
                try:
                    vectors = self.embedder.embed_documents(batch)
                except Exception as error:
                    for entry, _position in owners[start : start + window]:
                        if id(entry) not in failed:
                            failed.add(id(entry))
                            entry["error"] = error
                    continue
                run.measure("text", sum(len(text) for text in batch), time.time() - began)
                for (entry, position), vector in zip(owners[start : start + window], vectors, strict=True):
                    entry["vectors"][position] = vector
                done_share = sum(
                    entry["item"].cost * sum(vector is not None for vector in entry["vectors"]) / max(len(entry["vectors"]), 1)
                    for entry in queue
                )
                current = owners[min(start + window, len(owners)) - 1][0]["item"].path
                yield run.progress(current, done_share)
            for entry in queue:
                item = entry["item"]
                if "error" in entry:
                    yield self._fail(item, root_ids[item.root], entry["error"], run)
                else:
                    records = [
                        ChunkRecord(
                            item.kind, entry["vectors"][position], ord=position,
                            start_line=chunk.start_line, end_line=chunk.end_line, hash=chunk.digest, text=chunk.text,
                        )
                        for position, chunk in enumerate(entry["chunks"])
                    ]
                    yield self._commit(item, root_ids[item.root], records, run)
                yield run.progress(item.path)
            queue, queued = [], 0

        for item in items:
            if run.cancel.is_set():
                break
            try:
                chunks = chunk_text(read_text(item.path), budget)
            except (OSError, NotTextError) as error:
                yield self._fail(item, root_ids[item.root], error, run)
                continue
            reuse = self.store.chunk_vectors_by_hash(item.file_id) if item.file_id else {}
            vectors: list[np.ndarray | None] = []
            need: list[int] = []
            dimension = self.index.dimension
            for position, chunk in enumerate(chunks):
                cached = reuse.get(chunk.digest)
                if cached is not None and len(cached) == dimension:
                    vectors.append(cached)
                else:
                    vectors.append(None)
                    need.append(position)
            queue.append({"item": item, "title": self._title(item), "chunks": chunks, "vectors": vectors, "need": need})
            queued += len(need)
            if queued >= window:
                yield from flush()
        if queue and not run.cancel.is_set():
            yield from flush()

    def _run_images(self, items: list[PlanItem], root_ids: dict[str, int], run: Run) -> Iterator[dict[str, Any]]:
        size = self.embedder.media_batch_size()
        tokens = self.config.get("image.tokens")
        for start in range(0, len(items), size):
            if run.cancel.is_set():
                return
            loaded: list[tuple[PlanItem, Any]] = []
            leaders: set[str] = set()
            copies: dict[str, list[PlanItem]] = {}
            for item in items[start : start + size]:
                reused = self._reuse_twin(item, root_ids[item.root], run)
                if reused:
                    yield reused
                    yield run.progress(item.path)
                    continue
                if item.hash in leaders:
                    copies.setdefault(item.hash, []).append(item)
                    continue
                try:
                    loaded.append((item, media.load_image(item.path)))
                except Exception as error:
                    yield self._fail(item, root_ids[item.root], error, run)
                    continue
                if item.hash:
                    leaders.add(item.hash)
            if not loaded:
                continue
            began = time.time()
            try:
                vectors = self.embedder.embed_images([image for _item, image in loaded])
            except Exception as error:
                for item in [*(item for item, _image in loaded), *(copy for group in copies.values() for copy in group)]:
                    yield self._fail(item, root_ids[item.root], error, run)
                continue
            run.measure("vision", tokens * len(loaded), time.time() - began)
            for (item, _image), vector in zip(loaded, vectors, strict=True):
                for member in [item, *copies.get(item.hash or "", [])]:
                    yield self._commit(member, root_ids[member.root], [ChunkRecord("image", vector)], run)
            yield run.progress(loaded[-1][0].path)

    def _run_media(self, item: PlanItem, root_ids: dict[str, int], run: Run) -> Iterator[dict[str, Any]]:
        root_id = root_ids[item.root]
        records: list[ChunkRecord] = []
        duration = item.duration or 0.0
        yield run.progress(item.path)
        reused = self._reuse_twin(item, root_id, run)
        if reused:
            yield reused
            yield run.progress(item.path)
            return
        try:
            info = media.probe(item.path)
            duration = info.duration or duration
            with_audio = item.kind == "audio" or (
                self.config.get("video.audio_track") and self.config.has("audio") and info.has_audio
            )
            video_share = 1.0
            if item.kind == "video" and with_audio and item.cost > 0:
                video_share = 1.0 - min(duration * self._rate("audio") / item.cost, 0.9)
            if item.kind == "video":
                if not info.has_video:
                    raise EngineError("no video stream found")
                fps = self.config.get("video.fps")
                tokens = self.config.get("video.frame_tokens")
                for begin, end, frames in media.iter_video_segments(item.path, fps, self.config.get("video.segment_seconds")):
                    if run.cancel.is_set():
                        return
                    began = time.time()
                    vector = self.embedder.embed_video(frames, fps)
                    run.measure("vision", tokens * len(frames), time.time() - began)
                    records.append(ChunkRecord("video", vector, ord=len(records), start_sec=begin, end_sec=end))
                    fraction = min(end / duration, 1.0) if duration else 0.0
                    yield run.progress(item.path, item.cost * video_share * fraction)
            if with_audio:
                seconds = self.config.get("audio.segment_seconds")
                for begin, end, samples in media.iter_audio_segments(item.path, seconds):
                    if run.cancel.is_set():
                        return
                    began = time.time()
                    vector = self.embedder.embed_audio(samples)
                    run.measure("audio", end - begin, time.time() - began)
                    records.append(ChunkRecord("audio", vector, ord=len(records), start_sec=begin, end_sec=end))
                    fraction = min(end / duration, 1.0) if duration else 0.0
                    base = item.cost * video_share if item.kind == "video" else 0.0
                    share = (1.0 - video_share) if item.kind == "video" else 1.0
                    yield run.progress(item.path, base + item.cost * share * fraction)
        except Exception as error:
            yield self._fail(item, root_id, error, run)
            return
        yield self._commit(item, root_id, records, run)
        yield run.progress(item.path)

    def _embed_like(self, path: str) -> tuple[np.ndarray, str]:
        if not os.path.isfile(path):
            raise EngineError(f"{path} is not a file")
        kind = classify(path)
        if kind is None:
            raise EngineError(f"{path} is not a supported file type")
        if kind == "image":
            return self.embedder.embed_images([media.load_image(path)])[0], kind
        if kind == "audio":
            for _begin, _end, samples in media.iter_audio_segments(path, self.config.get("audio.segment_seconds")):
                return self.embedder.embed_audio(samples), kind
            raise EngineError(f"{path} has no audible content")
        if kind == "video":
            fps = self.config.get("video.fps")
            for _begin, _end, frames in media.iter_video_segments(path, fps, self.config.get("video.segment_seconds")):
                return self.embedder.embed_video(frames, fps), kind
            raise EngineError(f"{path} has no video frames")
        chunks = chunk_text(read_text(path), self.config.get("index.chunk_chars"))
        if not chunks:
            raise EngineError(f"{path} is empty")
        text = document_prompt(os.path.basename(path), chunks[0].text)
        return self.embedder.embed_documents([text])[0], kind

    def search(
        self,
        query: str | None = None,
        like: str | None = None,
        kinds: list[str] | None = None,
        under: str | None = None,
        limit: int | None = None,
        mode: str | None = None,
        per_file: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        limit = limit or self.config.get("search.limit")
        mode = mode or self.config.get("search.mode")
        per_file = self.config.get("search.per_file") if per_file is None else per_file
        if kinds:
            unknown = [kind for kind in kinds if kind not in KINDS]
            if unknown:
                raise EngineError(f"unknown type {unknown[0]!r}, choose from {', '.join(KINDS)}")
        if not query and not like:
            raise EngineError("give a search query or a file to compare against")
        file_filter = self.store.file_ids_under(under) if under else None
        pool = max(limit * 6, 60)
        scored: dict[int, tuple[float, float]] = {}
        keyword_hits: list[tuple[int, float]] = []
        primary = code_query = None
        calibrate = True
        if like:
            primary, _kind = self._embed_like(like)
            mode = "vector"
            calibrate = False
        elif mode != "keyword":
            self.ensure_anchors()
            embedded = self.embedder.embed_query(query)
            primary, code_query = embedded["search"], embedded["code"]
        if primary is not None:
            for chunk_id, score, raw in self.index.search(primary, pool, code_query, kinds, file_filter, calibrate):
                scored[chunk_id] = (score, raw)
        if query and mode != "vector":
            keyword_hits = self.store.keyword_search(query, pool, kinds, under)
            missing = [chunk_id for chunk_id, _rank in keyword_hits if chunk_id not in scored]
            if missing and primary is not None:
                scored.update(self.index.score_of(missing, primary, code_query, calibrate))

        keyword_ids = [chunk_id for chunk_id, _rank in keyword_hits]
        details = self.store.chunk_details(list(dict.fromkeys([*scored, *keyword_ids])))
        coverage = self._keyword_coverage(query, keyword_ids, details) if keyword_hits else {}
        if mode == "keyword":
            ranked = [chunk_id for chunk_id in keyword_ids if chunk_id in details]
            final = {chunk_id: None for chunk_id in ranked}
        else:
            final = {
                chunk_id: score + KEYWORD_BONUS * coverage.get(chunk_id, 0.0) for chunk_id, (score, _raw) in scored.items()
            }
            ranked = sorted(final, key=lambda chunk_id: -final[chunk_id])

        hits = []
        per_file_count: dict[int, int] = {}
        for chunk_id in ranked:
            detail = details.get(chunk_id)
            if detail is None:
                continue
            if like and detail["path"] == like:
                continue
            score = final[chunk_id]
            if min_score is not None and (score is None or score < min_score):
                continue
            count = per_file_count.get(detail["file_id"], 0)
            if per_file and count >= per_file:
                continue
            per_file_count[detail["file_id"]] = count + 1
            raw = scored.get(chunk_id, (None, None))[1]
            hits.append(
                {
                    "path": detail["path"],
                    "kind": detail["file_kind"],
                    "part": detail["kind"],
                    "score": None if score is None else round(score, 4),
                    "similarity": None if raw is None else round(raw, 4),
                    "keyword": round(coverage.get(chunk_id, 0.0), 3),
                    "start_line": detail["start_line"],
                    "end_line": detail["end_line"],
                    "start_sec": detail["start_sec"],
                    "end_sec": detail["end_sec"],
                    "text": detail["text"],
                    "stale": detail["status"] == STATUS_STALE,
                    "exists": os.path.exists(detail["path"]),
                }
            )
            if len(hits) >= limit:
                break
        return {
            "hits": hits,
            "seconds": round(time.time() - started, 3),
            "mode": mode,
            "searched": len(self.index),
            "calibrated": calibrate and mode != "keyword",
        }

    def _keyword_coverage(self, query: str, chunk_ids: list[int], details: dict[int, dict[str, Any]]) -> dict[int, float]:
        terms = query_terms(query)
        if not terms:
            return {}
        total = max(self.store.chunk_count(), 1)
        counts = self.store.term_document_counts(terms)
        weights = {
            term: math.log(1.0 + (total - counts.get(term, 0) + 0.5) / (counts.get(term, 0) + 0.5)) for term in terms
        }
        whole = sum(weights.values()) or 1.0
        coverage = {}
        for chunk_id in chunk_ids:
            detail = details.get(chunk_id)
            if detail is None:
                continue
            present = set(tokenize(detail["path"])) | set(tokenize(detail["text"] or ""))
            coverage[chunk_id] = sum(weight for term, weight in weights.items() if term in present) / whole
        return coverage

    def forget(self, path: str) -> dict[str, Any]:
        root = self.store.root_by_path(path)
        if root is not None:
            file_ids = self.store.delete_root(root.id)
            self.index.remove_files(file_ids)
            return {"path": path, "files": len(file_ids), "root_removed": True}
        root = self.store.root_containing(path)
        if root is None:
            raise EngineError(f"{path} is not tracked")
        rows = self.store.files_under(path)
        file_ids = [row.id for row in rows]
        self.store.delete_files(file_ids)
        self.index.remove_files(file_ids)
        relative = path[len(root.path.rstrip("/")) + 1 :]
        selection = normalize_selection(root.selection)
        pattern = "/" + _escape_pattern(relative)
        if pattern not in selection["exclude"]:
            selection["exclude"].append(pattern)
            self.store.update_root_selection(root.id, selection)
        return {"path": path, "files": len(file_ids), "root_removed": False, "root": root.path, "excluded": pattern}

    def check(self, path: str) -> dict[str, Any]:
        root = self.store.root_containing(path)
        anchor = root.path if root else (path if os.path.isdir(path) else os.path.dirname(path))
        selection = normalize_selection(root.selection if root else None)
        rules = self._rules(anchor, selection)
        verdict = rules.explain(path) if path != anchor else rules.check(path, os.path.isdir(path))
        kind = classify(path) if os.path.isfile(path) else None
        row = self.store.file_by_path(path)
        reason = None
        if verdict.ignored:
            reason = "ignored"
        elif os.path.isfile(path):
            if kind is None:
                reason = "unsupported file type"
            elif extension_of(path) in selection["skip_ext"]:
                reason = "its extension is skipped for this directory"
            elif selection["kinds"] is not None and kind not in selection["kinds"]:
                reason = f"{kind} files are not selected for this directory"
            elif not self.config.has(KIND_CAPABILITY[kind]):
                reason = f"the {KIND_CAPABILITY[kind]} capability is turned off"
        if row is not None and row.pinned:
            reason = None
        return {
            "path": path,
            "exists": os.path.exists(path),
            "root": root.path if root else None,
            "kind": kind,
            "included": reason is None,
            "reason": reason,
            "source": verdict.source if verdict.ignored else None,
            "status": row.status if row else None,
            "pinned": bool(row and row.pinned),
            "error": row.error if row else None,
        }

    def status(self) -> dict[str, Any]:
        embedder = self.embedder
        job = None
        if self.job:
            run: Run = self.job["run"]
            job = run.progress()
        return {
            "pid": os.getpid(),
            "uptime": round(time.time() - self.started, 1),
            "home": str(self.paths.home),
            "model": {
                "id": self.config.get("model.id"),
                "loaded": bool(getattr(embedder, "loaded", True)),
                "device": getattr(embedder, "device", None),
                "dtype": getattr(embedder, "dtype", None),
                "gpu_memory_mb": embedder.gpu_memory_mb() if hasattr(embedder, "gpu_memory_mb") else None,
                "capabilities": list(self.config.get("model.capabilities")),
                "dimension": self.config.get("model.dimension"),
            },
            "index": {
                **self.store.counts(),
                "vectors": len(self.index),
                "db_bytes": self.paths.index_bytes(),
            },
            "roots": self.store.root_summaries(),
            "errors": self.store.error_files(),
            "job": job,
        }
