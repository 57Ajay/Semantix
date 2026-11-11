from __future__ import annotations

import threading

import numpy as np

from .filetypes import KINDS

KIND_CODES = {kind: code for code, kind in enumerate(KINDS)}
CODE_KIND = KIND_CODES["code"]


class VectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._lock = threading.RLock()
        self._size = 0
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._ids = np.zeros(0, dtype=np.int64)
        self._files = np.zeros(0, dtype=np.int64)
        self._kinds = np.zeros(0, dtype=np.uint8)
        self._alive = np.zeros(0, dtype=bool)

    def __len__(self) -> int:
        with self._lock:
            return int(self._alive[: self._size].sum())

    def _reserve(self, extra: int) -> None:
        needed = self._size + extra
        capacity = len(self._ids)
        if needed <= capacity:
            return
        capacity = max(needed, capacity * 2, 1024)
        vectors = np.zeros((capacity, self.dimension), dtype=np.float32)
        vectors[: self._size] = self._vectors[: self._size]
        self._vectors = vectors
        for name, dtype in (("_ids", np.int64), ("_files", np.int64), ("_kinds", np.uint8), ("_alive", bool)):
            grown = np.zeros(capacity, dtype=dtype)
            grown[: self._size] = getattr(self, name)[: self._size]
            setattr(self, name, grown)

    def add(self, ids, file_ids, kinds, vectors: np.ndarray) -> None:
        count = len(ids)
        if not count:
            return
        with self._lock:
            self._reserve(count)
            end = self._size + count
            self._vectors[self._size : end] = vectors
            self._ids[self._size : end] = ids
            self._files[self._size : end] = file_ids
            self._kinds[self._size : end] = [KIND_CODES[kind] for kind in kinds]
            self._alive[self._size : end] = True
            self._size = end

    def remove_files(self, file_ids) -> None:
        file_ids = np.asarray(list(file_ids), dtype=np.int64)
        if not len(file_ids):
            return
        with self._lock:
            self._alive[: self._size] &= ~np.isin(self._files[: self._size], file_ids)
            if self._size > 4096 and self._alive[: self._size].sum() < self._size // 2:
                self._compact()

    def _compact(self) -> None:
        keep = self._alive[: self._size]
        count = int(keep.sum())
        self._vectors = np.ascontiguousarray(self._vectors[: self._size][keep])
        self._ids = self._ids[: self._size][keep].copy()
        self._files = self._files[: self._size][keep].copy()
        self._kinds = self._kinds[: self._size][keep].copy()
        self._alive = np.ones(count, dtype=bool)
        self._size = count

    def scores(self, query: np.ndarray, code_query: np.ndarray | None = None) -> np.ndarray:
        with self._lock:
            matrix = self._vectors[: self._size]
            scores = matrix @ query
            if code_query is not None:
                is_code = self._kinds[: self._size] == CODE_KIND
                if is_code.any():
                    scores = np.where(is_code, matrix @ code_query, scores)
            return scores

    def search(
        self,
        query: np.ndarray,
        limit: int,
        code_query: np.ndarray | None = None,
        kinds: list[str] | None = None,
        file_ids: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        with self._lock:
            if not self._size:
                return []
            scores = self.scores(query, code_query)
            mask = self._alive[: self._size].copy()
            if kinds is not None:
                codes = np.asarray([KIND_CODES[kind] for kind in kinds], dtype=np.uint8)
                mask &= np.isin(self._kinds[: self._size], codes)
            if file_ids is not None:
                mask &= np.isin(self._files[: self._size], file_ids)
            candidates = np.flatnonzero(mask)
            if not len(candidates):
                return []
            picked = scores[candidates]
            limit = min(limit, len(candidates))
            top = np.argpartition(-picked, limit - 1)[:limit]
            top = top[np.argsort(-picked[top])]
            return [(int(self._ids[candidates[i]]), float(picked[i])) for i in top]

    def score_of(self, chunk_ids: list[int], query: np.ndarray, code_query: np.ndarray | None = None) -> dict[int, float]:
        with self._lock:
            if not self._size or not chunk_ids:
                return {}
            positions = np.flatnonzero(np.isin(self._ids[: self._size], np.asarray(chunk_ids, dtype=np.int64)))
            result = {}
            for position in positions:
                vector = self._vectors[position]
                use_code = code_query is not None and self._kinds[position] == CODE_KIND
                result[int(self._ids[position])] = float(vector @ (code_query if use_code else query))
            return result
