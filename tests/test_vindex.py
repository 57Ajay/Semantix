from __future__ import annotations

import numpy as np

from semantix.vindex import VectorIndex


def unit(rows: int, dimension: int = 8, seed: int = 0) -> np.ndarray:
    vectors = np.random.default_rng(seed).standard_normal((rows, dimension)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_first_load_is_adopted_without_a_copy():
    vectors = unit(5000)
    index = VectorIndex(8)
    index.add(np.arange(5000), np.arange(5000) // 10, ["code"] * 5000, vectors)
    assert index._vectors is vectors
    assert len(index) == 5000
    top = index.search(vectors[1234], 1, calibrate=False)
    assert top[0][0] == 1234


def test_growth_after_a_load_stays_modest():
    index = VectorIndex(8)
    index.add(np.arange(4000), np.zeros(4000, dtype=np.int64), ["text"] * 4000, unit(4000))
    index.add([4000], [1], ["image"], unit(1, seed=1))
    assert len(index._ids) == 5000
    assert len(index) == 4001
    extra = unit(1, seed=1)[0]
    assert index.search(extra, 1, kinds=["image"], calibrate=False)[0][0] == 4000


def test_removed_files_disappear_from_results():
    vectors = unit(20)
    index = VectorIndex(8)
    index.add(np.arange(20), np.arange(20) // 5, ["code"] * 20, vectors)
    index.remove_files([0])
    assert len(index) == 15
    assert all(chunk_id >= 5 for chunk_id, _score, _raw in index.search(vectors[0], 20, calibrate=False))
