from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from semantix.config import Config
from semantix.engine import Engine
from semantix.paths import Paths

WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    device = "cpu"
    dtype = "float32"
    loaded = True

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.calls = 0
        self.texts_embedded = 0

    def text_batch_size(self) -> int:
        return 4

    def media_batch_size(self) -> int:
        return 2

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "little") % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_documents(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        return np.stack([self._vector(text.split("| text: ", 1)[-1]) for text in texts])

    def embed_query(self, text):
        vector = self._vector(text)
        return {"search": vector, "code": vector}

    def embed_images(self, images):
        return np.stack([self._vector(f"image {image.size[0]} {image.size[1]}") for image in images])

    def embed_video(self, frames, fps):
        return self._vector(f"video {len(frames)}")

    def embed_audio(self, samples):
        return self._vector(f"audio {len(samples)}")

    def load(self):
        pass

    def unload(self):
        pass


@pytest.fixture
def home(tmp_path):
    return Paths(tmp_path / "home").ensure()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def engine(home, embedder):
    instance = Engine(home, Config(), embedder)
    yield instance
    instance.close()


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "work" / "project"
    (root / "src").mkdir(parents=True)
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src" / "deploy.py").write_text(
        "import boto3\n\n\ndef deploy_service(cluster, service):\n"
        "    client = boto3.client('ecs')\n    client.update_service(cluster=cluster, service=service)\n"
    )
    (root / "src" / "billing.py").write_text(
        "def invoice_total(items):\n    return sum(item.price * item.quantity for item in items)\n"
    )
    (root / "docs" / "notes.md").write_text("# Notes\n\nRemember to rotate the database credentials monthly.\n")
    (root / "node_modules" / "left-pad" / "index.js").write_text("module.exports = function leftpad() {}\n")
    (root / "debug.log").write_text("log line\n" * 20)
    (root / ".gitignore").write_text("*.log\n")
    return root


def run_sync(engine, targets=None, selection=None, groups=None, skip=None, force=False):
    plan = engine.plan(targets, selection, force)
    events = list(engine.execute(plan["id"], groups, skip))
    return plan, events[-1]
