from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec

BUILTIN_PATTERNS = """
.git/
.hg/
.svn/
.semantix/
node_modules/
bower_components/
.venv/
venv/
__pycache__/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.tox/
.nox/
.cache/
.gradle/
.idea/
.next/
.nuxt/
.svelte-kit/
.turbo/
.parcel-cache/
.terraform/
.serverless/
.nyc_output/
coverage/
site-packages/
.eggs/
*.egg-info/
dist/
build/
target/
.DS_Store
Thumbs.db
*.min.js
*.min.css
*.map
*.lock
package-lock.json
pnpm-lock.yaml
go.sum
*.svg
*.pyc
*.pyo
*.class
*.o
*.a
*.so
*.dylib
*.dll
*.exe
*.bin
*.wasm
*.zip
*.tar
*.gz
*.bz2
*.xz
*.7z
*.rar
*.jar
*.iso
*.dmg
*.sqlite
*.sqlite3
*.db
*.parquet
*.npy
*.npz
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.h5
.env
.env.*
!.env.example
!.env.sample
!.env.template
*.pem
*.key
*.p12
*.pfx
*.keystore
id_rsa
id_dsa
id_ecdsa
id_ed25519
""".split()

GLOBAL_IGNORE_TEMPLATE = """\
# Global Semantix ignore rules, applied to every directory you sync.
# Same syntax as .gitignore. Use !pattern to bring something back.
"""


@dataclass(frozen=True)
class Layer:
    source: str
    base: str
    spec: GitIgnoreSpec


@dataclass(frozen=True)
class Verdict:
    ignored: bool
    source: str | None = None


def _read_spec(path: str) -> GitIgnoreSpec | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    spec = GitIgnoreSpec.from_lines(lines)
    return spec if len(spec) else None


def _relative(path: str, base: str) -> str | None:
    if base == "/":
        return path.lstrip("/")
    if path == base:
        return ""
    prefix = base + "/"
    return path[len(prefix) :] if path.startswith(prefix) else None


class IgnoreRules:
    def __init__(
        self,
        root: str,
        global_ignore: Path | None = None,
        respect_gitignore: bool = True,
        exclude: list[str] | None = None,
    ):
        self.root = root
        self.respect_gitignore = respect_gitignore
        anchor = root if os.path.isdir(root) else os.path.dirname(root)
        self.anchor = anchor
        self._builtin = Layer("built-in rules", anchor, GitIgnoreSpec.from_lines(BUILTIN_PATTERNS))
        self._global: Layer | None = None
        if global_ignore is not None:
            spec = _read_spec(str(global_ignore))
            if spec:
                self._global = Layer(str(global_ignore), anchor, spec)
        self._cli: Layer | None = None
        if exclude:
            self._cli = Layer("exclude patterns", anchor, GitIgnoreSpec.from_lines(exclude))
        self._git_above: list[Layer] = []
        self._own_above: list[Layer] = []
        self._load_ancestors(anchor)
        self._git_cache: dict[str, list[Layer]] = {}
        self._own_cache: dict[str, list[Layer]] = {}

    def _load_ancestors(self, anchor: str) -> None:
        ancestors = []
        current = os.path.dirname(anchor)
        while True:
            ancestors.append(current)
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        repo_top = None
        for directory in [anchor, *ancestors]:
            if os.path.exists(os.path.join(directory, ".git")):
                repo_top = directory
                break
        for directory in reversed(ancestors):
            own = _read_spec(os.path.join(directory, ".semantixignore"))
            if own:
                self._own_above.append(Layer(os.path.join(directory, ".semantixignore"), directory, own))
            inside_repo = repo_top is not None and (
                directory == repo_top or directory.startswith(repo_top.rstrip("/") + "/")
            )
            if self.respect_gitignore and inside_repo:
                git = _read_spec(os.path.join(directory, ".gitignore"))
                if git:
                    self._git_above.append(Layer(os.path.join(directory, ".gitignore"), directory, git))

    def _dir_layers(self, directory: str) -> tuple[list[Layer], list[Layer]]:
        cached = self._git_cache.get(directory)
        if cached is not None:
            return cached, self._own_cache[directory]
        if directory == self.anchor:
            git, own = list(self._git_above), list(self._own_above)
        else:
            parent_git, parent_own = self._dir_layers(os.path.dirname(directory))
            git, own = list(parent_git), list(parent_own)
        if self.respect_gitignore:
            spec = _read_spec(os.path.join(directory, ".gitignore"))
            if spec:
                git.append(Layer(os.path.join(directory, ".gitignore"), directory, spec))
        spec = _read_spec(os.path.join(directory, ".semantixignore"))
        if spec:
            own.append(Layer(os.path.join(directory, ".semantixignore"), directory, spec))
        self._git_cache[directory] = git
        self._own_cache[directory] = own
        return git, own

    def check(self, path: str, is_dir: bool) -> Verdict:
        parent = os.path.dirname(path)
        if parent != self.anchor and not parent.startswith(self.anchor.rstrip("/") + "/"):
            parent = self.anchor
        git, own = self._dir_layers(parent)
        layers = [self._builtin, *git]
        if self._global:
            layers.append(self._global)
        layers.extend(own)
        if self._cli:
            layers.append(self._cli)
        verdict = Verdict(False)
        for layer in layers:
            relative = _relative(path, layer.base)
            if not relative:
                continue
            result = layer.spec.check_file(relative + "/" if is_dir else relative)
            if result.include is None:
                continue
            verdict = Verdict(bool(result.include), layer.source)
        return verdict

    def explain(self, path: str) -> Verdict:
        relative = _relative(path, self.anchor)
        if not relative:
            return Verdict(False)
        current = self.anchor
        parts = relative.split("/")
        for index, part in enumerate(parts):
            current = os.path.join(current, part)
            is_last = index == len(parts) - 1
            verdict = self.check(current, is_dir=not is_last or os.path.isdir(current))
            if verdict.ignored:
                return verdict
        return verdict
