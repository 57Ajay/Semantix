from __future__ import annotations

from semantix.ignore import IgnoreRules
from semantix.scanner import Scanner, normalize_selection


def make_tree(root):
    (root / "src").mkdir(parents=True)
    (root / "shots").mkdir()
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "src" / "nested").mkdir()
    (root / ".git").mkdir()
    for name in ("src/app.py", "src/secret_keys.py", "src/nested/deep.py", "shots/one.png", "node_modules/pkg/i.js",
                 "trace.log", ".env", ".env.example", "notes.md"):
        (root / name).write_text("content\n")
    (root / ".gitignore").write_text("shots/\n*.log\n")
    (root / ".semantixignore").write_text("!shots/\nsecret*\n")
    (root / "src" / "nested" / ".semantixignore").write_text("deep.py\n")


def test_layers_and_precedence(tmp_path):
    make_tree(tmp_path)
    rules = IgnoreRules(str(tmp_path))

    def ignored(relative, is_dir=False):
        return rules.check(str(tmp_path / relative), is_dir).ignored

    assert not ignored("src/app.py")
    assert ignored("src/secret_keys.py")
    assert ignored("node_modules", True)
    assert ignored("trace.log")
    assert not ignored("shots", True)
    assert ignored(".env") and not ignored(".env.example")
    assert ignored("src/nested/deep.py")
    assert rules.explain(str(tmp_path / "node_modules/pkg/i.js")).source == "built-in rules"
    assert rules.explain(str(tmp_path / "trace.log")).source.endswith(".gitignore")


def test_global_ignore_and_gitignore_switch(tmp_path):
    make_tree(tmp_path / "project")
    global_ignore = tmp_path / "ignore"
    global_ignore.write_text("*.md\n")
    root = str(tmp_path / "project")
    rules = IgnoreRules(root, global_ignore=global_ignore, respect_gitignore=False)
    assert rules.check(root + "/notes.md", False).ignored
    assert not rules.check(root + "/trace.log", False).ignored


def test_parent_ignore_files_apply_to_a_nested_root(tmp_path):
    make_tree(tmp_path)
    rules = IgnoreRules(str(tmp_path / "src"))
    assert rules.check(str(tmp_path / "src" / "secret_keys.py"), False).ignored
    assert not rules.check(str(tmp_path / "src" / "app.py"), False).ignored


def test_scanner_applies_selection_and_capabilities(tmp_path):
    make_tree(tmp_path)
    root = str(tmp_path)
    selection = normalize_selection({"skip_ext": ["md"], "exclude": ["nested/"]})
    scanner = Scanner(IgnoreRules(root, exclude=selection["exclude"]), selection, ("text",))
    result = scanner.scan(root)
    names = sorted(candidate.path[len(root) + 1 :] for candidate in result.candidates)
    assert names == [".env.example", ".gitignore", ".semantixignore", "src/app.py"]
    assert result.off_kinds == {"image": 1}
    assert result.skipped["skipped extension"] == 1
