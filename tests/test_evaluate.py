from __future__ import annotations

import io
import json
import tarfile

import pytest
from conftest import FakeEmbedder

from semantix import evaluate
from semantix.evaluate import EvalError, Judgment, Query


def write_set(root, queries: str, files: dict[str, str] | None = None):
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    for name, text in (files or {}).items():
        target = corpus / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (root / "set.toml").write_text('name = "tiny"\nversion = "1"\n')
    (root / "queries.toml").write_text(queries)
    return root


FILES = {
    "notes/garden.md": "# Garden\n\nPlant tomatoes in spring and water the basil every morning.\n",
    "notes/taxes.md": "# Taxes\n\nFile the quarterly tax return before the deadline in April.\n",
    "code/retry.py": "def retry_with_backoff(call, attempts):\n    for attempt in range(attempts):\n        call()\n",
}

QUERIES = """
[[query]]
id = "garden"
text = "when to plant tomatoes"
category = "notes"
relevant = ["notes/garden.md"]

[[query]]
id = "taxes"
text = "quarterly tax return deadline"
category = "notes"
split = "test"
relevant = [{ path = "notes/taxes.md", lines = [1, 3] }]

[[query]]
id = "retry"
text = "retry with backoff attempts"
category = "code"
relevant = ["code/retry.py"]
acceptable = ["notes/garden.md"]

[[query]]
id = "nothing"
text = "a submarine under the arctic ice"
category = "no-answer"
"""


def hit(path, start_line=None, end_line=None, start_sec=None, end_sec=None, score=0.5, kind="text", part="text"):
    return {
        "path": path, "start_line": start_line, "end_line": end_line, "start_sec": start_sec, "end_sec": end_sec,
        "score": score, "kind": kind, "part": part, "text": "",
    }


def test_load_set_reads_queries(tmp_path):
    eval_set = evaluate.load_set(write_set(tmp_path / "set", QUERIES, FILES))
    assert eval_set.name == "tiny" and eval_set.version == "1"
    assert [query.id for query in eval_set.queries] == ["garden", "taxes", "retry", "nothing"]
    taxes = eval_set.queries[1]
    assert taxes.split == "test" and taxes.relevant == [Judgment("notes/taxes.md", 1.0, (1, 3))]
    assert eval_set.queries[2].judgments[1].grade == 0.5
    assert not eval_set.queries[3].answered


@pytest.mark.parametrize(
    "queries, message",
    [
        ('[[query]]\nid = "a"\ntext = "x"\ncategory = "c"\nrelevant = ["missing.md"]\n', "not in the collection"),
        ('[[query]]\nid = "a"\ntext = "x"\ncategory = "c"\n[[query]]\nid = "a"\ntext = "y"\ncategory = "c"\n', "used twice"),
        ('[[query]]\nid = "a"\ntext = "x"\nlike = "notes/garden.md"\ncategory = "c"\n', "either text or like"),
        ('[[query]]\nid = "a"\ntext = "x"\ncategory = "c"\nsplit = "train"\n', "split must be"),
        ('[[query]]\nid = "a"\ntext = "x"\ncategory = "c"\nrelevant = [{ path = "notes/garden.md", lines = [5, 2] }]\n', "ends before"),
        ('[[query]]\nid = "a"\ntext = "x"\ncategory = "c"\nwrong = 1\n', "unknown field"),
    ],
)
def test_load_set_rejects_mistakes(tmp_path, queries, message):
    with pytest.raises(EvalError, match=message):
        evaluate.load_set(write_set(tmp_path / "set", queries, FILES))


def test_score_query_ranks_grades_and_ranges():
    corpus = "/c"
    query = Query(
        "q", "code", "dev", "text",
        judgments=(
            Judgment("a.py", 1.0, (10, 20)),
            Judgment("b.md", 1.0),
            Judgment("c.md", 0.5),
            Judgment("d.md", 0.0),
        ),
    )
    hits = [
        hit("/c/a.py", 1, 9),
        hit("/c/c.md"),
        hit("/c/d.md"),
        hit("/c/a.py", 15, 30),
        hit("/c/a.py", 18, 25),
        hit("/c/e.md"),
        hit("/c/b.md"),
    ]
    record = evaluate.score_query(query, hits, corpus)
    assert record["rank"] == 4
    assert record["mrr"] == 0.25 and record["hit@1"] == 0.0
    assert record["recall@5"] == 0.5 and record["recall@10"] == 1.0
    assert record["located"] is False
    assert [item["hit"] for item in record["unjudged"]] == ["a.py:1-9"]
    ideal = 1 + 1 / 1.5849625 + 0.5 / 2
    actual = 0.5 / 1.5849625 + 1 / 2.3219281 + 1 / 3
    assert record["ndcg@10"] == pytest.approx(actual / ideal, rel=1e-5)


def test_seconds_ranges_do_not_match_touching_segments():
    judgment = Judgment("v.mp4", 1.0, seconds=(30.0, 45.0))
    assert not evaluate.matches(hit("/c/v.mp4", start_sec=0.0, end_sec=30.0), judgment, "/c")
    assert evaluate.matches(hit("/c/v.mp4", start_sec=28.0, end_sec=56.0), judgment, "/c")


def test_separation_finds_threshold():
    records = [
        {"answered": True, "rank": 1, "top_score": 0.3},
        {"answered": True, "rank": 1, "top_score": 0.2},
        {"answered": True, "rank": 2, "top_score": 0.01},
        {"answered": False, "rank": None, "top_score": 0.05},
        {"answered": False, "rank": None, "top_score": 0.1},
    ]
    result = evaluate.separation(records)
    assert result["correct"] == 4 and result["total"] == 4
    assert 0.1 < result["threshold"] < 0.2


def test_run_scores_and_caches(tmp_path):
    eval_set = evaluate.load_set(write_set(tmp_path / "set", QUERIES, FILES))
    embedder = FakeEmbedder()
    calls = {"query": 0}
    original = embedder.embed_query

    def counted(text):
        calls["query"] += 1
        return original(text)

    embedder.embed_query = counted
    home = tmp_path / "home"
    report = evaluate.run(eval_set, home, tmp_path / "models", modes=evaluate.MODES, embedder=embedder)
    assert report["sync"]["embedded"] == 3 and report["sync"]["failed"] == 0
    hybrid = report["modes"]["hybrid"]
    assert hybrid["overall"]["queries"] == 3
    assert hybrid["overall"]["hit@1"] == 1.0
    assert hybrid["categories"]["notes"]["queries"] == 2
    assert hybrid["no_answer"]["no_answer_queries"] == 1
    assert set(report["modes"]) == set(evaluate.MODES)
    first_calls = calls["query"]
    assert first_calls > 0

    again = evaluate.run(eval_set, home, tmp_path / "models", modes=("hybrid",), split="test", embedder=embedder)
    assert again["sync"]["embedded"] == 0
    assert calls["query"] == first_calls
    assert [record["id"] for record in again["queries"]["hybrid"]] == ["taxes"]

    changed = evaluate.run(
        eval_set, home, tmp_path / "models", {"index.chunk_chars": 400}, embedder=embedder, keyword_bonus=0.0
    )
    assert changed["sync"]["embedded"] == 3
    assert changed["sync"]["reused_from"] is not None
    assert changed["keyword_bonus"] == 0.0


def test_new_settings_reuse_unaffected_types(tmp_path):
    from PIL import Image

    root = write_set(tmp_path / "set", QUERIES, FILES)
    Image.new("RGB", (64, 48), "red").save(root / "corpus" / "red.png")
    eval_set = evaluate.load_set(root)
    home = tmp_path / "home"
    first = evaluate.run(eval_set, home, tmp_path / "models", embedder=FakeEmbedder())
    assert first["sync"]["embedded"] == 4 and first["sync"]["reused_from"] is None
    images = evaluate.run(eval_set, home, tmp_path / "models", {"image.tokens": 70}, embedder=FakeEmbedder())
    assert images["sync"]["embedded"] == 1
    text = evaluate.run(eval_set, home, tmp_path / "models", {"index.chunk_chars": 800}, embedder=FakeEmbedder())
    assert text["sync"]["embedded"] == 3
    assert text["sync"]["parts"] == first["sync"]["parts"]


def test_compare_lists_moved_queries():
    def report(rank, version="1"):
        record = {"id": "q", "answered": True, "rank": rank}
        summary = {"overall": dict.fromkeys(evaluate.METRICS, 1.0 / rank), "categories": {}}
        return {
            "set": {"version": version}, "settings": {"index.chunk_chars": 1600}, "split": "all",
            "modes": {"hybrid": summary}, "queries": {"hybrid": [record]},
        }

    result = evaluate.compare(report(4), report(1, version="2"))
    assert result["modes"]["hybrid"]["moved"] == [{"id": "q", "before": 4, "after": 1}]
    assert result["modes"]["hybrid"]["deltas"]["overall"]["mrr"] == 0.75
    assert any("version 1 to 2" in note for note in result["notes"])


def make_archive(path, members: dict[str, bytes]):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_fetch_set_unpacks_archive(tmp_path, monkeypatch):
    source = write_set(tmp_path / "source", QUERIES, FILES)
    members = {
        f"Semantix-eval/{path.relative_to(source)}": path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    archive = tmp_path / "eval.tar.gz"
    make_archive(archive, members)
    monkeypatch.setattr(evaluate, "SET_ARCHIVE", archive.as_uri())
    target = tmp_path / "home" / "eval" / "set"
    eval_set = evaluate.fetch_set(target)
    assert eval_set.root == target and len(eval_set.queries) == 4
    assert (target / "corpus" / "notes" / "garden.md").is_file()
    assert [path.name for path in target.parent.iterdir()] == ["set"]


def test_extract_rejects_unsafe_paths(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    make_archive(archive, {"Semantix-eval/../escape.txt": b"x"})
    with pytest.raises(EvalError, match="unsafe"):
        evaluate._extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_save_and_find_latest_report(tmp_path):
    report = {"set": {"name": "tiny"}, "value": 1}
    first = evaluate.save_report(report, tmp_path)
    second = evaluate.save_report(report, tmp_path)
    assert first != second
    assert evaluate.latest_report(tmp_path, "tiny") in (first, second)
    assert json.loads(first.read_text()) == report
