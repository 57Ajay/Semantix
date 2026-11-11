from __future__ import annotations

import os

from conftest import run_sync


def paths_of(result):
    return [os.path.basename(hit["path"]) for hit in result["hits"]]


def test_first_sync_respects_ignore_rules(engine, project):
    plan, done = run_sync(engine, [str(project)])
    assert plan["totals"]["files"] == 4
    assert done["embedded"] == 4 and done["failed"] == 0
    tracked = {os.path.basename(row.path) for row in engine.store.files_under(str(project))}
    assert tracked == {"deploy.py", "billing.py", "notes.md", ".gitignore"}


def test_search_finds_the_right_file(engine, project):
    run_sync(engine, [str(project)])
    result = engine.search("update ecs service cluster")
    assert paths_of(result)[0] == "deploy.py"
    assert result["hits"][0]["start_line"] == 1
    only_text = engine.search("rotate database credentials", kinds=["text"])
    assert paths_of(only_text) == ["notes.md"]


def test_second_sync_is_a_no_op(engine, project, embedder):
    run_sync(engine, [str(project)])
    before = embedder.texts_embedded
    plan, done = run_sync(engine)
    assert plan["totals"]["files"] == 0 and plan["totals"]["unchanged"] == 4
    assert embedder.texts_embedded == before and done["embedded"] == 0


def test_changed_file_is_embedded_again(engine, project):
    run_sync(engine, [str(project)])
    target = project / "src" / "billing.py"
    target.write_text(target.read_text() + "\n\ndef apply_discount(total, percent):\n    return total * (1 - percent / 100)\n")
    plan, done = run_sync(engine)
    assert plan["totals"]["files"] == 1 and plan["roots"][0]["groups"][0]["changed"] == 1
    assert done["embedded"] == 1
    assert paths_of(engine.search("apply discount percent"))[0] == "billing.py"


def test_touch_without_change_is_not_embedded(engine, project, embedder):
    run_sync(engine, [str(project)])
    target = project / "src" / "billing.py"
    os.utime(target, (1_000_000, 1_000_000))
    before = embedder.texts_embedded
    plan, _done = run_sync(engine)
    assert plan["totals"]["files"] == 0 and embedder.texts_embedded == before


def test_relative_file_inside_tracked_root_is_not_duplicated(engine, project):
    run_sync(engine, [str(project)])
    fresh = project / "src" / "newfile.py"
    fresh.write_text("def brand_new_helper():\n    return 'fresh'\n")
    plan, done = run_sync(engine, [str(fresh)])
    assert done["embedded"] == 1
    assert [root.path for root in engine.store.roots()] == [str(project)]
    assert plan["roots"][0]["new"] is False
    assert len([row for row in engine.store.files_under(str(project)) if row.path == str(fresh)]) == 1


def test_deleted_and_moved_files(engine, project, embedder):
    run_sync(engine, [str(project)])
    (project / "docs" / "notes.md").unlink()
    os.rename(project / "src" / "deploy.py", project / "src" / "release.py")
    before = embedder.texts_embedded
    plan, done = run_sync(engine)
    assert plan["totals"]["deletes"] == 1 and plan["totals"]["moves"] == 1
    assert done["embedded"] == 0 and embedder.texts_embedded == before
    assert paths_of(engine.search("update ecs service cluster"))[0] == "release.py"
    assert "notes.md" not in paths_of(engine.search("rotate database credentials"))


def test_deferred_items_become_pending(engine, project):
    plan = engine.plan([str(project)])
    done = list(engine.execute(plan["id"], groups=[[str(project), "text"]]))[-1]
    assert done["embedded"] == 1 and done["deferred"] == 3
    summary = engine.store.root_summaries()[0]
    assert summary["waiting"] == 3
    _plan, done = run_sync(engine)
    assert done["embedded"] == 3


def test_selection_is_remembered(engine, project):
    run_sync(engine, [str(project)], selection={"kinds": ["code"], "skip_ext": ["md"]})
    (project / "docs" / "extra.md").write_text("another note\n")
    plan, _done = run_sync(engine)
    assert plan["totals"]["files"] == 0
    kinds = {row.kind for row in engine.store.files_under(str(project))}
    assert kinds == {"code"}


def test_parent_root_absorbs_child_root(engine, project):
    run_sync(engine, [str(project / "src")])
    plan, done = run_sync(engine, [str(project)])
    assert [root.path for root in engine.store.roots()] == [str(project)]
    assert done["embedded"] == 2 and plan["totals"]["unchanged"] == 2


def test_explicit_ignored_file_is_pinned(engine, project):
    run_sync(engine, [str(project)])
    log_file = project / "debug.log"
    _plan, done = run_sync(engine, [str(log_file)])
    assert done["embedded"] == 1
    plan, _done = run_sync(engine)
    assert plan["totals"]["deletes"] == 0
    assert engine.check(str(log_file))["included"] is True


def test_forget_subpath_stays_forgotten(engine, project):
    run_sync(engine, [str(project)])
    result = engine.forget(str(project / "docs"))
    assert result["files"] == 1 and result["excluded"] == "/docs"
    plan, _done = run_sync(engine)
    assert plan["totals"]["files"] == 0
    assert engine.forget(str(project))["root_removed"] is True
    assert engine.store.roots() == [] and len(engine.index) == 0


def test_index_survives_restart(home, embedder, project):
    from semantix.config import Config
    from semantix.engine import Engine

    first = Engine(home, Config(), embedder)
    run_sync(first, [str(project)])
    count = len(first.index)
    first.close()
    second = Engine(home, Config(), embedder)
    assert len(second.index) == count
    assert paths_of(second.search("update ecs service cluster"))[0] == "deploy.py"
    second.close()


def test_missing_root_is_not_wiped(engine, project, tmp_path):
    run_sync(engine, [str(project)])
    os.rename(project, tmp_path / "elsewhere")
    plan = engine.plan()
    assert plan["totals"]["deletes"] == 0 and "not available" in plan["notes"][0]


def test_failed_file_is_retried_only_after_it_changes(engine, project):
    broken = project / "docs" / "broken.txt"
    broken.write_bytes(b"\x00\x01\x02" * 50)
    _plan, done = run_sync(engine, [str(project)])
    assert done["failed"] == 1 and done["embedded"] == 4
    plan = engine.plan()
    assert plan["totals"]["files"] == 0 and plan["roots"][0]["failed"] == 1
    assert engine.status()["errors"][0]["path"] == str(broken)
    broken.write_text("now it is readable text about broken things\n")
    plan, done = run_sync(engine)
    assert plan["totals"]["files"] == 1 and done["embedded"] == 1 and done["failed"] == 0
    assert engine.status()["errors"] == []


def test_missing_target_is_reported(engine, project):
    run_sync(engine, [str(project)])
    plan = engine.plan([str(project / "nope"), "/definitely/not/here"])
    assert len(plan["notes"]) == 2 and all("does not exist" in note for note in plan["notes"])
    assert plan["roots"] == []


def test_duplicate_media_reuses_vectors(engine, tmp_path, embedder, monkeypatch):
    from PIL import Image

    folder = tmp_path / "pics"
    folder.mkdir()
    Image.new("RGB", (40, 30), "red").save(folder / "one.png")
    (folder / "two.png").write_bytes((folder / "one.png").read_bytes())
    calls = []
    original = embedder.embed_images
    monkeypatch.setattr(embedder, "embed_images", lambda images: calls.append(len(images)) or original(images))
    _plan, done = run_sync(engine, [str(folder)])
    assert done["embedded"] == 2 and sum(calls) == 1
    assert len(engine.search(like=str(folder / "one.png"))["hits"]) == 1


def test_capability_off_keeps_existing_vectors(home, embedder, tmp_path):
    from PIL import Image

    from semantix.config import Config
    from semantix.engine import Engine

    folder = tmp_path / "mixed"
    folder.mkdir()
    Image.new("RGB", (20, 20), "blue").save(folder / "pic.png")
    (folder / "note.md").write_text("a note\n")
    first = Engine(home, Config(), embedder)
    run_sync(first, [str(folder)])
    first.close()
    second = Engine(home, Config({"model.capabilities": "text"}), embedder)
    plan = second.plan()
    assert plan["totals"]["deletes"] == 0 and plan["off_kinds"] == {"image": 1}
    assert len(second.index) == 2
    second.close()


def test_lower_dimension_truncates_and_higher_resets(home, embedder, project):
    from semantix.config import Config
    from semantix.engine import Engine

    first = Engine(home, Config(), embedder)
    run_sync(first, [str(project)])
    first.close()
    embedder.dimension = 256
    smaller = Engine(home, Config({"model.dimension": 256}), embedder)
    assert smaller.store.stored_dimension() == 256 and len(smaller.index) == 4
    assert paths_of(smaller.search("update ecs service cluster"))[0] == "deploy.py"
    smaller.close()
    embedder.dimension = 768
    bigger = Engine(home, Config(), embedder)
    assert len(bigger.index) == 0 and bigger.store.root_summaries()[0]["waiting"] == 4
    bigger.close()


def two_locations(tmp_path):
    body = "def handler(event):\n    return process(event)\n" * 5
    inbox, archive = tmp_path / "inbox", tmp_path / "archive"
    inbox.mkdir()
    archive.mkdir()
    (inbox / "one.py").write_text(body + "# one\n")
    (inbox / "stay.py").write_text(body + "# stay\n")
    (archive / "keep.py").write_text(body + "# keep\n")
    return inbox, archive


def test_move_between_tracked_locations_keeps_vectors(engine, tmp_path, embedder):
    inbox, archive = two_locations(tmp_path)
    run_sync(engine, [str(inbox)])
    run_sync(engine, [str(archive)])
    parts = len(engine.index)
    os.rename(inbox / "one.py", archive / "one.py")
    before = embedder.texts_embedded
    plan, done = run_sync(engine)
    assert plan["totals"]["moves"] == 1 and plan["totals"]["deletes"] == 0 and plan["totals"]["files"] == 0
    assert done["embedded"] == 0 and embedder.texts_embedded == before and len(engine.index) == parts
    moved = engine.store.file_by_path(str(archive / "one.py"))
    assert moved is not None and engine.store.root_containing(moved.path).path == str(archive)
    assert engine.store.file_by_path(str(inbox / "one.py")) is None
    hits = engine.search("handler process event one", under=str(archive), per_file=0)["hits"]
    assert str(archive / "one.py") in {hit["path"] for hit in hits}


def test_move_is_found_when_only_the_destination_is_synced(engine, tmp_path, embedder):
    inbox, archive = two_locations(tmp_path)
    run_sync(engine, [str(inbox)])
    run_sync(engine, [str(archive)])
    os.rename(inbox / "one.py", archive / "one.py")
    before = embedder.texts_embedded
    plan, _done = run_sync(engine, [str(archive)])
    assert plan["totals"]["moves"] == 1 and embedder.texts_embedded == before
    plan = engine.plan()
    assert plan["totals"]["deletes"] == 0 and plan["totals"]["files"] == 0


def test_a_copy_is_not_treated_as_a_move(engine, tmp_path):
    inbox, archive = two_locations(tmp_path)
    run_sync(engine, [str(inbox)])
    run_sync(engine, [str(archive)])
    (archive / "one.py").write_bytes((inbox / "one.py").read_bytes())
    plan, done = run_sync(engine)
    assert plan["totals"]["moves"] == 0 and done["embedded"] == 1
    assert engine.store.file_by_path(str(inbox / "one.py")) is not None
    assert engine.store.file_by_path(str(archive / "one.py")) is not None


def test_unavailable_location_is_not_raided_for_moves(engine, tmp_path):
    inbox, archive = two_locations(tmp_path)
    run_sync(engine, [str(inbox)])
    run_sync(engine, [str(archive)])
    (archive / "one.py").write_bytes((inbox / "one.py").read_bytes())
    os.rename(inbox, tmp_path / "unplugged")
    plan = engine.plan([str(archive)])
    assert plan["totals"]["moves"] == 0 and plan["totals"]["files"] == 1
