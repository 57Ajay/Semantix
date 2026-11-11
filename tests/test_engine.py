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
