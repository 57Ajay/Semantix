from __future__ import annotations

import pytest

from semantix.chunking import NotTextError, chunk_text, document_prompt, read_text


def test_chunks_cover_every_line_in_order():
    source = "\n".join(f"def function_{n}():\n    return {n}\n" for n in range(120))
    chunks = chunk_text(source, 400)
    assert len(chunks) > 5
    assert chunks[0].start_line == 1
    lines = source.splitlines()
    for chunk in chunks:
        assert len(chunk.text) <= 400
        assert "\n".join(lines[chunk.start_line - 1 : chunk.end_line]).strip("\n") == chunk.text
    assert all(later.start_line > earlier.end_line - 1 for earlier, later in zip(chunks, chunks[1:], strict=False))


def test_chunks_break_at_top_level_definitions():
    block = "def handler(event):\n" + "".join(f"    step_{n}(event)\n" for n in range(12))
    source = "\n\n".join(block.replace("handler", f"handler_{n}") for n in range(6))
    chunks = chunk_text(source, 500)
    assert all(chunk.text.startswith("def handler_") for chunk in chunks)


def test_edit_only_changes_nearby_chunks():
    sections = [f"## Section {n}\n\n" + f"Paragraph about topic {n}. " * 12 for n in range(10)]
    before = chunk_text("\n\n".join(sections), 600)
    sections[7] += "\n\nAn extra sentence was added here."
    after = chunk_text("\n\n".join(sections), 600)
    unchanged = {chunk.digest for chunk in before} & {chunk.digest for chunk in after}
    assert len(unchanged) >= len(before) - 2


def test_very_long_line_is_split():
    chunks = chunk_text("a" * 5000, 1000)
    assert [len(chunk.text) for chunk in chunks] == [1000] * 5
    assert {chunk.start_line for chunk in chunks} == {1}


def test_blank_input_gives_no_chunks():
    assert chunk_text("", 500) == []
    assert chunk_text("\n\n   \n", 500) == []


def test_binary_file_is_rejected(tmp_path):
    target = tmp_path / "blob.txt"
    target.write_bytes(b"\x00\x01\x02" * 100)
    with pytest.raises(NotTextError):
        read_text(str(target))


def test_notebook_is_reduced_to_its_cells(tmp_path):
    target = tmp_path / "analysis.ipynb"
    target.write_text('{"cells": [{"source": ["import pandas\\n", "df = load()"]}, {"source": "plot(df)"}]}')
    assert read_text(str(target)) == "import pandas\ndf = load()\n\nplot(df)"


def test_document_prompt_format():
    assert document_prompt("src/app.py", "print(1)") == "title: src/app.py | text: print(1)"
    assert document_prompt("", "x") == "title: none | text: x"
