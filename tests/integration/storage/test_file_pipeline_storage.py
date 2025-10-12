# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Blob Storage Tests."""

import os
import re
from datetime import datetime
from pathlib import Path

from graphrag.storage.file_pipeline_storage import (
    FilePipelineStorage,
)

__dirname__ = os.path.dirname(__file__)


async def test_find():
    storage = FilePipelineStorage()
    items = list(
        storage.find(
            base_dir="tests/fixtures/text/input",
            file_pattern=re.compile(r".*\.txt$"),
            progress=None,
            file_filter=None,
        )
    )
    assert items == [
        (
            Path("tests/fixtures/text/input/dulce.txt").as_posix(),
            {},
        )
    ]
    output = await storage.get("tests/fixtures/text/input/dulce.txt")
    assert len(output) > 0

    await storage.set("test.txt", "Hello, World!", encoding="utf-8")
    output = await storage.get("test.txt")
    assert output == "Hello, World!"
    await storage.delete("test.txt")
    output = await storage.get("test.txt")
    assert output is None


def test_find_respects_base_dir(tmp_path):
    storage = FilePipelineStorage(root_dir=str(tmp_path))

    base_dir = "input"
    (tmp_path / base_dir).mkdir()
    (tmp_path / base_dir / "a.txt").write_text("A")
    (tmp_path / f"{base_dir}file").mkdir()
    (tmp_path / f"{base_dir}file" / "b.txt").write_text("B")

    items = list(
        storage.find(
            base_dir=base_dir,
            file_pattern=re.compile(r".*\.txt$"),
        )
    )
    items = [item[0] for item in items]
    assert items == [f"{base_dir}/a.txt"]


def test_find_respects_max_count(tmp_path):
    storage = FilePipelineStorage(root_dir=str(tmp_path))

    for idx in range(3):
        (tmp_path / f"file_{idx}.txt").write_text(f"file-{idx}")

    items = list(
        storage.find(
            file_pattern=re.compile(r".*\.txt$"),
            max_count=2,
        )
    )

    assert len(items) == 2


def test_find_respects_total_size_limit(tmp_path):
    storage = FilePipelineStorage(root_dir=str(tmp_path))

    tiny_file = tmp_path / "tiny.txt"
    small_file = tmp_path / "small.txt"
    large_file = tmp_path / "large.txt"
    tiny_file.write_bytes(b"a" * 256 * 1024)
    small_file.write_bytes(b"b" * 512 * 1024)
    large_file.write_bytes(b"c" * 2 * 1024 * 1024)

    items = list(
        storage.find(
            file_pattern=re.compile(r".*\.txt$"),
            max_total_size_mb=1,
        )
    )

    assert sorted(items) == sorted(
        [(tiny_file.name, {}), (small_file.name, {})]
    )


async def test_get_creation_date():
    storage = FilePipelineStorage()

    creation_date = await storage.get_creation_date(
        "tests/fixtures/text/input/dulce.txt"
    )

    datetime_format = "%Y-%m-%d %H:%M:%S %z"
    parsed_datetime = datetime.strptime(creation_date, datetime_format).astimezone()

    assert parsed_datetime.strftime(datetime_format) == creation_date


async def test_child():
    storage = FilePipelineStorage()
    storage = storage.child("tests/fixtures/text/input")
    items = list(storage.find(re.compile(r".*\.txt$")))
    assert items == [(str(Path("dulce.txt")), {})]

    output = await storage.get("dulce.txt")
    assert len(output) > 0

    await storage.set("test.txt", "Hello, World!", encoding="utf-8")
    output = await storage.get("test.txt")
    assert output == "Hello, World!"
    await storage.delete("test.txt")
    output = await storage.get("test.txt")
    assert output is None
