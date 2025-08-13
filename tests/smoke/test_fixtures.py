# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import io
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import pandas as pd
import pytest

from graphrag.query.context_builder.community_context import (
    NO_COMMUNITY_RECORDS_WARNING,
)
from graphrag.storage.blob_pipeline_storage import BlobPipelineStorage

log = logging.getLogger(__name__)

debug = os.environ.get("DEBUG") is not None
gh_pages = os.environ.get("GH_PAGES") is not None

WELL_KNOWN_AZURITE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1"

KNOWN_WARNINGS = [NO_COMMUNITY_RECORDS_WARNING]

env_vars = {
    "BLOB_STORAGE_CONNECTION_STRING": os.getenv(
        "GRAPHRAG_CACHE_CONNECTION_STRING", WELL_KNOWN_AZURITE_CONNECTION_STRING
    ),
    "LOCAL_BLOB_STORAGE_CONNECTION_STRING": WELL_KNOWN_AZURITE_CONNECTION_STRING,
    "GRAPHRAG_CHUNK_SIZE": "1200",
    "GRAPHRAG_CHUNK_OVERLAP": "0",
    "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
    "GRAPHRAG_EMBEDDING_TYPE": "azure_openai_embedding",
    "GRAPHRAG_API_KEY": "2wyINFmywb4HXysCHVlymJsBwH9KjcYVqSDlX0wsGDnfbl25nZ5tJQQJ99BAACYeBjFXJ3w3AAABACOGf2aQ",
    "GRAPHRAG_API_BASE": "https://dc-law-chatbot.openai.azure.com",
    "GRAPHRAG_API_VERSION": "2025-01-01-preview",
    "GRAPHRAG_LLM_DEPLOYMENT_NAME": "gpt-35-turbo",
    "GRAPHRAG_LLM_MODEL": "gpt-35-turbo",
    "GRAPHRAG_LLM_TPM": "1000",
    "GRAPHRAG_LLM_RPM": "1000",
    "GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME": "text-embedding-3-small",
    "GRAPHRAG_EMBEDDING_MODEL": "text-embedding-ada-002",
}
env_vars = {k: v for k, v in env_vars.items() if v is not None}


def _load_fixtures():
    log.debug("Loading test fixtures...")
    params = []
    fixtures_path = Path("./tests/fixtures/")
    subfolders = ["min-csv"] if gh_pages else sorted(os.listdir(fixtures_path))
    log.debug("Subfolders found: %s", subfolders)

    for subfolder in subfolders:
        full_path = fixtures_path / subfolder
        if not os.path.isdir(full_path):
            log.debug("Skipping non-directory: %s", full_path)
            continue

        config_file = full_path / "config.json"
        log.debug("Reading config from %s", config_file)
        params.append((subfolder, json.loads(config_file.read_bytes().decode("utf-8"))))

    log.debug("Loaded %d fixture(s)", len(params[1:]))
    return params[1:]


def pytest_generate_tests(metafunc):
    log.debug("Generating pytest tests for: %s", metafunc.function.__name__)
    run_slow = metafunc.config.getoption("run_slow")
    configs = metafunc.cls.params[metafunc.function.__name__]

    if not run_slow:
        configs = [config for config in configs if not config[1].get("slow", False)]
        log.debug("Filtered out slow configs; remaining: %d", len(configs))

    funcarglist = [params[1] for params in configs]
    id_list = [params[0] for params in configs]
    argnames = sorted(arg for arg in funcarglist[0] if arg != "slow")
    log.debug("Parametrizing test with args: %s", argnames)

    metafunc.parametrize(
        argnames,
        [[funcargs[name] for name in argnames] for funcargs in funcarglist],
        ids=id_list,
    )


def cleanup(skip: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log.debug("Entering cleanup wrapper for %s", func.__name__)
            try:
                return func(*args, **kwargs)
            except AssertionError as e:
                log.error("Assertion failed in test: %s", e)
                raise
            finally:
                if not skip:
                    log.debug("Performing cleanup")
                    root = Path(kwargs["input_path"])
                    log.debug("Cleaning output and cache in: %s", root)
                    shutil.rmtree(root / "output", ignore_errors=True)
                    shutil.rmtree(root / "cache", ignore_errors=True)
        return wrapper
    return decorator


async def prepare_azurite_data(input_path: str, azure: dict) -> Callable[[], None]:
    log.info("Preparing Azurite data for: %s", input_path)
    input_container = azure["input_container"]
    input_base_dir = azure.get("input_base_dir")

    root = Path(input_path)
    input_storage = BlobPipelineStorage(
        connection_string=WELL_KNOWN_AZURITE_CONNECTION_STRING,
        container_name=input_container,
    )

    log.debug("Deleting and recreating container: %s", input_container)
    input_storage._delete_container()
    input_storage._create_container()

    data_files = list((root / "input").glob("*.txt")) + list((root / "input").glob("*.csv"))
    log.debug("Found %d data file(s)", len(data_files))

    for data_file in data_files:
        text = data_file.read_bytes().decode("utf-8")
        file_path = (
            str(Path(input_base_dir) / data_file.name)
            if input_base_dir
            else data_file.name
        )
        log.debug("Uploading file: %s as %s", data_file.name, file_path)
        await input_storage.set(file_path, text, encoding="utf-8")

    log.info("Azurite data preparation complete")
    return lambda: input_storage._delete_container()


class TestIndexer:
    params: ClassVar[dict[str, list[tuple[str, dict[str, Any]]]]] = {
        "test_fixture": _load_fixtures()
    }

    def __run_indexer(self, root: Path, input_file_type: str):
        log.info("Running indexer on %s (type=%s)", root, input_file_type)
        command = [
            sys.executable,
            "-m",
            "graphrag",
            "index",
            "--verbose" if debug else None,
            "--root",
            root.resolve().as_posix(),
            "--logger",
            "print",
            "--method",
            "standard",
        ]
        command = [arg for arg in command if arg]
        log.info("Command: %s", " ".join(command))

        env = os.environ.copy()
        env["GRAPHRAG_INPUT_FILE_TYPE"] = input_file_type

        completion = subprocess.run(command, env=env)
        log.debug("Subprocess completed with return code: %d", completion.returncode)
        assert completion.returncode == 0, f"Indexer failed with return code: {completion.returncode}"

    @cleanup(skip=debug)
    @mock.patch.dict(os.environ, env_vars)
    @pytest.mark.timeout(800)
    def test_fixture(self, input_path: str, input_file_type: str, workflow_config: dict[str, dict[str, Any]], query_config: list[dict[str, str]]):
        log.info("Starting test_fixture with input_path=%s, input_file_type=%s", input_path, input_file_type)

        if workflow_config.get("skip"):
            log.warning("Skipping test for: %s", input_path)
            print(f"skipping smoke test {input_path})")
            return

        azure = workflow_config.get("azure")
        root = Path(input_path)
        dispose = None

        if azure is not None:
            log.info("Detected Azure config, preparing data...")
            dispose = asyncio.run(prepare_azurite_data(input_path, azure))

        print("running indexer")
        self.__run_indexer(root, input_file_type)

        if dispose:
            log.info("Cleaning up Azure container")
            dispose()

        log.info("test_fixture completed successfully for: %s", input_path)
