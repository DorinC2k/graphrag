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

# cspell:disable-next-line well-known-key
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
    """Decorator to cleanup the output and cache folders after each test."""

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

        # Check all workflows run
        expected_workflows = set(workflow_config.keys())
        workflows = set(stats["workflows"].keys())
        assert workflows == expected_workflows, (
            f"Workflows missing from stats.json: {expected_workflows - workflows}. Unexpected workflows in stats.json: {workflows - expected_workflows}"
        )

        # [OPTIONAL] Check runtime
        for workflow, config in workflow_config.items():
            # Check expected artifacts
            workflow_artifacts = config.get("expected_artifacts", [])
            # Check max runtime
            max_runtime = config.get("max_runtime", None)
            if max_runtime:
                assert stats["workflows"][workflow]["overall"] <= max_runtime, (
                    f"Expected max runtime of {max_runtime}, found: {stats['workflows'][workflow]['overall']} for workflow: {workflow}"
                )
            # Check expected artifacts
            for artifact in workflow_artifacts:
                if artifact.endswith(".parquet"):
                    output_df = pd.read_parquet(output_path / artifact)

                    # Check number of rows between range
                    assert (
                        config["row_range"][0]
                        <= len(output_df)
                        <= config["row_range"][1]
                    ), (
                        f"Expected between {config['row_range'][0]} and {config['row_range'][1]}, found: {len(output_df)} for file: {artifact}"
                    )

                    # Get non-nan rows
                    nan_df = output_df.loc[
                        :,
                        ~output_df.columns.isin(config.get("nan_allowed_columns", [])),
                    ]
                    nan_df = nan_df[nan_df.isna().any(axis=1)]
                    assert len(nan_df) == 0, (
                        f"Found {len(nan_df)} rows with NaN values for file: {artifact} on columns: {nan_df.columns[nan_df.isna().any()].tolist()}"
                    )

    def __run_query(self, root: Path, query_config: dict[str, str]):
        command = [
            "poetry",
            "run",
            "poe",
            "query",
            "--root",
            root.resolve().as_posix(),
            "--method",
            query_config["method"],
            "--community-level",
            str(query_config.get("community_level", 2)),
            "--query",
            query_config["query"],
        ]

        log.info("running command ", " ".join(command))
        return subprocess.run(command, capture_output=True, text=True)

    @cleanup(skip=debug)
    @mock.patch.dict(
        os.environ,
        {
            **os.environ,
            "BLOB_STORAGE_CONNECTION_STRING": os.getenv(
                "GRAPHRAG_CACHE_CONNECTION_STRING", WELL_KNOWN_AZURITE_CONNECTION_STRING
            ),
            "LOCAL_BLOB_STORAGE_CONNECTION_STRING": WELL_KNOWN_AZURITE_CONNECTION_STRING,
            "GRAPHRAG_CHUNK_SIZE": "1200",
            "GRAPHRAG_CHUNK_OVERLAP": "0",
            "AZURE_AI_SEARCH_URL_ENDPOINT": os.getenv("AZURE_AI_SEARCH_URL_ENDPOINT"),
            "AZURE_AI_SEARCH_API_KEY": os.getenv("AZURE_AI_SEARCH_API_KEY"),
        },
        clear=True,
    )
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
        print("indexer complete")

        if dispose:
            log.info("Cleaning up Azure container")
            dispose()

        if not workflow_config.get("skip_assert"):
            print("performing dataset assertions")
            self.__assert_indexer_outputs(root, workflow_config)

        print("running queries")
        for query in query_config:
            result = self.__run_query(root, query)
            print(f"Query: {query}\nResponse: {result.stdout}")

            assert result.returncode == 0, "Query failed"
            assert result.stdout is not None, "Query returned no output"
            assert len(result.stdout) > 0, "Query returned empty output"
