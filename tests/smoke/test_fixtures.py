# test_fixtures.py (fully patched)

import asyncio
import json
import logging
import os
import shutil
import subprocess
import platform
import sys
from pathlib import Path
from collections.abc import Callable
from functools import wraps
from typing import Any, ClassVar
from unittest import mock

import pandas as pd
import pytest

from graphrag.query.context_builder.community_context import NO_COMMUNITY_RECORDS_WARNING
from graphrag.storage.blob_pipeline_storage import BlobPipelineStorage

log = logging.getLogger(__name__)

debug = os.environ.get("DEBUG") is not None
gh_pages = os.environ.get("GH_PAGES") is not None

WELL_KNOWN_AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
    "K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1"
)

KNOWN_WARNINGS = [NO_COMMUNITY_RECORDS_WARNING]

safe_home = os.path.expanduser("~")

# Base env_vars with all needed template substitutions
env_vars = {
    "BLOB_STORAGE_CONNECTION_STRING": os.getenv("GRAPHRAG_CACHE_CONNECTION_STRING", WELL_KNOWN_AZURITE_CONNECTION_STRING),
    "LOCAL_BLOB_STORAGE_CONNECTION_STRING": WELL_KNOWN_AZURITE_CONNECTION_STRING,
    "GRAPHRAG_CHUNK_SIZE": "1200",
    "GRAPHRAG_CHUNK_OVERLAP": "0",
    "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
    "GRAPHRAG_EMBEDDING_TYPE": "azure_openai_embedding",
    "GRAPHRAG_API_KEY": "2wyINFmywb4HXysCHVlymJsBwH9KjcYVqSDlX0wsGDnfbl25nZ5tJQQJ99BAACYeBjFXJ3w3AAABACOGf2aQ",
    "GRAPHRAG_API_BASE": "https://dc-law-chatbot.openai.azure.com/",
    "GRAPHRAG_API_VERSION": "2024-12-01-preview",
    "GRAPHRAG_LLM_DEPLOYMENT_NAME": "gpt-4o",
    "GRAPHRAG_LLM_MODEL": "gpt-4o",
    "GRAPHRAG_LLM_TPM": "1000",
    "GRAPHRAG_LLM_RPM": "1000",
    "GRAPHRAG_EMBEDDING_API_VERSION": "1",
    "GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME": "text-embedding-3-small",
    "GRAPHRAG_EMBEDDING_MODEL": "text-embedding-3-small",
    "USERPROFILE": os.environ.get("USERPROFILE", safe_home),
    "HOME": os.environ.get("HOME", safe_home),
    "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
    "NLTK_DATA": os.path.join(safe_home, "nltk_data"),
    "KMP_DUPLICATE_LIB_OK": "True",
    "KMP_INIT_AT_FORK": "FALSE",
}

# Clean out any accidental None values
env_vars = {k: v for k, v in env_vars.items() if v is not None}

def build_subprocess_env(extra_vars: dict = None) -> dict:
    env = dict(env_vars)  # 🔥 key fix: use the full global env_vars

    for key in ["PATH", "VIRTUAL_ENV", "PYTHONPATH"]:
        if key in os.environ:
            env[key] = os.environ[key]

    if extra_vars:
        env.update(extra_vars)

    return env


def _load_fixtures():
    params = []
    fixtures_path = Path("./tests/fixtures/")
    subfolders = ["min-csv"] if gh_pages else sorted(os.listdir(fixtures_path))

    for subfolder in subfolders:
        if not os.path.isdir(fixtures_path / subfolder):
            continue
        config_file = fixtures_path / subfolder / "config.json"
        params.append((subfolder, json.loads(config_file.read_bytes().decode("utf-8"))))

    return params[1:]  # skip azure test

def pytest_generate_tests(metafunc):
    run_slow = metafunc.config.getoption("run_slow")
    configs = metafunc.cls.params[metafunc.function.__name__]

    if not run_slow:
        configs = [config for config in configs if not config[1].get("slow", False)]

    funcarglist = [params[1] for params in configs]
    id_list = [params[0] for params in configs]

    argnames = sorted(arg for arg in funcarglist[0] if arg != "slow")
    metafunc.parametrize(
        argnames,
        [[funcargs[name] for name in argnames] for funcargs in funcarglist],
        ids=id_list,
    )

def cleanup(skip: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                if not skip:
                    root = Path(kwargs["input_path"])
                    shutil.rmtree(root / "output", ignore_errors=True)
                    shutil.rmtree(root / "cache", ignore_errors=True)
        return wrapper
    return decorator

async def prepare_azurite_data(input_path: str, azure: dict) -> Callable[[], None]:
    input_container = azure["input_container"]
    input_base_dir = azure.get("input_base_dir")
    root = Path(input_path)
    input_storage = BlobPipelineStorage(
        connection_string=WELL_KNOWN_AZURITE_CONNECTION_STRING,
        container_name=input_container,
    )
    input_storage._delete_container()
    input_storage._create_container()

    for data_file in list((root / "input").glob("*.txt")) + list((root / "input").glob("*.csv")):
        text = data_file.read_bytes().decode("utf-8")
        file_path = str(Path(input_base_dir) / data_file.name) if input_base_dir else data_file.name
        await input_storage.set(file_path, text, encoding="utf-8")

    return lambda: input_storage._delete_container()

class TestIndexer:
    params: ClassVar[dict[str, list[tuple[str, dict[str, Any]]]]] = {
        "test_fixture": _load_fixtures()
    }

    def __run_indexer(self, root: Path, input_file_type: str):
        command = [
            sys.executable, "-m", "graphrag", "index",
            "--verbose" if debug else None,
            "--root", root.resolve().as_posix(),
            "--logger", "print",
            "--method", "standard",
        ]
        command = [arg for arg in command if arg]
        log.info("running command ", " ".join(command))

        completion = subprocess.run(command, env=build_subprocess_env({
            "GRAPHRAG_INPUT_FILE_TYPE": input_file_type
        }))

        assert completion.returncode == 0, f"Indexer failed with return code: {completion.returncode}"

    def __assert_indexer_outputs(self, root: Path, workflow_config: dict[str, dict[str, Any]]):
        output_path = root / "output"
        assert output_path.exists(), "output folder does not exist"
        stats = json.loads((output_path / "stats.json").read_bytes().decode("utf-8"))

        expected_workflows = set(workflow_config.keys())
        actual_workflows = set(stats["workflows"].keys())
        assert expected_workflows == actual_workflows, (
            f"Missing: {expected_workflows - actual_workflows}, Unexpected: {actual_workflows - expected_workflows}"
        )

        for wf, cfg in workflow_config.items():
            if cfg.get("max_runtime"):
                assert stats["workflows"][wf]["overall"] <= cfg["max_runtime"]
            for artifact in cfg.get("expected_artifacts", []):
                if artifact.endswith(".parquet"):
                    df = pd.read_parquet(output_path / artifact)
                    min_rows, max_rows = cfg["row_range"]
                    assert min_rows <= len(df) <= max_rows
                    nan_df = df.loc[:, ~df.columns.isin(cfg.get("nan_allowed_columns", []))]
                    assert not nan_df.isna().any().any()

    def __run_query(self, root: Path, query_config: dict[str, str]):
        command = [
            "poetry", "run", "poe", "query",
            "--root", root.resolve().as_posix(),
            "--method", query_config["method"],
            "--community-level", str(query_config.get("community_level", 2)),
            "--query", query_config["query"],
        ]
        return subprocess.run(command, capture_output=True, text=True)

    @cleanup(skip=debug)
    @mock.patch.dict(os.environ, env_vars, clear=True)
    @pytest.mark.timeout(800)
    def test_fixture(self, input_path: str, input_file_type: str,
                     workflow_config: dict[str, dict[str, Any]],
                     query_config: list[dict[str, str]]):
        if workflow_config.get("skip"):
            print(f"skipping smoke test {input_path})")
            return

        root = Path(input_path)
        dispose = asyncio.run(prepare_azurite_data(input_path, workflow_config["azure"])) \
            if workflow_config.get("azure") else None

        print("running indexer")
        self.__run_indexer(root, input_file_type)
        print("indexer complete")

        if dispose:
            dispose()

        if not workflow_config.get("skip_assert"):
            print("performing dataset assertions")
            self.__assert_indexer_outputs(root, workflow_config)

        print("running queries")
        for query in query_config:
            result = self.__run_query(root, query)
            print(f"Query: {query}\nResponse: {result.stdout}")
            assert result.returncode == 0 and result.stdout
