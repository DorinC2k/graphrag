# test_fixtures.py (cross-platform ready)

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

import pytest

pytest.importorskip(
    "fnllm",
    reason=(
        "The fnllm package is required for smoke fixture tests. "
        "Install with `pip install fnllm[azure,openai]` to enable these tests."
    ),
)
pytest.importorskip(
    "pandas",
    reason=(
        "The pandas package is required for smoke fixture validations. "
        "Install with `pip install pandas pyarrow` to enable these tests."
    ),
)

import pandas as pd

from graphrag.query.context_builder.community_context import NO_COMMUNITY_RECORDS_WARNING
from graphrag.storage.blob_pipeline_storage import BlobPipelineStorage

log = logging.getLogger(__name__)

# Debug and platform flags
debug = True
gh_pages = os.environ.get("GH_PAGES") is not None

# Azurite connection string with override option
WELL_KNOWN_AZURITE_CONNECTION_STRING = os.getenv(
    "WELL_KNOWN_AZURITE_CONNECTION_STRING",
    (
        "DefaultEndpointsProtocol=http;"
        "AccountName=devstoreaccount1;"
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
        "K1SZFPTOtr/KBHBeksoGMGw==;"
        "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1"
    )
)

KNOWN_WARNINGS = [NO_COMMUNITY_RECORDS_WARNING]

safe_home = os.path.expanduser("~")
is_windows = platform.system() == "Windows"

# Base env_vars with all needed template substitutions
env_vars = {
    "BLOB_STORAGE_CONNECTION_STRING": os.getenv("GRAPHRAG_CACHE_CONNECTION_STRING", WELL_KNOWN_AZURITE_CONNECTION_STRING),
    "LOCAL_BLOB_STORAGE_CONNECTION_STRING": WELL_KNOWN_AZURITE_CONNECTION_STRING,
    "GRAPHRAG_CHUNK_SIZE": "1200",
    "GRAPHRAG_CHUNK_OVERLAP": "0",
    "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
    "GRAPHRAG_API_KEY": os.getenv("GRAPHRAG_API_KEY", "A8rMHLsf0lbEQfePHJ6RXgdx1OfTjev5DHZTmKT4s3l7yfAFFuIqJQQJ99BIACHYHv6XJ3w3AAAAACOGnVXk"),
    "GRAPHRAG_API_BASE": "https://ai-dorin2720ai095958338123.cognitiveservices.azure.com/",
    "GRAPHRAG_API_VERSION": "2025-08-07",
    "GRAPHRAG_LLM_DEPLOYMENT_NAME": "gpt-5-nano",
    "GRAPHRAG_LLM_MODEL": "gpt-5-nano",
    "GRAPHRAG_LLM_TPM": "1000",
    "GRAPHRAG_LLM_RPM": "1000",
    "GRAPHRAG_EMBEDDING_API_VERSION": "2",
    "GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME": "sentence-transformers/all-MiniLM-L6-v2",
    "GRAPHRAG_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "HUGGINGFACE_API_TOKEN": os.getenv("HUGGINGFACE_API_TOKEN", "a"),
    "HUGGINGFACE_API_BASE": os.getenv("HUGGINGFACE_API_BASE", "https://cgglxt1hbsrcvegq.us-east-1.aws.endpoints.huggingface.cloud"),
    "HOME": safe_home,
    "NLTK_DATA": os.path.join(safe_home, "nltk_data"),
    "KMP_DUPLICATE_LIB_OK": "True",
    "KMP_INIT_AT_FORK": "FALSE",
    "DEBUG": "True"
}

# Windows-specific overrides
if is_windows:
    env_vars["USERPROFILE"] = os.environ.get("USERPROFILE", safe_home)
    env_vars["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "C:\\Windows")

# Clean out any accidental None values
env_vars = {k: v for k, v in env_vars.items() if v is not None}

def build_subprocess_env(extra_vars: dict | None = None) -> dict:
    env = dict(env_vars)
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
        full_path = fixtures_path / subfolder
        if not full_path.is_dir():
            continue
        config_file = full_path / "config.json"
        params.append((subfolder, json.loads(config_file.read_text(encoding="utf-8"))))
    return params

def pytest_generate_tests(metafunc):
    run_slow = metafunc.config.getoption("run_slow")
    configs = metafunc.cls.params[metafunc.function.__name__]
    if not run_slow:
        configs = [config for config in configs if not config[1].get("slow", False)]
    if not configs:
        pytest.skip("No configurations to run")
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
                if not skip and "input_path" in kwargs:
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
        text = data_file.read_text(encoding="utf-8")
        file_path = str(Path(input_base_dir) / data_file.name) if input_base_dir else data_file.name
        await input_storage.set(file_path, text, encoding="utf-8")
    return lambda: input_storage._delete_container()

class TestIndexer:
    params: ClassVar[dict[str, list[tuple[str, dict[str, Any]]]]] = {
        "test_fixture": _load_fixtures()
    }

    def __run_indexer(self, root: Path, input_file_type: str, embedding_type: str):
        command = [
            sys.executable, "-m", "graphrag", "index",
            "--verbose" if debug else None,
            "--root", root.resolve().as_posix(),
            "--logger", "print",
            "--method", "standard",
        ]
        command = [arg for arg in command if arg]
        log.info("Running command: %s", " ".join(command))
        completion = subprocess.run(command, env=build_subprocess_env({
            "GRAPHRAG_INPUT_FILE_TYPE": input_file_type,
            "GRAPHRAG_EMBEDDING_TYPE": embedding_type,
        }))
        assert completion.returncode == 0, f"Indexer failed with return code: {completion.returncode}"

    def __assert_indexer_outputs(self, root: Path, workflow_config: dict[str, dict[str, Any]]):
        output_path = root / "output"
        assert output_path.exists(), "output folder does not exist"
        stats = json.loads((output_path / "stats.json").read_text(encoding="utf-8"))
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

    def __run_query(self, root: Path, query_config: dict[str, str], extra_vars: dict | None = None):
        command = [
            sys.executable, "-X", "utf8",  # ← force UTF-8 mode
            "-m", "graphrag", "query",
            "--root", root.resolve().as_posix(),
            "--method", query_config["method"],
            "--community-level", str(query_config.get("community_level", 2)),
            "--query", query_config["query"],
        ]
        result = subprocess.run(
            command,
            env=build_subprocess_env(extra_vars | {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"} if extra_vars else {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}),
            capture_output=True,
            text=True,
            encoding="utf-8",   # ← Python 3.11 supports this
            errors="replace",   # ← avoid hard failures on odd bytes
        )
        print("Command:", " ".join(command))
        if result.stdout:
            print("STDOUT:\n", result.stdout)
        if result.stderr:
            print("STDERR:\n", result.stderr)
        assert result.returncode == 0 and result.stdout, "Query failed; see STDERR above"
        return result



    def __run_queries(self, root: Path, query_config: list[dict[str, str]], embedding_type: str):
        for query in query_config:
            result = self.__run_query(root, query, {"GRAPHRAG_EMBEDDING_TYPE": embedding_type})
            print(f"Query: {query}\nResponse: {result.stdout}")
            assert result.returncode == 0 and result.stdout

    @cleanup(skip=debug)
    @mock.patch.dict(os.environ, env_vars, clear=True)
    @pytest.mark.timeout(2000)
    def test_fixture(self, request, input_path: str, input_file_type: str,
                     workflow_config: dict[str, dict[str, Any]],
                     query_config: list[dict[str, str]]):
        if workflow_config.get("skip"):
            pytest.skip(f"Skipping smoke test {input_path}")
        root = Path(input_path)
        embedding_type = "huggingface_embedding"
        queries_only = request.config.getoption("run_queries_only")
        if queries_only:
            output_path = root / "output"
            if not output_path.exists():
                pytest.skip(
                    "Query-only run requested, but no existing index output found. "
                    "Run the index step first or omit --run_queries_only."
                )
            self.__run_queries(root, query_config, embedding_type)
            return
        dispose = asyncio.run(prepare_azurite_data(input_path, workflow_config["azure"])) \
            if workflow_config.get("azure") else None
        self.__run_indexer(root, input_file_type, embedding_type)
        # if dispose:
            # dispose()
        if not workflow_config.get("skip_assert"):
            self.__assert_indexer_outputs(root, workflow_config)
        self.__run_queries(root, query_config, embedding_type)
