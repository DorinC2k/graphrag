# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Blob source module."""

import io
import logging
import os
from io import BytesIO

import pandas as pd
import streamlit as st
import yaml
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from knowledge_loader.data_sources.typing import Datasource

from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

from .default import blob_account_name, blob_container_name, blob_connection_string

logging.basicConfig(level=logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@st.cache_data(ttl=60 * 60 * 24)
def _get_container(
    account_name: str | None,
    container_name: str,
    connection_string: str | None = None,
) -> ContainerClient:
    """Return container from blob storage."""
    print("LOGIN---------------")  # noqa T201
    if connection_string:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    else:
        account_url = f"https://{account_name}.blob.core.windows.net"
        default_credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url, credential=default_credential)
    return blob_service_client.get_container_client(container_name)


def load_blob_prompt_config(
    dataset: str,
    account_name: str | None = blob_account_name,
    container_name: str | None = blob_container_name,
    connection_string: str | None = blob_connection_string,
) -> dict[str, str]:
    """Load blob prompt configuration."""
    if (account_name is None and connection_string is None) or container_name is None:
        return {}

    container_client = _get_container(account_name, container_name, connection_string)
    prompts = {}

    prefix = f"{dataset}/prompts"
    for file in container_client.list_blobs(name_starts_with=prefix):
        map_name = file.name.split("/")[-1].split(".")[0]
        prompts[map_name] = (
            container_client.download_blob(file.name).readall().decode("utf-8")
        )

    return prompts


def load_blob_file(
    dataset: str | None,
    file: str | None,
    account_name: str | None = blob_account_name,
    container_name: str | None = blob_container_name,
    connection_string: str | None = blob_connection_string,
) -> BytesIO:
    """Load blob file from container."""
    stream = io.BytesIO()

    if (account_name is None and connection_string is None) or container_name is None:
        logger.warning("No account name or container name provided")
        return stream

    container_client = _get_container(account_name, container_name, connection_string)

    # Ensure we have a valid file path to download
    if file is None and dataset is None:
        logger.warning("No file or dataset provided")
        return stream

    blob_path = f"{dataset}/{file}" if dataset is not None else file

    # blob_path is guaranteed to be a str here
    container_client.download_blob(blob_path).readinto(stream) # type: ignore

    return stream


def _log_missing_blob_details(dataset: str | None, filename: str) -> None:
    """Log where a missing blob was expected and list siblings."""
    expected_path = f"{dataset}/{filename}" if dataset else filename
    logger.warning("Expected settings file at '%s' but it was not found", expected_path)

    prefix = f"{dataset}/" if dataset else ""
    display_prefix = prefix or "<root>"

    if (
        blob_container_name is None
        or (blob_account_name is None and blob_connection_string is None)
    ):
        logger.warning(
            "Cannot list blobs under '%s' because blob storage configuration is incomplete",
            display_prefix,
        )
        return

    try:
        container_client = _get_container(
            blob_account_name, blob_container_name, blob_connection_string
        )
        files = [
            blob.name[len(prefix) :] if blob.name.startswith(prefix) else blob.name
            for blob in container_client.list_blobs(name_starts_with=prefix)
        ]
        if not files:
            logger.warning("No files found under '%s'", display_prefix)
        else:
            logger.warning(
                "Files available under '%s': %s",
                display_prefix,
                ", ".join(sorted(files)),
            )
    except Exception as err:  # noqa: BLE001
        logger.warning(
            "Unable to list blobs under '%s' due to error: %s", display_prefix, err
        )


DEFAULT_CONFIG_FILENAMES = ("settings.yaml", 
                            "settings.yml", 
                            "settings.json",
                            "/mnt/c/development/sandbox-projects/rg-md-law-setup/gr-law/md-law/gr-output-run-4/settings.yaml")

class BlobDatasource(Datasource):
    """Datasource that reads from a blob storage parquet file."""

    def __init__(self, database: str):
        """Init method definition."""
        self._database = database

    def read(
        self,
        table: str,
        throw_on_missing: bool = False,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read file from container."""
        try:
            data = load_blob_file(self._database, f"{table}.parquet")
        except Exception as err:
            if throw_on_missing:
                error_msg = f"Table {table} does not exist"
                raise FileNotFoundError(error_msg) from err
            logger.warning("Table %s does not exist", table)
            return pd.DataFrame(columns=columns) if columns else pd.DataFrame()

        return pd.read_parquet(data, columns=columns)

    # temp function to read settings from local storage
    def read_settings(
        self,
        file: str | None = None,
        throw_on_missing: bool = False,
    ) -> GraphRagConfig | None:
        """Read settings from blob; if not found, try absolute local path."""
        filenames = [file] if file else list(DEFAULT_CONFIG_FILENAMES)
        last_error: Exception | None = None

        # 1) Try to read from Blob, in order
        for filename in filenames:
            try:
                settings = load_blob_file(self._database, filename)
                settings.seek(0)
                str_settings = settings.read().decode("utf-8")
                config = os.path.expandvars(str_settings)
                settings_yaml = yaml.safe_load(config)
                return create_graphrag_config(values=settings_yaml)
            except ResourceNotFoundError as err:
                _log_missing_blob_details(self._database, filename)
                last_error = err
                continue
            except Exception as err:  # noqa: BLE001
                # Keep trying other candidate filenames
                last_error = err
                continue

        # 2) Fallback: absolute local path only (as requested)
        #    - If the caller provided an absolute path in `file`, try it.
        #    - If `file` was not absolute or None, we only consider entries in
        #      `filenames` that are absolute (per requirement: "via an absolute path").
        for candidate in filenames:
            if not candidate or not os.path.isabs(candidate):
                continue  # only accept absolute paths for local fallback
            try:
                logger.info("Blob not found. Trying local absolute path: %s", candidate)
                with open(candidate, "r", encoding="utf-8") as fh:
                    str_settings = fh.read()
                config = os.path.expandvars(str_settings)
                settings_yaml = yaml.safe_load(config)
                return create_graphrag_config(values=settings_yaml)
            except FileNotFoundError as err:
                logger.warning("Local file not found at absolute path: %s", candidate)
                last_error = err
                continue
            except Exception as err:  # noqa: BLE001
                logger.warning("Failed to load local settings from %s: %s", candidate, err)
                last_error = err
                continue

        # 3) Nothing found -> handle error/reporting
        if throw_on_missing:
            missing = file if file else ", ".join(DEFAULT_CONFIG_FILENAMES)
            error_msg = (
                "Settings not found in blob and no valid absolute local path resolved "
                f"for: {missing}"
            )
            if last_error is not None:
                raise FileNotFoundError(error_msg) from last_error
            raise FileNotFoundError(error_msg)

        missing = file if file else ", ".join(DEFAULT_CONFIG_FILENAMES)
        logger.warning(
            "Settings not found in blob and no valid absolute local path provided for: %s",
            missing,
        )
        return None
