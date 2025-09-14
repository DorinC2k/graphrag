# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Prepare documents for ChatGPT-based graph extraction.

This script reads plain-text documents from a local directory or an Azure Blob
container, splits them into token-limited chunks and groups them into
"binders" that comply with ChatGPT's attachment limits (20 files and roughly
2 million tokens per binder).  Each chunk is written as an individual text file
so it can be uploaded to the ChatGPT UI for manual prompt execution.

A `text_units.parquet` file containing the chunk metadata is produced so the
results from ChatGPT can later be merged back into GraphRAG.

Example usage:

    # Read local files
    python scripts/chatgpt_prepare_docs.py \
        --source local --input-dir ./corpus --out-dir ./binders

    # Read files from Azure Blob Storage
    python scripts/chatgpt_prepare_docs.py \
        --source azure --container mycontainer --prefix docs/ --out-dir ./binders

The Azure variant expects a storage connection string in the environment
variable ``AZURE_STORAGE_CONNECTION_STRING``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import tiktoken
import typer
from azure.storage.blob import ContainerClient

if TYPE_CHECKING:
    from collections.abc import Iterator

# Constants based on ChatGPT UI limits
MAX_FILES_PER_BINDER = 20
MAX_TOKENS_PER_BINDER = 2_000_000


def _iter_local_files(input_dir: Path) -> Iterator[tuple[str, str]]:
    for path in sorted(input_dir.rglob("*.txt")):
        yield path.name, path.read_text(encoding="utf-8")


def _iter_azure_files(
    container: ContainerClient, prefix: str
) -> Iterator[tuple[str, str]]:
    for blob in container.list_blobs(name_starts_with=prefix):
        if not blob.name.endswith(".txt"):
            continue
        data = container.download_blob(blob).readall().decode("utf-8")
        yield Path(blob.name).name, data


@dataclass
class Chunk:
    """Metadata for a single text chunk."""

    id: int
    document_id: str
    text: str


def chunk_text(
    text: str, chunk_size: int, overlap: int, enc: tiktoken.Encoding
) -> list[str]:
    """Split ``text`` into token-limited chunks using ``enc``."""
    tokens = enc.encode(text)
    if not tokens:
        return []
    step = max(chunk_size - overlap, 1)
    chunks: list[str] = []
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(enc.decode(chunk_tokens))
    return chunks


def prepare_bindings(
    source: str,
    input_dir: Path | None,
    container_name: str | None,
    prefix: str,
    out_dir: Path,
    chunk_size: int,
    overlap: int,
    model: str,
) -> None:
    """Create binder folders and write chunk files."""
    enc = tiktoken.encoding_for_model(model)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "local":
        if input_dir is None:
            msg = "input_dir is required when source='local'"
            raise ValueError(msg)
        file_iter = _iter_local_files(input_dir)
    else:
        if container_name is None:
            msg = "container is required when source='azure'"
            raise ValueError(msg)
        connection = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connection:
            msg = "AZURE_STORAGE_CONNECTION_STRING is not set"
            raise RuntimeError(msg)
        container = ContainerClient.from_connection_string(connection, container_name)
        file_iter = _iter_azure_files(container, prefix)

    binder_idx = 0
    file_count = 0
    token_count = 0
    chunks_meta: list[Chunk] = []

    for doc_id, text in file_iter:
        for chunk in chunk_text(text, chunk_size, overlap, enc):
            if (
                file_count >= MAX_FILES_PER_BINDER
                or token_count + len(enc.encode(chunk)) > MAX_TOKENS_PER_BINDER
            ):
                binder_idx += 1
                file_count = 0
                token_count = 0
            binder_path = out_dir / f"binder_{binder_idx:04d}"
            binder_path.mkdir(parents=True, exist_ok=True)

            chunk_id = len(chunks_meta) + 1
            file_path = binder_path / f"chunk_{chunk_id:06d}.txt"
            file_path.write_text(chunk, encoding="utf-8")

            chunks_meta.append(Chunk(chunk_id, doc_id, chunk))
            file_count += 1
            token_count += len(enc.encode(chunk))

    # Build text_units parquet
    df = pd.DataFrame([c.__dict__ for c in chunks_meta])
    df.to_parquet(out_dir / "text_units.parquet", index=False)


app = typer.Typer(help="Prepare documents for manual ChatGPT graph extraction")


@app.command()
def main(
    source: str = typer.Option("local", help="'local' or 'azure'"),
    input_dir: Path | None = typer.Option(None, help="Directory with .txt files"),
    container: str | None = typer.Option(None, help="Azure Blob container name"),
    prefix: str = typer.Option("", help="Blob name prefix"),
    out_dir: Path = typer.Option(..., help="Output directory for binders"),
    chunk_size: int = typer.Option(8000, help="Chunk size in tokens"),
    overlap: int = typer.Option(200, help="Token overlap between chunks"),
    model: str = typer.Option("gpt-4o-mini", help="Model name for tokenization"),
) -> None:
    """Create chunk files and a text_units table from source documents."""
    prepare_bindings(
        source, input_dir, container, prefix, out_dir, chunk_size, overlap, model
    )


if __name__ == "__main__":
    app()
