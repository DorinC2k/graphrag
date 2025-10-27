# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Text embedding strategy using HuggingFace sentence-transformers.

Supports the ``HUGGINGFACE_API_TOKEN`` environment variable for authentication,
falling back to ``HUGGINGFACEHUB_API_TOKEN`` and ``HUGGING_FACE_TOKEN_READ_KEY``
for backward compatibility.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Iterable

import numpy as np
import requests

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

from graphrag.config.defaults import ENCODING_MODEL
from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.embed_text.strategies.typing import TextEmbeddingResult
from graphrag.index.text_splitting.text_splitting import TokenTextSplitter
from graphrag.logger.progress import progress_ticker

DEFAULT_REMOTE_MAX_BATCH_SIZE = 32
DEFAULT_REMOTE_MAX_INPUT_TOKENS = 256


def _as_bearer_token(token: str) -> str:
    token = token.strip()
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def _strip_bearer_token(token: str | None) -> str | None:
    if token is None:
        return None
    value = token.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value


async def run(
    input: list[str],
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    args: dict[str, Any],
) -> TextEmbeddingResult:
    """Embed text using HuggingFace locally or via a remote endpoint."""
    if not input:
        return TextEmbeddingResult(embeddings=None)

    model_info = args.get("llm", {})
    model_name = model_info.get("model")
    api_base = model_info.get("api_base")
    # API key is sourced from model info or supported environment variables
    api_key = _strip_bearer_token(
        model_info.get("api_key")
        or os.getenv("HUGGINGFACE_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_TOKEN_READ_KEY")
    )

    if api_base:
        if not api_key:
            msg = "HuggingFace API key not provided"
            raise ValueError(msg)

        configured_batch_size = int(args.get("batch_size") or DEFAULT_REMOTE_MAX_BATCH_SIZE)
        max_remote_batch_size = int(
            args.get("max_remote_batch_size") or DEFAULT_REMOTE_MAX_BATCH_SIZE
        )
        configured_batch_size = max(configured_batch_size, 1)
        max_remote_batch_size = max(max_remote_batch_size, 1)
        effective_batch_size = min(configured_batch_size, max_remote_batch_size)

        chunk_overlap = int(args.get("chunk_overlap") or 0)
        remote_max_input_tokens = int(
            args.get("remote_max_input_tokens") or DEFAULT_REMOTE_MAX_INPUT_TOKENS
        )
        remote_max_input_tokens = max(remote_max_input_tokens, 1)
        batch_max_tokens = int(args.get("batch_max_tokens") or remote_max_input_tokens)
        batch_max_tokens = max(batch_max_tokens, 1)
        token_limit = min(batch_max_tokens, remote_max_input_tokens)

        splitter = TokenTextSplitter(
            encoding_name=model_info.get("encoding_model") or ENCODING_MODEL,
            chunk_size=token_limit,
            chunk_overlap=chunk_overlap,
        )

        texts, input_sizes = _prepare_embed_texts(input, splitter)
        if not texts:
            embeddings = _reconstitute_embeddings([], input_sizes)
            return TextEmbeddingResult(embeddings=embeddings)

        def _chunks(values: Iterable[str], size: int) -> Iterable[list[str]]:
            batch: list[str] = []
            for value in values:
                batch.append(value)
                if len(batch) == size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        ticker = progress_ticker(callbacks.progress, len(texts))
        all_embeddings: list[Any] = []

        for batch in _chunks(texts, effective_batch_size):
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    api_base,
                    headers={
                        "Authorization": _as_bearer_token(api_key),
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json={"inputs": batch},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:  # pragma: no cover - network failures
                status = getattr(e.response, "status_code", "unknown")
                text = getattr(e.response, "text", "")
                msg = f"HuggingFace embedding request failed: {status} {text}"
                callbacks.error(msg, e)
                raise RuntimeError(msg) from e

            if isinstance(data, dict):
                embeddings = data.get("embeddings")
            else:
                embeddings = data

            if embeddings is None:
                msg = "HuggingFace embedding response did not include embeddings"
                raise RuntimeError(msg)

            all_embeddings.extend(embeddings)
            ticker(len(batch))

        resolved_embeddings = _reconstitute_embeddings(all_embeddings, input_sizes)

        return TextEmbeddingResult(embeddings=resolved_embeddings)

    if model_name is None:
        msg = "HuggingFace model name not provided"
        raise ValueError(msg)

    if SentenceTransformer is None:  # pragma: no cover - optional dependency
        msg = "sentence-transformers is required to use local HuggingFace embeddings"
        raise ImportError(msg)

    model = SentenceTransformer(model_name)

    embeddings = await asyncio.to_thread(
        model.encode,
        input,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    ticker = progress_ticker(callbacks.progress, len(input))
    ticker(len(input))

    if isinstance(embeddings, np.ndarray):
        emb_list = embeddings.tolist()
    else:
        emb_list = [
            e.tolist() if isinstance(e, np.ndarray) else list(e) for e in embeddings
        ]

    return TextEmbeddingResult(embeddings=emb_list)


def _prepare_embed_texts(
    input: list[str], splitter: TokenTextSplitter
) -> tuple[list[str], list[int]]:
    sizes: list[int] = []
    snippets: list[str] = []

    for text in input:
        split_texts = splitter.split_text(text)
        if split_texts is None:
            sizes.append(0)
            continue
        split_texts = [item for item in split_texts if len(item) > 0]
        sizes.append(len(split_texts))
        snippets.extend(split_texts)

    return snippets, sizes


def _reconstitute_embeddings(
    raw_embeddings: list[list[float]], sizes: list[int]
) -> list[list[float] | None]:
    embeddings: list[list[float] | None] = []
    cursor = 0
    for size in sizes:
        if size == 0:
            embeddings.append(None)
            continue

        chunk = raw_embeddings[cursor : cursor + size]
        cursor += size

        if size == 1:
            first = chunk[0]
            embeddings.append(first.tolist() if isinstance(first, np.ndarray) else first)
            continue

        average = np.average(chunk, axis=0)
        norm = np.linalg.norm(average)
        normalized = average if norm == 0 else average / norm
        embeddings.append(
            normalized.tolist() if isinstance(normalized, np.ndarray) else normalized
        )

    return embeddings
