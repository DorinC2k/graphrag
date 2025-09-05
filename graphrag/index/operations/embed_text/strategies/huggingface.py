# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Text embedding strategy using HuggingFace sentence-transformers."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np
import requests

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.embed_text.strategies.typing import TextEmbeddingResult
from graphrag.logger.progress import progress_ticker


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
    api_key = model_info.get("api_key") or os.getenv("HUGGING_FACE_TOKEN_READ_KEY")

    if api_base:
        if not api_key:
            msg = "HuggingFace API key not provided"
            raise ValueError(msg)
        try:
            response = await asyncio.to_thread(
                requests.post,
                api_base,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": input},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:  # pragma: no cover - network failures
            callbacks.error("HuggingFace embedding request failed", e)
            msg = "HuggingFace embedding request failed"
            raise RuntimeError(msg) from e

        ticker = progress_ticker(callbacks.progress, len(input))
        ticker(len(input))

        if isinstance(data, dict):
            embeddings = data.get("embeddings")
        else:
            embeddings = data
        return TextEmbeddingResult(embeddings=embeddings)

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
