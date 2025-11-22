from __future__ import annotations

"""Hugging Face embedding model provider.

Supports the ``HUGGINGFACE_API_TOKEN`` environment variable for authentication,
falling back to ``HUGGINGFACEHUB_API_TOKEN`` and ``HUGGING_FACE_TOKEN_READ_KEY``
for backward compatibility.
"""

from typing import TYPE_CHECKING, Any
import asyncio
import importlib
import os

import numpy as np

try:  # pragma: no cover - optional dependency handling
    import requests
except ModuleNotFoundError:
    requests = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - dependency is optional
    SentenceTransformer = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import LanguageModelConfig


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


class HuggingFaceEmbeddingModel:
    """Embedding model backed by a Hugging Face `SentenceTransformer` or remote endpoint."""

    model: SentenceTransformer | None

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        self.config = config
        self.api_base = config.api_base
        # API key is sourced from config or supported environment variables
        self.api_key = _strip_bearer_token(
            config.api_key
            or os.getenv("HUGGINGFACE_API_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGING_FACE_TOKEN_READ_KEY")
        )
        self.encoding_model = config.encoding_model
        self.max_tokens_per_request = _get_max_tokens_per_request()

        if self.api_base:
            # Remote endpoint, no local model initialization
            self.model = None
            return

        if SentenceTransformer is None:  # pragma: no cover - dependency is optional
            raise ImportError(
                "sentence-transformers is required to use HuggingFace embeddings",
            )

        model_name = config.model or "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name, use_auth_token=self.api_key)

    async def aembed_batch(
        self, text_list: list[str], **kwargs: Any
    ) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_batch, text_list, **kwargs)

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        return (await self.aembed_batch([text], **kwargs))[0]

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        prepared_texts, chunk_sizes = _prepare_embed_texts(
            text_list,
            self.encoding_model,
            self.max_tokens_per_request,
        )

        if self.api_base:
            if requests is None:  # pragma: no cover - optional dependency handling
                raise ImportError(
                    "requests is required to call remote HuggingFace embeddings",
                )
            try:
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
                if self.api_key:
                    headers["Authorization"] = _as_bearer_token(self.api_key)
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json={"inputs": prepared_texts},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except requests.HTTPError as e:  # pragma: no cover - network failures
                status = e.response.status_code if e.response else "unknown"
                error_detail = ""
                if e.response is not None:
                    try:
                        error_detail = e.response.text
                    except Exception:  # pragma: no cover - best effort diagnostic
                        error_detail = ""
                msg = (
                    "HuggingFace embedding request failed "
                    f"(status {status}). {error_detail}".strip()
                )
                raise RuntimeError(msg) from e
            except requests.RequestException as e:  # pragma: no cover - network failures
                msg = f"HuggingFace embedding request failed: {e}"
                raise RuntimeError(msg) from e

            if isinstance(data, dict):
                embeddings = data.get("embeddings")
            else:
                embeddings = data

            return _reconstitute_embeddings(embeddings, chunk_sizes)

        embeddings = self.model.encode(
            prepared_texts, convert_to_numpy=True, **kwargs
        )
        return _reconstitute_embeddings(embeddings, chunk_sizes)

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_batch([text], **kwargs)[0]


def _get_max_tokens_per_request() -> int | None:
    value = os.getenv("GRAPHRAG_EMBEDDING_TPR")
    if value is None:
        return None

    try:
        limit = int(value)
    except ValueError:
        return None

    return limit if limit > 0 else None


def _prepare_embed_texts(
    text_list: list[str], encoding_model: str, max_tokens: int | None
) -> tuple[list[str], list[int]]:
    if max_tokens is None:
        return text_list, [1 for _ in text_list]

    if importlib.util.find_spec("tiktoken") is None:
        msg = "tiktoken is required when GRAPHRAG_EMBEDDING_TPR is set"
        raise ImportError(msg)

    from graphrag.index.text_splitting.text_splitting import TokenTextSplitter

    splitter = TokenTextSplitter(
        encoding_name=encoding_model,
        chunk_size=max_tokens,
        chunk_overlap=0,
    )

    snippets: list[str] = []
    sizes: list[int] = []

    for text in text_list:
        split_texts = splitter.split_text(text)
        if split_texts is None:
            sizes.append(0)
            continue

        split_texts = [item for item in split_texts if len(item) > 0]
        sizes.append(len(split_texts))
        snippets.extend(split_texts)

    return snippets, sizes


def _reconstitute_embeddings(
    raw_embeddings: list[list[float]] | np.ndarray, sizes: list[int]
) -> list[list[float] | None]:
    if not sizes:
        return []

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
            if isinstance(first, np.ndarray):
                embeddings.append(first.tolist())
            else:
                embeddings.append(first)
            continue

        array = np.array(chunk, dtype=float)
        average = array.mean(axis=0)
        norm = np.linalg.norm(average)
        normalized = average if norm == 0 else average / norm
        embeddings.append(normalized.tolist())

    return embeddings
