from __future__ import annotations

"""Hugging Face embedding model provider.

Supports the ``HUGGINGFACE_API_TOKEN`` environment variable for authentication,
falling back to ``HUGGINGFACEHUB_API_TOKEN`` and ``HUGGING_FACE_TOKEN_READ_KEY``
for backward compatibility.
"""

from typing import TYPE_CHECKING, Any
import asyncio
import os
import requests

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
        if self.api_base:
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
                    json={"inputs": text_list},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except (
                requests.RequestException
            ) as e:  # pragma: no cover - network failures
                msg = "HuggingFace embedding request failed"
                raise RuntimeError(msg) from e

            if isinstance(data, dict):
                return data.get("embeddings")
            return data

        embeddings = self.model.encode(text_list, convert_to_numpy=True, **kwargs)
        return embeddings.tolist()

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_batch([text], **kwargs)[0]
