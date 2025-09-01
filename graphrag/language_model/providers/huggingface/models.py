from __future__ import annotations

"""Hugging Face embedding model provider."""

from typing import TYPE_CHECKING, Any
import asyncio

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - dependency is optional
    SentenceTransformer = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import LanguageModelConfig


class HuggingFaceEmbeddingModel:
    """Embedding model backed by a Hugging Face `SentenceTransformer`."""

    model: SentenceTransformer

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        if SentenceTransformer is None:  # pragma: no cover - dependency is optional
            raise ImportError(
                "sentence-transformers is required to use HuggingFace embeddings"
            )
        model_name = config.model or "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name, use_auth_token=config.api_key)
        self.config = config

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(text_list, convert_to_numpy=True, **kwargs),
        )
        return embeddings.tolist()

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        return (await self.aembed_batch([text], **kwargs))[0]

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        embeddings = self.model.encode(text_list, convert_to_numpy=True, **kwargs)
        return embeddings.tolist()

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_batch([text], **kwargs)[0]
