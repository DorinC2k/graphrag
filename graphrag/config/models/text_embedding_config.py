# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType


class TextEmbeddingConfig(BaseModel):
    """Configuration section for text embeddings."""

    model_id: str = Field(
        description="The model ID to use for text embeddings.",
        default=graphrag_config_defaults.embed_text.model_id,
    )
    vector_store_id: str = Field(
        description="The vector store ID to use for text embeddings.",
        default=graphrag_config_defaults.embed_text.vector_store_id,
    )
    batch_size: int = Field(
        description="The batch size to use.",
        default=graphrag_config_defaults.embed_text.batch_size,
    )
    batch_max_tokens: int = Field(
        description="The batch max tokens to use.",
        default=graphrag_config_defaults.embed_text.batch_max_tokens,
    )
    names: list[str] = Field(
        description="The specific embeddings to perform.",
        default=graphrag_config_defaults.embed_text.names,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.",
        default=graphrag_config_defaults.embed_text.strategy,
    )

    def resolved_strategy(self, model_config: LanguageModelConfig) -> dict:
        """Get the resolved text embedding strategy."""
        from graphrag.index.operations.embed_text import (
            TextEmbedStrategyType,
        )

        strategy = dict(self.strategy) if self.strategy else {}

        default_type = (
            TextEmbedStrategyType.huggingface
            if model_config.type == ModelType.HuggingFaceEmbedding
            else TextEmbedStrategyType.openai
        )

        strategy_type_raw = strategy.get("type", default_type)

        try:
            strategy_type = (
                TextEmbedStrategyType(strategy_type_raw)
                if isinstance(strategy_type_raw, str)
                else strategy_type_raw
            )
        except ValueError:
            strategy_type = strategy_type_raw

        if isinstance(strategy_type, TextEmbedStrategyType):
            strategy["type"] = strategy_type

        strategy.setdefault("num_threads", model_config.concurrent_requests)
        strategy.setdefault("batch_size", self.batch_size)
        strategy.setdefault("batch_max_tokens", self.batch_max_tokens)

        if isinstance(strategy_type, TextEmbedStrategyType):
            if strategy_type in (
                TextEmbedStrategyType.openai,
                TextEmbedStrategyType.huggingface,
            ):
                strategy.setdefault("llm", model_config.model_dump())
        else:
            if strategy_type_raw in (
                TextEmbedStrategyType.openai.value,
                TextEmbedStrategyType.huggingface.value,
            ):
                strategy.setdefault("llm", model_config.model_dump())

        return strategy
