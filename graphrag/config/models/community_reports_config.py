# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pathlib import Path

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.models.language_model_config import LanguageModelConfig


class CommunityReportsConfig(BaseModel):
    """Configuration section for community reports."""

    model_id: str = Field(
        description="The model ID to use for community reports.",
        default=graphrag_config_defaults.community_reports.model_id,
    )
    graph_prompt: str | None = Field(
        description="The community report extraction prompt to use for graph-based summarization.",
        default=graphrag_config_defaults.community_reports.graph_prompt,
    )
    text_prompt: str | None = Field(
        description="The community report extraction prompt to use for text-based summarization.",
        default=graphrag_config_defaults.community_reports.text_prompt,
    )
    max_length: int = Field(
        description="The community report maximum length in tokens.",
        default=graphrag_config_defaults.community_reports.max_length,
    )
    max_input_length: int = Field(
        description="The maximum input length in tokens to use when generating reports.",
        default=graphrag_config_defaults.community_reports.max_input_length,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.",
        default=graphrag_config_defaults.community_reports.strategy,
    )

    def resolved_strategy(
        self, root_dir: str, model_config: LanguageModelConfig
    ) -> dict:
        """Get the resolved community report extraction strategy."""
        from graphrag.index.operations.summarize_communities.typing import (
            CreateCommunityReportsStrategyType,
        )

        strategy = dict(self.strategy) if self.strategy else {}

        default_type = CreateCommunityReportsStrategyType.graph_intelligence
        strategy_type_raw = strategy.get("type", default_type)

        try:
            strategy_type = (
                CreateCommunityReportsStrategyType(strategy_type_raw)
                if isinstance(strategy_type_raw, str)
                else strategy_type_raw
            )
        except ValueError:
            strategy_type = strategy_type_raw

        if isinstance(strategy_type, CreateCommunityReportsStrategyType):
            strategy["type"] = strategy_type

        if self.graph_prompt and "graph_prompt" not in strategy:
            strategy["graph_prompt"] = (Path(root_dir) / self.graph_prompt).read_text(
                encoding="utf-8"
            )

        if self.text_prompt and "text_prompt" not in strategy:
            strategy["text_prompt"] = (Path(root_dir) / self.text_prompt).read_text(
                encoding="utf-8"
            )

        if isinstance(strategy_type, CreateCommunityReportsStrategyType):
            if strategy_type == default_type:
                strategy.setdefault("llm", model_config.model_dump())
        else:
            if strategy_type_raw == default_type.value:
                strategy.setdefault("llm", model_config.model_dump())

        strategy.setdefault("max_report_length", self.max_length)
        strategy.setdefault("max_input_length", self.max_input_length)

        return strategy
