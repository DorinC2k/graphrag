# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests covering the extract graph configuration defaults."""

from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.index.operations.extract_graph.typing import (
    ExtractEntityStrategyType,
)

from .utils import DEFAULT_MODEL_CONFIG


def test_resolved_strategy_defaults_to_mrebel() -> None:
    config = create_graphrag_config({"models": DEFAULT_MODEL_CONFIG})
    model_config = config.get_language_model_config(config.extract_graph.model_id)

    strategy = config.extract_graph.resolved_strategy(config.root_dir, model_config)

    assert strategy["type"] == ExtractEntityStrategyType.huggingface_mrebel
    assert "llm" not in strategy
