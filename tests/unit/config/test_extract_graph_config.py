# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests covering the extract graph configuration defaults."""

from pathlib import Path

from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.index.operations.extract_graph.typing import (
    ExtractEntityStrategyType,
)

from .utils import DEFAULT_MODEL_CONFIG


def test_resolved_strategy_defaults_to_graph_intelligence() -> None:
    config = create_graphrag_config({"models": DEFAULT_MODEL_CONFIG})
    model_config = config.get_language_model_config(config.extract_graph.model_id)

    strategy = config.extract_graph.resolved_strategy(config.root_dir, model_config)

    assert strategy["type"] == ExtractEntityStrategyType.graph_intelligence
    assert strategy["llm"]["model"] == model_config.model


def test_resolved_strategy_merges_overrides_and_injects_llm() -> None:
    root_dir = Path(__file__).parent
    prompt_path = "prompt-a.txt"
    config = create_graphrag_config(
        {
            "models": DEFAULT_MODEL_CONFIG,
            "extract_graph": {
                "prompt": prompt_path,
                "max_gleanings": 5,
                "strategy": {
                    "type": "graph_intelligence",
                    "max_input_length": 1024,
                },
            },
        },
        root_dir=str(root_dir),
    )
    model_config = config.get_language_model_config(config.extract_graph.model_id)

    strategy = config.extract_graph.resolved_strategy(config.root_dir, model_config)

    assert strategy["type"] == ExtractEntityStrategyType.graph_intelligence
    assert strategy["llm"]["model"] == model_config.model
    assert strategy["max_gleanings"] == 5
    assert strategy["max_input_length"] == 1024
    assert (
        strategy["extraction_prompt"]
        == (root_dir / prompt_path).read_text(encoding="utf-8")
    )


def test_resolved_strategy_does_not_inject_llm_for_non_graph_strategy() -> None:
    root_dir = Path(__file__).parent
    config = create_graphrag_config(
        {
            "models": DEFAULT_MODEL_CONFIG,
            "extract_graph": {
                "strategy": {
                    "type": "huggingface_mrebel",
                },
            },
        },
        root_dir=str(root_dir),
    )
    model_config = config.get_language_model_config(config.extract_graph.model_id)

    strategy = config.extract_graph.resolved_strategy(config.root_dir, model_config)

    assert strategy["type"] == "huggingface_mrebel"
    assert "llm" not in strategy
