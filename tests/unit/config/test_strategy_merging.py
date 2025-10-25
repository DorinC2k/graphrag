"""Tests covering strategy override merging behaviour."""

from pathlib import Path

from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.index.operations.extract_covariates.typing import (
    ClaimExtractionStrategyType,
)
from graphrag.index.operations.summarize_communities.typing import (
    CreateCommunityReportsStrategyType,
)
from graphrag.index.operations.summarize_descriptions.typing import (
    SummarizeStrategyType,
)

from .utils import DEFAULT_MODEL_CONFIG


def _get_config(values: dict) -> tuple:
    root_dir = Path(__file__).parent
    config = create_graphrag_config(
        {"models": DEFAULT_MODEL_CONFIG, **values},
        root_dir=str(root_dir),
    )
    return config, root_dir


def test_summarize_descriptions_strategy_injects_llm() -> None:
    config, root_dir = _get_config(
        {
            "summarize_descriptions": {
                "prompt": "prompt-b.txt",
                "strategy": {"type": "graph_intelligence"},
            }
        }
    )
    model_config = config.get_language_model_config(
        config.summarize_descriptions.model_id
    )

    strategy = config.summarize_descriptions.resolved_strategy(
        config.root_dir, model_config
    )

    assert strategy["type"] == SummarizeStrategyType.graph_intelligence
    assert strategy["llm"]["model"] == model_config.model
    assert (
        strategy["summarize_prompt"]
        == (root_dir / "prompt-b.txt").read_text(encoding="utf-8")
    )


def test_community_reports_strategy_injects_llm() -> None:
    config, root_dir = _get_config(
        {
            "community_reports": {
                "graph_prompt": "prompt-c.txt",
                "text_prompt": "prompt-d.txt",
                "strategy": {"type": "graph_intelligence"},
            }
        }
    )
    model_config = config.get_language_model_config(
        config.community_reports.model_id
    )

    strategy = config.community_reports.resolved_strategy(
        config.root_dir, model_config
    )

    assert strategy["type"] == CreateCommunityReportsStrategyType.graph_intelligence
    assert strategy["llm"]["model"] == model_config.model
    assert (
        strategy["graph_prompt"]
        == (root_dir / "prompt-c.txt").read_text(encoding="utf-8")
    )
    assert (
        strategy["text_prompt"]
        == (root_dir / "prompt-d.txt").read_text(encoding="utf-8")
    )


def test_claim_extraction_strategy_injects_llm() -> None:
    config, root_dir = _get_config(
        {
            "extract_claims": {
                "prompt": "prompt-a.txt",
                "strategy": {"type": "graph_intelligence"},
            }
        }
    )
    model_config = config.get_language_model_config(config.extract_claims.model_id)

    strategy = config.extract_claims.resolved_strategy(
        config.root_dir, model_config
    )

    assert strategy["type"] == ClaimExtractionStrategyType.graph_intelligence
    assert strategy["llm"]["model"] == model_config.model
    assert (
        strategy["extraction_prompt"]
        == (root_dir / "prompt-a.txt").read_text(encoding="utf-8")
    )
