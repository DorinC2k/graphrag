from dataclasses import dataclass

from graphrag.config.enums import ModelType
from graphrag.config.models.text_embedding_config import TextEmbeddingConfig
from graphrag.index.operations.embed_text.embed_text import TextEmbedStrategyType


@dataclass
class DummyModelConfig:
    type: ModelType
    concurrent_requests: int = 1

    def model_dump(self) -> dict:
        return {"type": self.type, "model": "dummy"}


def test_resolved_strategy_huggingface():
    model_config = DummyModelConfig(type=ModelType.HuggingFaceEmbedding)
    config = TextEmbeddingConfig()
    strategy = config.resolved_strategy(model_config)
    assert strategy["type"] == TextEmbedStrategyType.huggingface


def test_resolved_strategy_merges_overrides_and_injects_llm() -> None:
    model_config = DummyModelConfig(type=ModelType.OpenAIEmbedding)
    config = TextEmbeddingConfig(
        strategy={
            "type": "openai",
            "batch_size": 10,
        }
    )

    strategy = config.resolved_strategy(model_config)

    assert strategy["type"] == TextEmbedStrategyType.openai
    assert strategy["llm"]["model"] == "dummy"
    assert strategy["batch_size"] == 10
    assert strategy["num_threads"] == model_config.concurrent_requests


def test_resolved_strategy_does_not_inject_llm_for_custom_strategy() -> None:
    model_config = DummyModelConfig(type=ModelType.OpenAIEmbedding)
    config = TextEmbeddingConfig(strategy={"type": "custom"})

    strategy = config.resolved_strategy(model_config)

    assert strategy["type"] == "custom"
    assert "llm" not in strategy
