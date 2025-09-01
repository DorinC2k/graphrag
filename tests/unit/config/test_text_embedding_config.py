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
