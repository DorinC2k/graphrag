import pytest
pytest.importorskip("sentence_transformers")

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.factory import ModelFactory

from tests.hf_provider import register_huggingface_embedding


@pytest.mark.asyncio
async def test_huggingface_embedding_vector_shape():
    register_huggingface_embedding()
    config = LanguageModelConfig(
        api_key="test",
        type="huggingface_embedding",
        model="sentence-transformers/all-MiniLM-L6-v2",
        encoding_model="cl100k_base",
    )
    model = ModelFactory.create_embedding_model(
        "huggingface_embedding", name="hf", config=config
    )
    vector = await model.aembed("GraphRAG integrates HF models")
    assert len(vector) == model.model.get_sentence_embedding_dimension()
