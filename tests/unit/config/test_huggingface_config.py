import yaml

import graphrag.config.defaults as defs
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.enums import ModelType
from graphrag.language_model.factory import ModelFactory
from tests.mock_provider import MockEmbeddingLLM


def test_graph_rag_config_supports_hf_and_azure_chat():
    ModelFactory.register_embedding(
        "huggingface_embedding", lambda **kwargs: MockEmbeddingLLM()
    )
    with open("tests/fixtures/huggingface/settings.yml", "r", encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)

    config = create_graphrag_config(settings)

    chat_model = config.get_language_model_config(defs.DEFAULT_CHAT_MODEL_ID)
    embed_model = config.get_language_model_config(defs.DEFAULT_EMBEDDING_MODEL_ID)

    assert chat_model.type == ModelType.AzureOpenAIChat
    assert chat_model.model == "gpt-5-nano"
    assert embed_model.type == "huggingface_embedding"
    assert embed_model.model == "sentence-transformers/all-MiniLM-L6-v2"
