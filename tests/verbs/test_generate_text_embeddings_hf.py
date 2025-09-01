import pytest
pytest.importorskip("sentence_transformers")

import graphrag.config.defaults as defs
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.enums import ModelType
from graphrag.config.embeddings import all_embeddings
from graphrag.index.operations.embed_text.embed_text import TextEmbedStrategyType
from graphrag.index.workflows.generate_text_embeddings import run_workflow
from graphrag.utils.storage import load_table_from_storage

from tests.hf_provider import register_huggingface_embedding
from tests.verbs.util import FAKE_API_KEY, create_test_context


@pytest.mark.asyncio
async def test_generate_text_embeddings_with_huggingface():
    register_huggingface_embedding()
    context = await create_test_context(
        storage=[
            "documents",
            "relationships",
            "text_units",
            "entities",
            "community_reports",
        ]
    )

    model_configs = {
        defs.DEFAULT_CHAT_MODEL_ID: {
            "api_key": FAKE_API_KEY,
            "type": ModelType.AzureOpenAIChat.value,
            "api_base": "https://example.openai.azure.com/",
            "api_version": "2023-05-15",
            "deployment_name": "gpt-4o",
            "model": "gpt-4o",
            "model_supports_json": True,
        },
        defs.DEFAULT_EMBEDDING_MODEL_ID: {
            "api_key": FAKE_API_KEY,
            "type": "huggingface_embedding",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "encoding_model": "cl100k_base",
        },
    }

    config = create_graphrag_config({"models": model_configs})

    assert (
        config.get_language_model_config(defs.DEFAULT_CHAT_MODEL_ID).model
        == "gpt-4o"
    )
    assert (
        config.get_language_model_config(defs.DEFAULT_EMBEDDING_MODEL_ID).model
        == "sentence-transformers/all-MiniLM-L6-v2"
    )

    llm_settings = config.get_language_model_config(
        config.embed_text.model_id
    ).model_dump()
    config.embed_text.strategy = {
        "type": TextEmbedStrategyType.openai,
        "llm": llm_settings,
    }
    config.embed_text.names = list(all_embeddings)
    config.snapshots.embeddings = True

    await run_workflow(config, context)

    parquet_files = context.output_storage.keys()
    for field in all_embeddings:
        assert f"embeddings.{field}.parquet" in parquet_files

    entity_desc = await load_table_from_storage(
        "embeddings.entity.description", context.output_storage
    )
    first = entity_desc.iloc[0]["embedding"]
    assert len(first) == 384
