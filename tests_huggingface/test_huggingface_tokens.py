import pytest

pytest.importorskip("tiktoken")

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.providers.huggingface.models import (
    HuggingFaceEmbeddingModel,
)


def test_huggingface_model_uses_primary_env_token(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "hf_primary")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_hub")
    monkeypatch.setenv("HUGGING_FACE_TOKEN_READ_KEY", "hf_legacy")

    config = LanguageModelConfig(
        type="huggingface_embedding",
        model="sentence-transformers/all-MiniLM-L6-v2",
        encoding_model="cl100k_base",
        api_base="https://example.endpoint",
    )

    model = HuggingFaceEmbeddingModel(name="hf", config=config)

    assert model.api_key == "hf_primary"


def test_huggingface_model_strips_bearer_prefix(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "Bearer hf_secret")

    config = LanguageModelConfig(
        type="huggingface_embedding",
        model="sentence-transformers/all-MiniLM-L6-v2",
        encoding_model="cl100k_base",
        api_base="https://example.endpoint",
    )

    model = HuggingFaceEmbeddingModel(name="hf", config=config)

    assert model.api_key == "hf_secret"


def test_huggingface_model_falls_back_to_hub_env_token(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_API_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_hub")

    config = LanguageModelConfig(
        type="huggingface_embedding",
        model="sentence-transformers/all-MiniLM-L6-v2",
        encoding_model="cl100k_base",
        api_base="https://example.endpoint",
    )

    model = HuggingFaceEmbeddingModel(name="hf", config=config)

    assert model.api_key == "hf_hub"
