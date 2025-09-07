"""Tests for HuggingFaceEmbeddingModel remote API usage via ModelFactory."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.factory import ModelFactory


@pytest.mark.asyncio
async def test_remote_hf_embedding_model_factory_send_auth_and_parse():
    fake_response = MagicMock()
    fake_response.json.return_value = [[0.1, 0.2]]
    fake_response.raise_for_status.return_value = None

    config = LanguageModelConfig(
        api_key="tok",
        api_base="https://example.com",
        model="unused",
        type=ModelType.HuggingFaceEmbedding,
        encoding_model="cl100k_base",
    )
    model = ModelFactory.create_embedding_model(
        ModelType.HuggingFaceEmbedding, name="hf", config=config
    )

    with patch(
        "graphrag.language_model.providers.huggingface.models.requests.post",
        return_value=fake_response,
    ) as mock_post:
        result = await model.aembed_batch(["hello"])

    mock_post.assert_called_once()
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer tok"
    assert headers["Accept"] == "application/json"
    assert headers["Content-Type"] == "application/json"
    assert result == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_remote_hf_embedding_model_factory_http_error():
    config = LanguageModelConfig(
        api_key="tok",
        api_base="https://example.com",
        model="unused",
        type=ModelType.HuggingFaceEmbedding,
        encoding_model="cl100k_base",
    )
    model = ModelFactory.create_embedding_model(
        ModelType.HuggingFaceEmbedding, name="hf", config=config
    )

    with patch(
        "graphrag.language_model.providers.huggingface.models.requests.post",
        side_effect=requests.HTTPError("boom"),
    ), pytest.raises(RuntimeError):
        await model.aembed_batch(["hi"])

