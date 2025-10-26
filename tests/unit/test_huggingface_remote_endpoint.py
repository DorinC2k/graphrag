# Copyright (c) 2025 Microsoft Corporation.
# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the remote Hugging Face embedding strategy."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.embed_text.strategies.huggingface import (
    DEFAULT_REMOTE_MAX_BATCH_SIZE,
    run,
)


@pytest.mark.asyncio
async def test_remote_hf_embeddings_send_auth_and_parse():
    cache = MagicMock(spec=PipelineCache)
    callbacks = MagicMock(spec=WorkflowCallbacks)
    callbacks.progress = MagicMock()

    fake_response = MagicMock()
    fake_response.json.return_value = [[0.1, 0.2]]
    fake_response.raise_for_status.return_value = None

    with patch(
        "graphrag.index.operations.embed_text.strategies.huggingface.requests.post",
        return_value=fake_response,
    ) as mock_post:
        result = await run(
            ["hello"],
            callbacks,
            cache,
            {
                "llm": {
                    "model": "unused",
                    "api_base": "https://example.com",
                    "api_key": "tok",
                }
            },
        )

    mock_post.assert_called_once()
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer tok"
    assert headers["Accept"] == "application/json"
    assert headers["Content-Type"] == "application/json"
    assert result.embeddings == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_remote_hf_embeddings_respects_max_batch_size():
    cache = MagicMock(spec=PipelineCache)
    callbacks = MagicMock(spec=WorkflowCallbacks)
    callbacks.progress = MagicMock()

    inputs = [f"text {i}" for i in range(7)]
    batches = [[0, 1, 2], [3, 4, 5], [6]]

    responses = []
    for batch in batches:
        fake_response = MagicMock()
        fake_response.json.return_value = [[float(i)] for i in batch]
        fake_response.raise_for_status.return_value = None
        responses.append(fake_response)

    with patch(
        "graphrag.index.operations.embed_text.strategies.huggingface.requests.post",
        side_effect=responses,
    ) as mock_post:
        result = await run(
            inputs,
            callbacks,
            cache,
            {
                "batch_size": 128,
                "max_remote_batch_size": len(batches[0]),
                "llm": {
                    "model": "unused",
                    "api_base": "https://example.com",
                    "api_key": "tok",
                },
            },
        )

    assert mock_post.call_count == len(batches)
    for call, batch in zip(mock_post.call_args_list, batches, strict=True):
        assert call.kwargs["json"]["inputs"] == [inputs[i] for i in batch]

    assert [row[0] for row in result.embeddings] == [float(i) for i in range(len(inputs))]
    # Ensure the default cap still exists so users are protected even without overrides
    assert DEFAULT_REMOTE_MAX_BATCH_SIZE == 32


@pytest.mark.asyncio
async def test_remote_hf_embeddings_http_error():
    cache = MagicMock(spec=PipelineCache)
    callbacks = MagicMock(spec=WorkflowCallbacks)
    callbacks.progress = MagicMock()

    with (
        patch(
            "graphrag.index.operations.embed_text.strategies.huggingface.requests.post",
            side_effect=requests.HTTPError("boom"),
        ),
        pytest.raises(RuntimeError),
    ):
        await run(
            ["hi"],
            callbacks,
            cache,
            {
                "llm": {
                    "model": "unused",
                    "api_base": "https://example.com",
                    "api_key": "tok",
                }
            },
        )
