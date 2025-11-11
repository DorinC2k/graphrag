"""Tests for fnllm OpenAI chat providers using Responses API."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from graphrag.cache.json_pipeline_cache import JsonPipelineCache
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.providers.fnllm.models import (
    AzureOpenAIChatFNLLM,
    OpenAIChatFNLLM,
)
from graphrag.storage.memory_pipeline_storage import MemoryPipelineStorage


class _DummyResponse:
    def __init__(self, content: str = "content") -> None:
        self._content = content
        self.usage = None

    def output_text(self) -> str:
        return self._content

    def model_dump(self) -> dict[str, Any]:
        return {"output": self._content}


class _DummyResponses:
    def __init__(self, capture: dict[str, Any]) -> None:
        self._capture = capture
        self._calls = 0

    async def create(self, **kwargs: Any) -> _DummyResponse:
        self._calls += 1
        self._capture.clear()
        self._capture.update(kwargs)
        self._capture["__call_count__"] = self._calls
        return _DummyResponse()


class _DummyAsyncClient:
    def __init__(self, capture: dict[str, Any], **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.responses = _DummyResponses(capture)


class _DummyChatCompletions:
    def __init__(self, capture: dict[str, Any]) -> None:
        self._capture = capture
        self._calls = 0

    async def create(self, **kwargs: Any) -> _DummyChatCompletionResponse:
        self._calls += 1
        self._capture.clear()
        self._capture.update(kwargs)
        self._capture["__call_count__"] = self._calls
        return _DummyChatCompletionResponse()


class _DummyChat:
    def __init__(self, capture: dict[str, Any]) -> None:
        self.completions = _DummyChatCompletions(capture)


class _DummyChatCompletionMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _DummyChatCompletionChoice:
    def __init__(self, content: str) -> None:
        self.delta = SimpleNamespace(content=content)
        self.message = _DummyChatCompletionMessage(content)


class _DummyChatCompletionResponse:
    def __init__(self, content: str = "content") -> None:
        self._content = content
        self.choices = [_DummyChatCompletionChoice(content)]
        self.usage = None

    def model_dump(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": self._content}}]}


def test_openai_gpt5_requests_use_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    async_client = _DummyAsyncClient

    monkeypatch.setattr(
        "graphrag.language_model.providers.fnllm.models.AsyncOpenAI",
        lambda **kwargs: async_client(captured_request, **kwargs),
    )

    config = LanguageModelConfig(
        type=ModelType.OpenAIChat.value,
        model="gpt-5-nano",
        api_key="test-key",
    )

    model = OpenAIChatFNLLM(name="test", config=config)

    response = asyncio.run(model.achat("hello"))

    assert response.output.content == "content"
    assert "messages" not in captured_request
    assert captured_request["model"] == "gpt-5-nano"
    assert captured_request["input"][-1]["content"][0]["text"] == "hello"
    assert "temperature" not in captured_request


def test_openai_gpt5_json_mode_sets_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    async_client = _DummyAsyncClient

    monkeypatch.setattr(
        "graphrag.language_model.providers.fnllm.models.AsyncOpenAI",
        lambda **kwargs: async_client(captured_request, **kwargs),
    )

    config = LanguageModelConfig(
        type=ModelType.OpenAIChat.value,
        model="gpt-5-nano",
        api_key="test-key",
    )

    model = OpenAIChatFNLLM(name="test", config=config)

    response = asyncio.run(model.achat("hello", json=True))

    assert response.output.content == "content"
    assert captured_request["response_format"] == {"type": "json_object"}


def test_openai_gpt5_uses_pipeline_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    async_client = _DummyAsyncClient

    monkeypatch.setattr(
        "graphrag.language_model.providers.fnllm.models.AsyncOpenAI",
        lambda **kwargs: async_client(captured_request, **kwargs),
    )

    config = LanguageModelConfig(
        type=ModelType.OpenAIChat.value,
        model="gpt-5-nano",
        api_key="test-key",
    )

    cache = JsonPipelineCache(MemoryPipelineStorage())
    model = OpenAIChatFNLLM(name="test", config=config, cache=cache)

    response_first = asyncio.run(model.achat("hello"))
    assert response_first.cache_hit is False
    assert captured_request["__call_count__"] == 1

    response_second = asyncio.run(model.achat("hello"))
    assert response_second.cache_hit is True
    assert captured_request["__call_count__"] == 1


def test_azure_gpt5_requests_use_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    class _AzureDummyClient:
        def __init__(self, capture: dict[str, Any], **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.chat = _DummyChat(capture)

    monkeypatch.setattr(
        "graphrag.language_model.providers.fnllm.models.AsyncAzureOpenAI",
        lambda **kwargs: _AzureDummyClient(captured_request, **kwargs),
    )

    config = LanguageModelConfig(
        type=ModelType.AzureOpenAIChat.value,
        model="gpt-5-nano",
        deployment_name="gpt5-deployment",
        api_key="test-key",
        api_base="https://example.openai.azure.com/",
        api_version="2024-05-01-preview",
    )

    model = AzureOpenAIChatFNLLM(name="test", config=config)

    response = asyncio.run(model.achat("hi"))

    assert response.output.content == "content"
    assert captured_request["model"] == "gpt5-deployment"
    assert captured_request["messages"][-1]["content"] == "hi"
    assert "temperature" not in captured_request


def test_azure_gpt5_json_mode_sets_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    class _AzureDummyClient:
        def __init__(self, capture: dict[str, Any], **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.chat = _DummyChat(capture)

    monkeypatch.setattr(
        "graphrag.language_model.providers.fnllm.models.AsyncAzureOpenAI",
        lambda **kwargs: _AzureDummyClient(captured_request, **kwargs),
    )

    config = LanguageModelConfig(
        type=ModelType.AzureOpenAIChat.value,
        model="gpt-5-nano",
        deployment_name="gpt5-deployment",
        api_key="test-key",
        api_base="https://example.openai.azure.com/",
        api_version="2024-05-01-preview",
    )

    model = AzureOpenAIChatFNLLM(name="test", config=config)

    response = asyncio.run(model.achat("hi", json=True))

    assert response.output.content == "content"
    assert captured_request["response_format"] == {"type": "json_object"}


def test_azure_gpt5_uses_pipeline_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    class _AzureDummyClient:
        def __init__(self, capture: dict[str, Any], **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.chat = _DummyChat(capture)

    monkeypatch.setattr(
        "graphrag.language_model.providers.fnllm.models.AsyncAzureOpenAI",
        lambda **kwargs: _AzureDummyClient(captured_request, **kwargs),
    )

    config = LanguageModelConfig(
        type=ModelType.AzureOpenAIChat.value,
        model="gpt-5-nano",
        deployment_name="gpt5-deployment",
        api_key="test-key",
        api_base="https://example.openai.azure.com/",
        api_version="2024-05-01-preview",
    )

    cache = JsonPipelineCache(MemoryPipelineStorage())
    model = AzureOpenAIChatFNLLM(name="test", config=config, cache=cache)

    response_first = asyncio.run(model.achat("hi"))
    assert response_first.cache_hit is False
    assert captured_request["__call_count__"] == 1

    response_second = asyncio.run(model.achat("hi"))
    assert response_second.cache_hit is True
    assert captured_request["__call_count__"] == 1

