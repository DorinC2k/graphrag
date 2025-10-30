# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing mock model provider definitions."""

from collections.abc import AsyncGenerator, Generator
from typing import Any

try:  # pragma: no cover - optional dependency for tests
    from pydantic import BaseModel
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for minimal environments
    class BaseModel:  # type: ignore[no-redef]
        """Fallback BaseModel implementation used when pydantic is unavailable."""

        def model_dump_json(self) -> str:
            from json import dumps

            return dumps(self.__dict__)

from graphrag.config.enums import ModelType
try:  # pragma: no cover - optional dependency chain for tests
    from graphrag.config.models.language_model_config import LanguageModelConfig
except Exception:  # pragma: no cover - fallback when pydantic or other deps are unavailable
    class LanguageModelConfig:  # type: ignore[no-redef]
        """Lightweight stand-in used when full configuration models cannot be imported."""

        def __init__(
            self,
            *,
            type: Any,
            model: str,
            api_key: str | None = None,
            responses: list[str] | None = None,
            **_: Any,
        ) -> None:
            self.type = type
            self.model = model
            self.api_key = api_key
            self.responses = responses or []
try:  # pragma: no cover - optional dependency chain for tests
    from graphrag.language_model.response.base import (
        BaseModelOutput,
        BaseModelResponse,
        ModelResponse,
    )
except Exception:  # pragma: no cover - fallback when pydantic is unavailable
    class BaseModelOutput:  # type: ignore[no-redef]
        def __init__(
            self,
            *,
            content: str,
            full_response: dict[str, Any] | None = None,
        ) -> None:
            self.content = content
            self.full_response = full_response

    class BaseModelResponse:  # type: ignore[no-redef]
        def __init__(
            self,
            *,
            output: BaseModelOutput,
            parsed_response: BaseModel | None = None,
            history: list[Any] | None = None,
            tool_calls: list[Any] | None = None,
            metrics: Any | None = None,
            cache_hit: bool | None = None,
        ) -> None:
            self.output = output
            self.parsed_response = parsed_response
            self.history = history or []
            self.tool_calls = tool_calls or []
            self.metrics = metrics
            self.cache_hit = cache_hit

    ModelResponse = BaseModelResponse  # type: ignore[assignment]


class MockChatLLM:
    """A mock chat LLM provider."""

    def __init__(
        self,
        responses: list[str | BaseModel] | None = None,
        config: LanguageModelConfig | None = None,
        json: bool = False,
        **kwargs: Any,
    ):
        self.responses = config.responses if config and config.responses else responses
        self.response_index = 0
        self.config = config or LanguageModelConfig(
            type=ModelType.MockChat, model="gpt-4o", api_key="mock"
        )

    async def achat(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Return the next response in the list."""
        return self.chat(prompt, history, **kwargs)

    async def achat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Return the next response in the list."""
        if not self.responses:
            return

        for response in self.responses:
            response = (
                response.model_dump_json()
                if isinstance(response, BaseModel)
                else response
            )

            yield response

    def chat(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Return the next response in the list."""
        if not self.responses:
            return BaseModelResponse(output=BaseModelOutput(content=""))

        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1

        parsed_json = response if isinstance(response, BaseModel) else None
        response = (
            response.model_dump_json() if isinstance(response, BaseModel) else response
        )

        return BaseModelResponse(
            output=BaseModelOutput(content=response),
            parsed_response=parsed_json,
        )

    def chat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> Generator[str, None]:
        """Return the next response in the list."""
        raise NotImplementedError


class MockEmbeddingLLM:
    """A mock embedding LLM provider."""

    def __init__(self, **kwargs: Any):
        self.config = LanguageModelConfig(
            type=ModelType.MockEmbedding, model="text-embedding-ada-002", api_key="mock"
        )

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate an embedding for the input text."""
        if isinstance(text_list, str):
            return [[1.0, 1.0, 1.0]]
        return [[1.0, 1.0, 1.0] for _ in text_list]

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate an embedding for the input text."""
        return [1.0, 1.0, 1.0]

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate an embedding for the input text."""
        return [1.0, 1.0, 1.0]

    async def aembed_batch(
        self, text_list: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Generate an embedding for the input text."""
        if isinstance(text_list, str):
            return [[1.0, 1.0, 1.0]]
        return [[1.0, 1.0, 1.0] for _ in text_list]
