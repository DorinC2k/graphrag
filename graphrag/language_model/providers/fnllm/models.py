# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing fnllm model provider definitions."""

from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING, Any

from fnllm.openai import (
    create_openai_chat_llm,
    create_openai_client,
    create_openai_embeddings_llm,
)
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._types import NOT_GIVEN
from openai.lib.streaming.responses._events import ResponseTextDeltaEvent

from graphrag.language_model.providers.fnllm.events import FNLLMEvents
from graphrag.language_model.providers.fnllm.utils import (
    _create_cache,
    _create_error_handler,
    _create_openai_config,
    get_openai_model_parameters_from_config,
    is_responses_model,
    run_coroutine_sync,
)
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from fnllm.openai.types.client import OpenAIChatLLM as FNLLMChatLLM
    from fnllm.openai.types.client import OpenAIEmbeddingsLLM as FNLLMEmbeddingLLM

    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import (
        LanguageModelConfig,
    )


class _OpenAIResponsesChatModel:
    """Chat model wrapper that uses the OpenAI Responses API."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
        azure: bool = False,
    ) -> None:
        self.config = config
        self._error_handler = _create_error_handler(callbacks) if callbacks else None
        if azure:
            client_kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "azure_endpoint": config.api_base,
                "api_version": config.api_version,
                "timeout": config.request_timeout,
                "max_retries": config.max_retries,
            }
            self._client = AsyncAzureOpenAI(
                **{k: v for k, v in client_kwargs.items() if v is not None}
            )
        else:
            client_kwargs = {
                "api_key": config.api_key,
                "base_url": config.api_base or NOT_GIVEN,
                "organization": config.organization or NOT_GIVEN,
                "timeout": config.request_timeout,
                "max_retries": config.max_retries,
            }
            self._client = AsyncOpenAI(
                **{k: v for k, v in client_kwargs.items() if v is not NOT_GIVEN}
            )
        self._default_params = get_openai_model_parameters_from_config(config)
        self._model_identifier = (
            config.deployment_name if azure and config.deployment_name else config.model
        )

    @staticmethod
    def _normalize_history(history: list | None) -> list[dict[str, str]]:
        if history is None:
            return []
        normalized: list[dict[str, str]] = []
        for message in history:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                    else:
                        text_parts.append(str(part))
                content = "".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)
            normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _to_response_messages(
        history: list[dict[str, str]], prompt: str
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in history:
            messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        )
        return messages

    @staticmethod
    def _build_history(
        history: list[dict[str, str]], prompt: str, assistant_response: str
    ) -> list[dict[str, str]]:
        updated = [dict(message) for message in history]
        updated.append({"role": "user", "content": prompt})
        updated.append({"role": "assistant", "content": assistant_response})
        return updated

    def _merge_parameters(
        self,
        model_parameters: dict[str, Any] | None,
        extra_parameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {k: v for k, v in self._default_params.items() if v is not None}
        if model_parameters:
            params.update({k: v for k, v in model_parameters.items() if v is not None})
        if extra_parameters:
            params.update({k: v for k, v in extra_parameters.items() if v is not None})
        return params

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        json_mode = kwargs.pop("json", False)
        model_parameters = kwargs.pop("model_parameters", None)
        extra_parameters = kwargs.pop("model_params", None)
        params = self._merge_parameters(model_parameters, extra_parameters)

        normalized_history = self._normalize_history(history)
        input_messages = self._to_response_messages(normalized_history, prompt)

        request_kwargs: dict[str, Any] = {
            "model": self._model_identifier,
            "input": input_messages,
        }
        request_kwargs.update(params)
        if json_mode:
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        try:
            response = await self._client.responses.create(**request_kwargs)
        except Exception as error:  # pragma: no cover - defensive
            if self._error_handler is not None:
                self._error_handler(
                    error,
                    traceback.format_exc(),
                    {"prompt": prompt, "history_length": len(normalized_history)},
                )
            raise

        try:
            content = response.output_text()
        except TypeError:  # pragma: no cover - defensive
            content = response.output_text  # type: ignore[assignment]
        if not isinstance(content, str):
            content = str(content or "")
        parsed_json: Any | None = None
        if json_mode and content:
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                parsed_json = None

        history_payload = self._build_history(normalized_history, prompt, content)

        metrics = None
        if response.usage is not None:
            metrics = {"usage": response.usage.model_dump()}

        return BaseModelResponse(
            output=BaseModelOutput(
                content=content or "",
                full_response=response.model_dump(),
            ),
            parsed_response=parsed_json,
            history=history_payload,
            cache_hit=False,
            tool_calls=[],
            metrics=metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        json_mode = kwargs.pop("json", False)
        model_parameters = kwargs.pop("model_parameters", None)
        extra_parameters = kwargs.pop("model_params", None)
        params = self._merge_parameters(model_parameters, extra_parameters)

        normalized_history = self._normalize_history(history)
        input_messages = self._to_response_messages(normalized_history, prompt)

        request_kwargs: dict[str, Any] = {
            "model": self._model_identifier,
            "input": input_messages,
            "stream": True,
        }
        request_kwargs.update(params)
        if json_mode:
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        stream = await self._client.responses.create(**request_kwargs)
        async with stream:
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    if isinstance(event, ResponseTextDeltaEvent):
                        yield event.delta


class OpenAIChatFNLLM:
    """An OpenAI Chat Model provider using the fnllm library."""

    model: FNLLMChatLLM | None

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        self.config = config
        self._responses_model: _OpenAIResponsesChatModel | None = None

        if is_responses_model(config.model):
            self.model = None
            self._responses_model = _OpenAIResponsesChatModel(
                name=name, config=config, callbacks=callbacks, cache=cache
            )
        else:
            model_config = _create_openai_config(config, azure=False)
            error_handler = _create_error_handler(callbacks) if callbacks else None
            model_cache = _create_cache(cache, name)
            client = create_openai_client(model_config)
            self.model = create_openai_chat_llm(
                model_config,
                client=client,
                cache=model_cache,
                events=FNLLMEvents(error_handler) if error_handler else None,
            )

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if self._responses_model is not None:
            return await self._responses_model.achat(
                prompt, history=history, **kwargs
            )

        if self.model is None:
            msg = "Chat model is not initialized"
            raise RuntimeError(msg)

        if history is None:
            response = await self.model(prompt, **kwargs)
        else:
            response = await self.model(prompt, history=history, **kwargs)
        return BaseModelResponse(
            output=BaseModelOutput(
                content=response.output.content,
                full_response=response.output.raw_model.to_dict(),
            ),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if self._responses_model is not None:
            async for chunk in self._responses_model.achat_stream(
                prompt, history=history, **kwargs
            ):
                yield chunk
            return

        if self.model is None:
            msg = "Chat model is not initialized"
            raise RuntimeError(msg)

        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class OpenAIEmbeddingFNLLM:
    """An OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))


class AzureOpenAIChatFNLLM:
    """An Azure OpenAI Chat LLM provider using the fnllm library."""

    model: FNLLMChatLLM | None

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        self._responses_model: _OpenAIResponsesChatModel | None = None

        if is_responses_model(config.model):
            self.model = None
            self._responses_model = _OpenAIResponsesChatModel(
                name=name,
                config=config,
                callbacks=callbacks,
                cache=cache,
                azure=True,
            )
        else:
            model_config = _create_openai_config(config, azure=True)
            error_handler = _create_error_handler(callbacks) if callbacks else None
            model_cache = _create_cache(cache, name)
            client = create_openai_client(model_config)
            self.model = create_openai_chat_llm(
                model_config,
                client=client,
                cache=model_cache,
                events=FNLLMEvents(error_handler) if error_handler else None,
            )
        self.config = config

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if self._responses_model is not None:
            return await self._responses_model.achat(prompt, history=history, **kwargs)

        if history is None:
            response = await self.model(prompt, **kwargs)
        else:
            response = await self.model(prompt, history=history, **kwargs)
        return BaseModelResponse(
            output=BaseModelOutput(
                content=response.output.content,
                full_response=response.output.raw_model.to_dict(),
            ),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if self._responses_model is not None:
            async for chunk in self._responses_model.achat_stream(
                prompt, history=history, **kwargs
            ):
                yield chunk
            return

        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class AzureOpenAIEmbeddingFNLLM:
    """An Azure OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))
