# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'JsonPipelineCache' model."""

import json
from typing import Any

from pydantic import BaseModel
from pydantic.errors import PydanticUserError

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.storage.pipeline_storage import PipelineStorage


class JsonPipelineCache(PipelineCache):
    """File pipeline cache class definition."""

    _storage: PipelineStorage
    _encoding: str

    def __init__(self, storage: PipelineStorage, encoding="utf-8"):
        """Init method definition."""
        self._storage = storage
        self._encoding = encoding

    async def get(self, key: str) -> str | None:
        """Get method definition."""
        if await self.has(key):
            try:
                data = await self._storage.get(key, encoding=self._encoding)
                data = json.loads(data)
            except UnicodeDecodeError:
                await self._storage.delete(key)
                return None
            except json.decoder.JSONDecodeError:
                await self._storage.delete(key)
                return None
            else:
                return data.get("result")

        return None

    async def set(self, key: str, value: Any, debug_data: dict | None = None) -> None:
        """Set method definition."""
        if value is None:
            return
        serialized_value = value

        if isinstance(value, BaseModel):
            try:
                serialized_value = value.model_dump()
            except PydanticUserError:
                serialized_value = str(value)

        data = {"result": serialized_value, **(debug_data or {})}

        try:
            payload = json.dumps(data, ensure_ascii=False)
        except TypeError:
            data["result"] = str(serialized_value)
            payload = json.dumps(data, ensure_ascii=False)

        await self._storage.set(key, payload, encoding=self._encoding)

    async def has(self, key: str) -> bool:
        """Has method definition."""
        return await self._storage.has(key)

    async def delete(self, key: str) -> None:
        """Delete method definition."""
        if await self.has(key):
            await self._storage.delete(key)

    async def clear(self) -> None:
        """Clear method definition."""
        await self._storage.clear()

    def child(self, name: str) -> "JsonPipelineCache":
        """Child method definition."""
        return JsonPipelineCache(self._storage.child(name), encoding=self._encoding)
