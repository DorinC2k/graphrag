# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity type generation module for fine-tuning."""

from pydantic import BaseModel
from pydantic.errors import PydanticUserError

from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompt_tune.defaults import DEFAULT_TASK
from graphrag.prompt_tune.prompt.entity_types import (
    ENTITY_TYPE_GENERATION_JSON_PROMPT,
    ENTITY_TYPE_GENERATION_PROMPT,
)


class EntityTypesResponse(BaseModel):
    """Entity types response model."""

    entity_types: list[str]


async def generate_entity_types(
    model: ChatModel,
    domain: str,
    persona: str,
    docs: str | list[str],
    task: str = DEFAULT_TASK,
    json_mode: bool = False,
) -> str | list[str]:
    """
    Generate entity type categories from a given set of (small) documents.

    Example Output:
    "entity_types": ['military unit', 'organization', 'person', 'location', 'event', 'date', 'equipment']
    """
    formatted_task = task.format(domain=domain)

    docs_str = "\n".join(docs) if isinstance(docs, list) else docs

    entity_types_prompt = (
        ENTITY_TYPE_GENERATION_JSON_PROMPT
        if json_mode
        else ENTITY_TYPE_GENERATION_PROMPT
    ).format(task=formatted_task, input_text=docs_str)

    history = [{"role": "system", "content": persona}]

    if json_mode:
        response = await model.achat(
            entity_types_prompt,
            history=history,
            json=json_mode,
            json_model=EntityTypesResponse,
        )
        parsed_model = response.parsed_response
        if parsed_model is None:
            return []

        if isinstance(parsed_model, BaseModel):
            data: dict | None = None

            dump_method = getattr(parsed_model, "model_dump", None)
            if callable(dump_method):
                try:
                    data = dump_method()
                except (PydanticUserError, TypeError, AttributeError):
                    data = None

            if data is None:
                dict_method = getattr(parsed_model, "dict", None)
                if callable(dict_method):
                    try:
                        data = dict_method()
                    except (TypeError, AttributeError):
                        data = None

            if data is None:
                try:
                    data = parsed_model.__dict__
                except AttributeError:
                    return []

            if not isinstance(data, dict):
                return []
        elif isinstance(parsed_model, dict):
            data = parsed_model
        else:
            return []

        entity_types = data.get("entity_types")
        return entity_types or []

    response = await model.achat(entity_types_prompt, history=history, json=json_mode)
    return str(response.output.content)
