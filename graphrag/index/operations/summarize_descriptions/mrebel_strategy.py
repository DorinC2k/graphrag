"""Hugging Face mREBEL strategy for description summarization."""

from __future__ import annotations

import asyncio

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.summarize_descriptions.typing import (
    StrategyConfig,
    SummarizedDescriptionResult,
)
from graphrag.index.operations.shared.mrebel import (
    DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_BEAMS,
    get_or_create_mrebel_model,
)


async def run_huggingface_mrebel_summarization(
    id: str | tuple[str, str],
    descriptions: list[str],
    callbacks: WorkflowCallbacks | None,
    cache: PipelineCache | None,
    args: StrategyConfig,
) -> SummarizedDescriptionResult:
    """Summarize entity or relationship descriptions with mREBEL."""

    del cache  # kept for interface parity

    model_name = args.get("model_name", DEFAULT_MODEL_NAME)
    revision = args.get("revision")
    device = args.get("device")

    max_input_length = int(args.get("max_input_length", DEFAULT_MAX_INPUT_LENGTH))
    max_new_tokens = int(args.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    num_beams = int(args.get("num_beams", DEFAULT_NUM_BEAMS))
    generation_kwargs = args.get("generation_kwargs", {}) or {}
    max_summary_length = int(args.get("max_summary_length", 500))

    model = get_or_create_mrebel_model(model_name, device=device, revision=revision)

    sentences: list[str] = []

    for text in descriptions:
        if not text:
            continue

        try:
            triples = await asyncio.to_thread(
                model.extract_triples,
                text,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                generation_kwargs=generation_kwargs,
            )
        except Exception as exc:  # pragma: no cover - surface model failures
            if callbacks:
                callbacks.error("Description Summarization Error", exc, text, args)
            raise

        for subject, relation, obj in triples:
            subject = subject.strip()
            relation = relation.strip()
            obj = obj.strip()

            if not subject or not relation or not obj:
                continue

            sentence = f"{subject} {relation} {obj}".strip()
            if sentence not in sentences:
                sentences.append(sentence)

    if not sentences:
        summary = " ".join(description for description in descriptions if description).strip()
    else:
        summary = ". ".join(sentences)

    summary = summary.strip()
    if len(summary) > max_summary_length:
        summary = summary[: max_summary_length - 1].rstrip() + "…"

    return SummarizedDescriptionResult(id=id, description=summary)


__all__ = ["run_huggingface_mrebel_summarization"]
