"""Hugging Face mREBEL strategy for claim extraction."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.extract_covariates.typing import (
    Covariate,
    CovariateExtractionResult,
)
from graphrag.index.operations.shared.mrebel import (
    DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_BEAMS,
    get_or_create_mrebel_model,
)


async def run_huggingface_mrebel_claims(
    input: Iterable[str],
    entity_types: list[str],
    resolved_entities_map: dict[str, str],
    callbacks: WorkflowCallbacks | None,
    cache: PipelineCache | None,
    args: dict[str, Any],
) -> CovariateExtractionResult:
    """Extract claim-like covariates with the mREBEL model."""

    del cache, entity_types  # parameters kept for interface parity

    model_name = args.get("model_name", DEFAULT_MODEL_NAME)
    revision = args.get("revision")
    device = args.get("device")

    max_input_length = int(args.get("max_input_length", DEFAULT_MAX_INPUT_LENGTH))
    max_new_tokens = int(args.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    num_beams = int(args.get("num_beams", DEFAULT_NUM_BEAMS))
    generation_kwargs = args.get("generation_kwargs", {}) or {}

    model = get_or_create_mrebel_model(model_name, device=device, revision=revision)

    covariates: list[Covariate] = []
    record_id = 0

    for text in input:
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
                callbacks.error("Claim Extraction Error", exc, text, args)
            raise

        for subject, relation, obj in triples:
            subject = resolved_entities_map.get(subject, subject).strip()
            obj = resolved_entities_map.get(obj, obj).strip()
            relation = relation.strip()

            if not subject or not obj or not relation:
                continue

            description = f"{subject} {relation} {obj}".strip()
            record_id += 1

            covariates.append(
                Covariate(
                    subject_id=subject,
                    object_id=obj,
                    type=relation,
                    description=description,
                    source_text=[text],
                    record_id=record_id,
                )
            )

    return CovariateExtractionResult(covariates)


__all__ = ["run_huggingface_mrebel_claims"]
