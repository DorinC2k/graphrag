"""Hugging Face mREBEL strategy for graph extraction."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import networkx as nx

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.extract_graph.typing import (
    Document,
    EntityExtractionResult,
    EntityTypes,
    StrategyConfig,
)
from graphrag.index.operations.shared.mrebel import (
    DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_BEAMS,
    get_or_create_mrebel_model,
    parse_mrebel_output,
)

DEFAULT_RELATION_WEIGHT = 1.0


def _get_or_create_model(
    model_name: str,
    *,
    device: str | int | None = None,
    revision: str | None = None,
):
    """Compatibility wrapper for the shared model cache."""

    return get_or_create_mrebel_model(model_name, device=device, revision=revision)


async def run_huggingface_mrebel(
    docs: list[Document],
    entity_types: EntityTypes,
    callbacks: WorkflowCallbacks | None,
    cache: PipelineCache | None,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the mREBEL entity extraction strategy."""

    del cache  # cache is unused but kept for interface parity

    model_name = args.get("model_name", DEFAULT_MODEL_NAME)
    revision = args.get("revision")
    device = args.get("device")
    default_entity_type = args.get(
        "default_entity_type", entity_types[0] if entity_types else "entity"
    )
    relation_weight = float(args.get("relation_weight", DEFAULT_RELATION_WEIGHT))

    max_input_length = int(args.get("max_input_length", DEFAULT_MAX_INPUT_LENGTH))
    max_new_tokens = int(args.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    num_beams = int(args.get("num_beams", DEFAULT_NUM_BEAMS))
    generation_kwargs = args.get("generation_kwargs", {}) or {}

    model = _get_or_create_model(model_name, device=device, revision=revision)

    graph = nx.DiGraph()

    node_sources: dict[str, set[str]] = defaultdict(set)
    edge_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    edge_descriptions: dict[tuple[str, str], set[str]] = defaultdict(set)

    for doc in docs:
        try:
            triples = await asyncio.to_thread(
                model.extract_triples,
                doc.text,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                generation_kwargs=generation_kwargs,
            )
        except Exception as exc:  # pragma: no cover - surface model failures
            if callbacks:
                callbacks.error("Entity Extraction Error", exc, doc.text, args)
            raise

        for subject, relation, obj in triples:
            subject = subject.strip()
            relation = relation.strip()
            obj = obj.strip()

            if not subject or not obj:
                continue

            if not graph.has_node(subject):
                graph.add_node(
                    subject,
                    title=subject,
                    type=default_entity_type,
                    description="",
                    source_id="",
                )
            if not graph.has_node(obj):
                graph.add_node(
                    obj,
                    title=obj,
                    type=default_entity_type,
                    description="",
                    source_id="",
                )

            node_sources[subject].add(doc.id)
            node_sources[obj].add(doc.id)

            if not graph.has_edge(subject, obj):
                graph.add_edge(
                    subject,
                    obj,
                    description="",
                    source_id="",
                    weight=0.0,
                )

            edge = graph[subject][obj]
            edge["weight"] = edge.get("weight", 0.0) + relation_weight
            edge_sources[(subject, obj)].add(doc.id)
            if relation:
                edge_descriptions[(subject, obj)].add(relation)

    # Finalize node attributes
    for node, data in graph.nodes(data=True):
        sources = node_sources.get(node, set())
        data["source_id"] = ",".join(sorted(sources))
        data.setdefault("title", node)
        data.setdefault("type", default_entity_type)
        data.setdefault("description", "")

    # Finalize edge attributes
    for subject, obj, data in graph.edges(data=True):
        sources = edge_sources.get((subject, obj), set())
        descriptions = edge_descriptions.get((subject, obj), set())
        data["source_id"] = ",".join(sorted(sources))
        if descriptions:
            data["description"] = "; ".join(sorted(descriptions))
        else:
            data.setdefault("description", "")

    entities = [
        {
            "title": node,
            "type": data.get("type", default_entity_type),
            "description": data.get("description", ""),
            "source_id": data.get("source_id", ""),
        }
        for node, data in graph.nodes(data=True)
    ]

    relationships = [
        {
            "source": source,
            "target": target,
            "description": data.get("description", ""),
            "source_id": data.get("source_id", ""),
            "weight": data.get("weight", relation_weight),
        }
        for source, target, data in graph.edges(data=True)
    ]

    return EntityExtractionResult(entities, relationships, graph)


def _parse_mrebel_output(text: str) -> list[tuple[str, str, str]]:
    """Compatibility wrapper for the shared parsing helper."""

    return parse_mrebel_output(text)


__all__ = ["run_huggingface_mrebel"]
