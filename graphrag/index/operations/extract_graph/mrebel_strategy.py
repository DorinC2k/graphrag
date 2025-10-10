"""Entity extraction strategy backed by the mREBEL Hugging Face model."""

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

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover - transformers is optional
    AutoModelForSeq2SeqLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


DEFAULT_MODEL_NAME = "Babelscape/mrebel-base"
DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_NUM_BEAMS = 3
DEFAULT_RELATION_WEIGHT = 1.0


class _MRebelModel:
    """Wrapper around the mREBEL Hugging Face model."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | int | None = None,
        revision: str | None = None,
    ) -> None:
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            msg = (
                "transformers is required to use the mREBEL extraction strategy. "
                "Install it with `pip install transformers`."
            )
            raise ImportError(msg)

        if torch is None:
            msg = (
                "torch is required to use the mREBEL extraction strategy. "
                "Install it with `pip install torch`."
            )
            raise ImportError(msg)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, revision=revision
        )
        self.model.eval()

        if device is None:
            if torch.cuda.is_available():  # pragma: no cover - depends on hardware
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)

    def extract_triples(
        self,
        text: str,
        *,
        max_input_length: int = DEFAULT_MAX_INPUT_LENGTH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = DEFAULT_NUM_BEAMS,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[str, str, str]]:
        """Run the underlying model and return extracted triples."""

        if generation_kwargs is None:
            generation_kwargs = {}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():  # type: ignore[union-attr]
            output_tokens = self.model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                **generation_kwargs,
            )

        decoded = self.tokenizer.batch_decode(  # type: ignore[union-attr]
            output_tokens,
            skip_special_tokens=True,
        )

        if not decoded:
            return []

        return _parse_mrebel_output(decoded[0])


_MODEL_CACHE: dict[tuple[str, str | int | None, str | None], _MRebelModel] = {}


def _get_or_create_model(
    model_name: str,
    *,
    device: str | int | None = None,
    revision: str | None = None,
) -> _MRebelModel:
    """Get a cached instance of the mREBEL model."""

    cache_key = (model_name, device, revision)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _MRebelModel(
            model_name,
            device=device,
            revision=revision,
        )
    return _MODEL_CACHE[cache_key]


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
    """Parse the raw mREBEL output into a list of triples."""

    if not text:
        return []

    normalized = text.replace("<subject>", "<subj>")
    normalized = normalized.replace("<relation>", "<rel>")
    normalized = normalized.replace("<object>", "<obj>")
    normalized = normalized.replace("</s>", " ").replace("<s>", " ")
    normalized = normalized.replace("<pad>", " ")

    triples: list[tuple[str, str, str]] = []
    for chunk in normalized.split("<triplet>"):
        chunk = chunk.strip()
        if not chunk:
            continue

        subject = _extract_component(chunk, "<subj>", "<rel>")
        relation = _extract_component(chunk, "<rel>", "<obj>")
        obj = _extract_component(chunk, "<obj>", None)

        if subject and relation and obj:
            triples.append((subject, relation, obj))

    return triples


def _extract_component(
    text: str,
    start_token: str,
    end_token: str | None,
) -> str | None:
    """Extract a component between two tokens."""

    start_index = text.find(start_token)
    if start_index == -1:
        return None

    start_index += len(start_token)
    if end_token is None:
        end_index = len(text)
    else:
        end_index = text.find(end_token, start_index)
        if end_index == -1:
            end_index = len(text)

    value = text[start_index:end_index].strip()
    if not value:
        return None

    return value


__all__ = ["run_huggingface_mrebel"]
