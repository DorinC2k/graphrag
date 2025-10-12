"""Utilities for working with the Hugging Face mREBEL model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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


class MRebelModel:
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

        return parse_mrebel_output(decoded[0])


_MODEL_CACHE: dict[tuple[str, str | int | None, str | None], MRebelModel] = {}


def get_or_create_mrebel_model(
    model_name: str,
    *,
    device: str | int | None = None,
    revision: str | None = None,
) -> MRebelModel:
    """Get a cached instance of the mREBEL model."""

    cache_key = (model_name, device, revision)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = MRebelModel(
            model_name,
            device=device,
            revision=revision,
        )
    return _MODEL_CACHE[cache_key]


def parse_mrebel_output(text: str) -> list[tuple[str, str, str]]:
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


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MAX_INPUT_LENGTH",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_NUM_BEAMS",
    "MRebelModel",
    "get_or_create_mrebel_model",
    "parse_mrebel_output",
]
