import pytest

from graphrag.index.operations.extract_graph.mrebel_strategy import (
    _parse_mrebel_output,
    run_huggingface_mrebel,
)
from graphrag.index.operations.extract_graph.typing import Document


def test_parse_mrebel_output_handles_basic_triplets():
    text = """
    <triplet> Barack Obama <subj> Barack Obama <rel> position held <obj> President of the United States
    <triplet> Michelle Obama <subject> Michelle Obama <relation> spouse <object> Barack Obama
    """.strip()

    triples = _parse_mrebel_output(text)

    assert triples == [
        ("Barack Obama", "position held", "President of the United States"),
        ("Michelle Obama", "spouse", "Barack Obama"),
    ]


@pytest.mark.asyncio
async def test_run_huggingface_mrebel_builds_expected_graph(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def extract_triples(self, text: str, **kwargs):
            self.calls.append((text, kwargs))
            return [
                ("Alice", "knows", "Bob"),
                ("Alice", "works at", "Contoso"),
            ]

    dummy_model = DummyModel()

    def fake_get_model(model_name, **kwargs):
        assert model_name == "dummy-model"
        return dummy_model

    monkeypatch.setattr(
        "graphrag.index.operations.extract_graph.mrebel_strategy._get_or_create_model",
        fake_get_model,
    )

    docs = [Document(text="Alice works with Bob at Contoso.", id="doc-1")]

    result = await run_huggingface_mrebel(
        docs,
        entity_types=["person"],
        callbacks=None,
        cache=None,
        args={"model_name": "dummy-model", "relation_weight": 2.0},
    )

    assert {entity["title"] for entity in result.entities} == {
        "Alice",
        "Bob",
        "Contoso",
    }
    assert all(entity["source_id"] == "doc-1" for entity in result.entities)

    relationship_map = {
        (rel["source"], rel["target"]): rel for rel in result.relationships
    }

    assert relationship_map[("Alice", "Bob")]["description"] == "knows"
    assert relationship_map[("Alice", "Bob")]["weight"] == 2.0
    assert relationship_map[("Alice", "Bob")]["source_id"] == "doc-1"
    assert relationship_map[("Alice", "Contoso")]["description"] == "works at"
    assert relationship_map[("Alice", "Contoso")]["weight"] == 2.0

    assert dummy_model.calls, "The dummy model should be invoked"
