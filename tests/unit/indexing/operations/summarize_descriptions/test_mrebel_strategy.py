import pytest

from graphrag.index.operations.summarize_descriptions.mrebel_strategy import (
    run_huggingface_mrebel_summarization,
)


@pytest.mark.asyncio
async def test_run_huggingface_mrebel_summarization_builds_summary(monkeypatch):
    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def extract_triples(self, text: str, **kwargs):
            self.calls.append(text)
            return [
                ("Alice", "works at", "Contoso"),
                ("Bob", "lives in", "Seattle"),
            ]

    dummy_model = DummyModel()

    def fake_get_model(model_name, **kwargs):
        assert model_name == "dummy-model"
        return dummy_model

    monkeypatch.setattr(
        "graphrag.index.operations.summarize_descriptions.mrebel_strategy.get_or_create_mrebel_model",
        fake_get_model,
    )

    result = await run_huggingface_mrebel_summarization(
        id="Alice",
        descriptions=["Alice works at Contoso", "Bob lives in Seattle"],
        callbacks=None,
        cache=None,
        args={"model_name": "dummy-model", "max_summary_length": 200},
    )

    assert dummy_model.calls == ["Alice works at Contoso", "Bob lives in Seattle"]
    assert "Alice works at Contoso" in result.description
    assert "Bob lives in Seattle" in result.description
