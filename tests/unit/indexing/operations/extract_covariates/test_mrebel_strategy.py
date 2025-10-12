import pytest

from graphrag.index.operations.extract_covariates.mrebel_strategy import (
    run_huggingface_mrebel_claims,
)


@pytest.mark.asyncio
async def test_run_huggingface_mrebel_claims_builds_covariates(monkeypatch):
    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def extract_triples(self, text: str, **kwargs):
            self.calls.append(text)
            return [
                ("Alice", "works at", "Contoso"),
                ("Alice", "knows", "Bob"),
            ]

    dummy_model = DummyModel()

    def fake_get_model(model_name, **kwargs):
        assert model_name == "dummy-model"
        return dummy_model

    monkeypatch.setattr(
        "graphrag.index.operations.extract_covariates.mrebel_strategy.get_or_create_mrebel_model",
        fake_get_model,
    )

    result = await run_huggingface_mrebel_claims(
        input=["Alice works with Bob at Contoso."],
        entity_types=[],
        resolved_entities_map={"Bob": "Robert"},
        callbacks=None,
        cache=None,
        args={"model_name": "dummy-model"},
    )

    assert dummy_model.calls == ["Alice works with Bob at Contoso."]
    assert len(result.covariate_data) == 2
    assert result.covariate_data[1].object_id == "Robert"
