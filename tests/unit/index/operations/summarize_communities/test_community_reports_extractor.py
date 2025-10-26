"""Tests for the community reports extractor helpers."""

import json
from pydantic import BaseModel
from pydantic.errors import PydanticUserError

from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportsExtractor,
)


class _FaultyModel(BaseModel):
    """Model whose modern serializer raises a user error."""

    value: str | None = None

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        raise PydanticUserError("boom", code="test")


class _TotallyFaultyModel(_FaultyModel):
    """Model that raises for both model_dump and dict."""

    def dict(self, *args, **kwargs):  # type: ignore[override]
        raise TypeError("nope")


def test_model_dump_falls_back_to_dict_when_model_dump_fails():
    model = _FaultyModel(value="example")

    dumped = CommunityReportsExtractor._model_dump(model)

    assert dumped == {"value": "example"}


def test_model_dump_handles_total_failure():
    model = _TotallyFaultyModel(value="ignored")

    dumped = CommunityReportsExtractor._model_dump(model)

    assert dumped == {"value": "ignored"}


def test_model_dump_json_handles_errors():
    model = _TotallyFaultyModel(value="ignored")

    dumped = CommunityReportsExtractor._model_dump_json(model)

    assert json.loads(dumped) == {"value": "ignored"}
