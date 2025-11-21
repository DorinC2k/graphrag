"""Tests for the community reports extractor helpers."""

import json
from pydantic import BaseModel
from pydantic.errors import PydanticUserError

from types import SimpleNamespace

from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportResponse,
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


def test_parse_llm_response_uses_json_content_when_parsed_response_missing():
    content = json.dumps(
        {
            "title": "Example",
            "summary": "Summary",
            "findings": [{"summary": "A", "explanation": "B"}],
            "rating": 1,
            "rating_explanation": "Because",
        }
    )
    response = SimpleNamespace(parsed_response=None, output=SimpleNamespace(content=content))

    parsed = CommunityReportsExtractor._parse_llm_response(response)

    assert isinstance(parsed, CommunityReportResponse)
    assert parsed.title == "Example"
    assert parsed.summary == "Summary"
