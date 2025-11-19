# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from graphrag.data_model.community import Community
from graphrag.data_model.community_report import CommunityReport
from graphrag.query.context_builder.dynamic_community_selection import (
    DynamicCommunitySelection,
)


def _community(short_id: str, level: str, parent: str, children: list[str]) -> Community:
    return Community(
        id=short_id,
        short_id=short_id,
        title=f"community-{short_id}",
        level=level,
        parent=parent,
        children=children,
    )


def _report(community_id: str) -> CommunityReport:
    return CommunityReport(
        id=f"report-{community_id}",
        short_id=community_id,
        title=f"report-{community_id}",
        community_id=community_id,
        summary=f"summary-{community_id}",
        full_content=f"full-{community_id}",
    )


def test_dynamic_selection_handles_missing_root_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communities = [
        _community(short_id="c1", level="1", parent="", children=["c2"]),
        _community(short_id="c2", level="2", parent="c1", children=[]),
    ]

    selector = DynamicCommunitySelection(
        community_reports=[_report("c1"), _report("c2")],
        communities=communities,
        model=MagicMock(),
        token_encoder=MagicMock(),
        threshold=1,
        max_level=3,
    )

    assert selector.starting_level == 1
    assert selector.starting_communities == ["c1"]

    async def fake_rate_relevancy(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "rating": 2,
            "llm_calls": 1,
            "prompt_tokens": 1,
            "output_tokens": 1,
        }

    monkeypatch.setattr(
        "graphrag.query.context_builder.dynamic_community_selection.rate_relevancy",
        fake_rate_relevancy,
    )

    reports, _ = asyncio.run(selector.select("test query"))

    # Only the deepest relevant community should remain after pruning parents
    assert {report.community_id for report in reports} == {"c2"}
