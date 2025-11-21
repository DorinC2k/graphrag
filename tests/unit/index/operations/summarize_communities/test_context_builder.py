"""Tests for community report context building."""

import pytest

# Some of the context builder's dependencies (e.g., Azure vector store clients) are optional
# and may not be installed in the test environment. If the Azure SDK is missing, skip the
# module so the import dependency chain does not fail before the test can run.
pytest.importorskip("azure")

import pandas as pd

import graphrag.data_model.schemas as schemas
from graphrag.index.operations.summarize_communities.text_unit_context.context_builder import (
    build_level_context,
)


def test_build_level_context_filters_valid_records_when_reports_exist():
    """When reports already exist, valid contexts should be returned."""

    local_context_df = pd.DataFrame(
        {
            schemas.COMMUNITY_ID: [1, 2],
            schemas.COMMUNITY_LEVEL: [0, 0],
            schemas.CONTEXT_EXCEED_FLAG: [False, False],
            schemas.ALL_CONTEXT: [[], []],
            schemas.CONTEXT_STRING: ["context-a", "context-b"],
            schemas.CONTEXT_SIZE: [10, 12],
        }
    )

    report_df = pd.DataFrame({schemas.COMMUNITY_ID: [99], schemas.COMMUNITY_LEVEL: [0]})

    context = build_level_context(
        report_df=report_df,
        community_hierarchy_df=pd.DataFrame(
            columns=[schemas.COMMUNITY_ID, schemas.SUB_COMMUNITY, schemas.COMMUNITY_LEVEL]
        ),
        local_context_df=local_context_df,
        level=0,
        max_context_tokens=50,
    )

    pd.testing.assert_frame_equal(
        context.reset_index(drop=True), local_context_df.reset_index(drop=True)
    )
