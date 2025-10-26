"""Tests for finalizing community reports."""

import pandas as pd

from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS
from graphrag.index.operations.finalize_community_reports import (
    finalize_community_reports,
)


def test_finalize_returns_empty_frame_when_no_reports():
    reports = pd.DataFrame()
    communities = pd.DataFrame(
        columns=["community", "parent", "children", "size", "period"]
    )

    finalized = finalize_community_reports(reports, communities)

    assert finalized.empty
    assert list(finalized.columns) == COMMUNITY_REPORTS_FINAL_COLUMNS
