# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to transform final entities."""

import logging
from uuid import uuid4

import pandas as pd

from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS

log = logging.getLogger(__name__)


def finalize_community_reports(
    reports: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    """All the steps to transform final community reports."""
    if reports.empty:
        if communities.empty:
            log.warning(
                "No communities were available for summarization; generated an empty community_reports table."
            )
        else:
            log.warning(
                "Summarization did not return any community reports for %s communities; generated an empty community_reports table.",
                len(communities),
            )
        return pd.DataFrame(columns=COMMUNITY_REPORTS_FINAL_COLUMNS)

    # Merge with communities to add shared fields
    community_reports = reports.merge(
        communities.loc[:, ["community", "parent", "children", "size", "period"]],
        on="community",
        how="left",
        copy=False,
    )

    community_reports["community"] = community_reports["community"].astype(int)
    community_reports["human_readable_id"] = community_reports["community"]
    community_reports["id"] = [uuid4().hex for _ in range(len(community_reports))]

    return community_reports.loc[
        :,
        COMMUNITY_REPORTS_FINAL_COLUMNS,
    ]
