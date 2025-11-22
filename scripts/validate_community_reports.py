"""Validate community and community report parquet outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from graphrag.data_model.schemas import (
    COMMUNITIES_FINAL_COLUMNS,
    COMMUNITY_REPORTS_FINAL_COLUMNS,
)


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        msg = f"Parquet file not found: {path}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * 80)


def _normalize_ids(values: Iterable) -> set[str]:
    return {str(v) for v in values if pd.notna(v)}


def describe_dataframes(
    communities: pd.DataFrame, community_reports: pd.DataFrame
) -> int:
    """Return an exit code after printing dataset diagnostics."""
    exit_code = 0

    _print_header("Community reports overview")
    print(f"Rows: {len(community_reports)}")
    print(f"Columns: {sorted(community_reports.columns.tolist())}")
    if community_reports.empty:
        print("WARNING: community_reports is empty")
        exit_code = 1

    report_missing_columns = set(COMMUNITY_REPORTS_FINAL_COLUMNS).difference(
        community_reports.columns
    )
    if report_missing_columns:
        print(f"WARNING: missing expected report columns: {sorted(report_missing_columns)}")
        exit_code = 1

    _print_header("Communities overview")
    print(f"Rows: {len(communities)}")
    print(f"Columns: {sorted(communities.columns.tolist())}")
    community_missing_columns = set(COMMUNITIES_FINAL_COLUMNS).difference(
        communities.columns
    )
    if community_missing_columns:
        print(
            f"WARNING: missing expected community columns: {sorted(community_missing_columns)}"
        )
        exit_code = 1

    _print_header("Coverage checks")
    community_ids = _normalize_ids(communities.get("community", []))
    report_ids = _normalize_ids(community_reports.get("community", []))

    missing_reports = sorted(community_ids.difference(report_ids), key=lambda v: int(v))
    if missing_reports:
        print(f"Missing reports for communities: {missing_reports}")
        exit_code = 1
    else:
        print("All communities have reports.")

    extra_reports = sorted(report_ids.difference(community_ids), key=lambda v: int(v))
    if extra_reports:
        print(f"Reports found without matching communities: {extra_reports}")
        exit_code = 1

    duplicate_report_ids = community_reports["community"].astype(str).duplicated(keep=False)
    if duplicate_report_ids.any():
        duplicates = sorted(
            community_reports.loc[duplicate_report_ids, "community"].astype(str).unique(),
            key=lambda v: int(v),
        )
        print(f"Duplicate community report entries found for IDs: {duplicates}")
        exit_code = 1

    if not community_reports.empty:
        _print_header("Sample community reports (first 3 rows)")
        print(community_reports.head(3).to_markdown(index=False))

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate community and community_report parquet files produced by the indexing pipeline."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory containing the parquet outputs (defaults to current working directory).",
    )
    parser.add_argument(
        "--communities",
        type=Path,
        default=None,
        help="Explicit path to communities parquet (overrides --root).",
    )
    parser.add_argument(
        "--community-reports",
        dest="community_reports",
        type=Path,
        default=None,
        help="Explicit path to community_reports parquet (overrides --root).",
    )
    args = parser.parse_args()

    communities_path = (
        args.communities
        if args.communities is not None
        else args.root / "output" / "communities.parquet"
    )
    community_reports_path = (
        args.community_reports
        if args.community_reports is not None
        else args.root / "output" / "community_reports.parquet"
    )

    communities = _load_parquet(communities_path)
    community_reports = _load_parquet(community_reports_path)

    return describe_dataframes(communities, community_reports)


if __name__ == "__main__":
    raise SystemExit(main())
