# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Merge ChatGPT extraction outputs into GraphRAG parquet tables.

After running the prompts in the ChatGPT UI, save the JSONL outputs for each
binder as ``entities.jsonl`` and ``relationships.jsonl``.  This script combines
all binder outputs and produces the ``entities.parquet`` and
``relationships.parquet`` files required by GraphRAG.  It also copies the
``text_units.parquet`` generated during preparation.

Example:
    python scripts/merge_chatgpt_graph.py --binder-root ./binders --out-dir ./graph
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import typer


def _read_jsonl(path: Path) -> list[dict]:
    """Return all JSON records from ``path`` if it exists."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merge(binder_root: Path, out_dir: Path) -> None:
    """Merge JSONL files from all binders into parquet tables."""
    binder_root = Path(binder_root)
    out_dir = Path(out_dir)

    entities: list[pd.DataFrame] = []
    relationships: list[pd.DataFrame] = []

    for binder in sorted(binder_root.glob("binder_*")):
        entities_path = binder / "entities.jsonl"
        relationships_path = binder / "relationships.jsonl"
        e_records = _read_jsonl(entities_path)
        r_records = _read_jsonl(relationships_path)
        if e_records:
            entities.append(pd.DataFrame(e_records))
        if r_records:
            relationships.append(pd.DataFrame(r_records))

    out_dir.mkdir(parents=True, exist_ok=True)
    if entities:
        df_entities = pd.concat(entities, ignore_index=True).drop_duplicates(
            subset="id"
        )
        df_entities.to_parquet(out_dir / "entities.parquet", index=False)
    if relationships:
        df_rel = pd.concat(relationships, ignore_index=True).drop_duplicates()
        df_rel.to_parquet(out_dir / "relationships.parquet", index=False)

    src_text_units = binder_root / "text_units.parquet"
    if src_text_units.exists():
        shutil.copy(src_text_units, out_dir / "text_units.parquet")


app = typer.Typer(help="Merge ChatGPT graph extraction outputs")


@app.command()
def main(
    binder_root: Path = typer.Option(..., help="Directory containing binder_* folders"),
    out_dir: Path = typer.Option(..., help="Where to place the output parquet files"),
) -> None:
    """CLI wrapper for :func:`merge`."""
    merge(binder_root, out_dir)


if __name__ == "__main__":
    app()
