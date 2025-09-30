#!/usr/bin/env python3
"""Convert ChatGPT tuple records into CSV tables for GraphRAG."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def normalise(text: str) -> str:
    return text.strip().strip('"').strip()


def parse_records(
    raw_text: str, record_delim: str, tuple_delim: str, completion: str
) -> Iterable[tuple[str, list[str]]]:
    cleaned = raw_text.replace(completion, "")
    for block in cleaned.split(record_delim):
        block = block.strip()
        if not block:
            continue
        if block.startswith("(") and block.endswith(")"):
            block = block[1:-1]
        parts = [normalise(part) for part in block.split(tuple_delim)]
        if not parts:
            continue
        kind = parts[0].lower()
        yield kind, parts[1:]


def write_entities(records: Iterable[list[str]], path: Path) -> None:
    fieldnames = ["name", "type", "description"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if len(record) < 3:
                continue
            name, typ, *desc_parts = record
            description = "|||".join(desc_parts).strip()
            writer.writerow({"name": name, "type": typ, "description": description})


def write_relationships(records: Iterable[list[str]], path: Path) -> None:
    fieldnames = ["source", "target", "description", "strength"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if len(record) < 3:
                continue
            source = record[0]
            target = record[1]
            description = record[2]
            strength = record[3] if len(record) > 3 else ""
            writer.writerow(
                {
                    "source": source,
                    "target": target,
                    "description": description,
                    "strength": strength,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--record-delim", required=True)
    parser.add_argument("--tuple-delim", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    raw_text = args.input.read_text(encoding="utf-8")
    entity_records: list[list[str]] = []
    relationship_records: list[list[str]] = []

    for kind, parts in parse_records(raw_text, args.record_delim, args.tuple_delim, args.completion):
        if kind == "entity":
            entity_records.append(parts)
        elif kind == "relationship":
            relationship_records.append(parts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_entities(entity_records, args.out_dir / "entities_raw.csv")
    write_relationships(relationship_records, args.out_dir / "relationships_raw.csv")


if __name__ == "__main__":
    main()
