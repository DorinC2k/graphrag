#!/usr/bin/env python3
"""Convert GraphRAG tuple output into entity and relationship tables."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

Record = dict[str, str]


def parse_records(
    text: str,
    record_delimiter: str,
    tuple_delimiter: str,
    completion_delimiter: str,
) -> Iterator[tuple[str, Sequence[str]]]:
    """Yield (kind, fields) pairs from the raw tuple output."""

    text = text.replace(completion_delimiter, "")
    parts = text.split(record_delimiter)
    pattern = re.compile(
        r"^\(\"(entity|relationship|claim)\""
        + re.escape(tuple_delimiter)
        + r"(.+)\)$",
        re.DOTALL,
    )

    for block in parts:
        block = block.strip()
        if not block:
            continue
        match = pattern.match(block)
        if not match:
            continue
        kind = match.group(1)
        payload = match.group(2)
        fields = [field.strip() for field in payload.split(tuple_delimiter)]
        yield kind, fields


def accumulate_entities(records: Iterable[Sequence[str]]) -> list[Record]:
    aggregated: dict[str, dict[str, object]] = {}
    for fields in records:
        if len(fields) < 3:
            continue
        name, typ, *descriptions = fields
        key = name.strip().upper()
        entry = aggregated.setdefault(
            key,
            {"name": name.strip(), "type": typ.strip(), "descriptions": []},
        )
        description = tuple_delim_join(descriptions)
        if description:
            entry["descriptions"].append(description)
    rows: list[Record] = []
    for entry in aggregated.values():
        for description in entry["descriptions"]:
            rows.append(
                {
                    "name": entry["name"],
                    "type": entry["type"],
                    "description": description,
                }
            )
    return rows


def accumulate_relationships(records: Iterable[Sequence[str]]) -> list[Record]:
    rows: list[Record] = []
    for fields in records:
        if len(fields) < 3:
            continue
        source = fields[0].strip()
        target = fields[1].strip()
        description = fields[2].strip()
        strength = fields[3].strip() if len(fields) > 3 else ""
        rows.append(
            {
                "source": source,
                "target": target,
                "description": description,
                "strength": strength,
            }
        )
    return rows


def accumulate_claims(records: Iterable[Sequence[str]]) -> list[Record]:
    rows: list[Record] = []
    headers = [
        "subject",
        "object",
        "claim_category",
        "status",
        "date",
        "description",
        "evidence",
    ]
    for fields in records:
        data = {header: "" for header in headers}
        for idx, header in enumerate(headers):
            if idx < len(fields):
                data[header] = fields[idx].strip()
        rows.append(data)
    return rows


def tuple_delim_join(parts: Sequence[str]) -> str:
    parts = [part.strip() for part in parts if part.strip()]
    return " ".join(parts)


def write_csv(path: Path, rows: Iterable[Record], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the raw tuple file")
    parser.add_argument("--record-delim", required=True, help="Record delimiter")
    parser.add_argument("--tuple-delim", required=True, help="Tuple delimiter")
    parser.add_argument("--completion", required=True, help="Completion delimiter")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where CSV tables should be written",
    )
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    records = list(
        parse_records(text, args.record_delim, args.tuple_delim, args.completion)
    )

    entities = [fields for kind, fields in records if kind == "entity"]
    relationships = [fields for kind, fields in records if kind == "relationship"]
    claims = [fields for kind, fields in records if kind == "claim"]

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "entities_raw.csv", accumulate_entities(entities), ["name", "type", "description"])
    write_csv(
        out_dir / "relationships_raw.csv",
        accumulate_relationships(relationships),
        ["source", "target", "description", "strength"],
    )

    if claims:
        write_csv(
            out_dir / "claims_raw.csv",
            accumulate_claims(claims),
            [
                "subject",
                "object",
                "claim_category",
                "status",
                "date",
                "description",
                "evidence",
            ],
        )

    summary = {
        "entities": len(entities),
        "relationships": len(relationships),
        "claims": len(claims),
    }
    Path(out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
