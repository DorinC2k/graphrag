#!/usr/bin/env python3
"""Convert ChatGPT UI markdown tables into CSV files for GraphRAG."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Iterator


def normalise_header(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def iter_table_blocks(text: str) -> Iterator[list[str]]:
    lines = text.splitlines()
    block: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("```"):
            # ignore code fences entirely
            continue
        is_table_line = line.startswith("|")
        if is_table_line:
            block.append(line)
            continue
        if block:
            yield block
            block = []
    if block:
        yield block


def parse_markdown_table(block: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    if len(block) < 2:
        raise ValueError("table block is too small")

    cleaned = [line.strip().strip("|") for line in block if line.strip()]
    if len(cleaned) < 2:
        raise ValueError("table block has no content")

    header_cells = [cell.strip() for cell in cleaned[0].split("|")]
    rows: list[dict[str, str]] = []

    for line in cleaned[1:]:
        divider = line.replace("-", "").replace(":", "").strip()
        if not divider:
            # alignment row like --- or :---:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < len(header_cells):
            cells.extend([""] * (len(header_cells) - len(cells)))
        row = {header_cells[i]: cells[i] for i in range(len(header_cells))}
        rows.append(row)

    return header_cells, rows


def find_column(headers: list[str], *keywords: str) -> str | None:
    for header in headers:
        normalised = normalise_header(header)
        for keyword in keywords:
            if keyword in normalised.split():
                return header
            if keyword in normalised.replace(" ", ""):
                return header
    return None


def is_entities_table(headers: list[str]) -> bool:
    name = find_column(headers, "entity", "name")
    typ = find_column(headers, "type")
    summary = find_column(headers, "summary", "description")
    return bool(name and typ and summary)


def is_relationships_table(headers: list[str]) -> bool:
    src = find_column(headers, "source", "subject", "from")
    dst = find_column(headers, "target", "object", "to")
    summary = find_column(headers, "summary", "description")
    return bool(src and dst and summary)


def is_claims_table(headers: list[str]) -> bool:
    subj = find_column(headers, "subject")
    obj = find_column(headers, "object")
    category = find_column(headers, "category", "claimtype")
    status = find_column(headers, "status")
    description = find_column(headers, "description", "summary")
    evidence = find_column(headers, "evidence", "quote", "source text")
    return bool(subj and obj and category and status and description and evidence)


def collect_tables(texts: Iterable[str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    entities: list[dict[str, str]] = []
    relationships: list[dict[str, str]] = []
    claims: list[dict[str, str]] = []

    for text in texts:
        for block in iter_table_blocks(text):
            try:
                headers, rows = parse_markdown_table(block)
            except ValueError:
                continue
            if not rows:
                continue

            if is_entities_table(headers):
                name_col = find_column(headers, "entity", "name")
                type_col = find_column(headers, "type")
                summary_col = find_column(headers, "summary", "description")
                for row in rows:
                    name = row.get(name_col or "", "").strip()
                    summary = row.get(summary_col or "", "").strip()
                    if not name and not summary:
                        continue
                    entities.append(
                        {
                            "name": name,
                            "type": row.get(type_col or "", "").strip(),
                            "summary": summary,
                        }
                    )
                continue

            if is_relationships_table(headers):
                source_col = find_column(headers, "source", "subject", "from")
                target_col = find_column(headers, "target", "object", "to")
                summary_col = find_column(headers, "summary", "description")
                strength_col = find_column(headers, "strength", "weight", "confidence")
                type_col = find_column(headers, "type", "predicate", "relationship")
                for row in rows:
                    source = row.get(source_col or "", "").strip()
                    target = row.get(target_col or "", "").strip()
                    if not source or not target:
                        continue
                    relationships.append(
                        {
                            "source": source,
                            "target": target,
                            "summary": row.get(summary_col or "", "").strip(),
                            "strength": row.get(strength_col or "", "").strip(),
                            "type": row.get(type_col or "", "").strip(),
                        }
                    )
                continue

            if is_claims_table(headers):
                subject_col = find_column(headers, "subject")
                object_col = find_column(headers, "object")
                category_col = find_column(headers, "category", "claimtype")
                status_col = find_column(headers, "status")
                description_col = find_column(headers, "description", "summary")
                evidence_col = find_column(headers, "evidence", "quote", "source text")
                source_col = find_column(headers, "source", "provenance", "binder")
                start_col = find_column(headers, "start", "date start")
                end_col = find_column(headers, "end", "date end")
                for row in rows:
                    subject = row.get(subject_col or "", "").strip()
                    obj = row.get(object_col or "", "").strip()
                    if not subject or not obj:
                        continue
                    claims.append(
                        {
                            "subject": subject,
                            "object": obj,
                            "category": row.get(category_col or "", "").strip(),
                            "status": row.get(status_col or "", "").strip(),
                            "description": row.get(description_col or "", "").strip(),
                            "evidence": row.get(evidence_col or "", "").strip(),
                            "source_file": row.get(source_col or "", "").strip(),
                            "start_date": row.get(start_col or "", "").strip(),
                            "end_date": row.get(end_col or "", "").strip(),
                        }
                    )
                continue

    return entities, relationships, claims


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    texts: list[str] = []
    for path in sorted(args.input_dir.glob("**/*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
            continue
        texts.append(path.read_text(encoding="utf-8"))

    entities, relationships, claims = collect_tables(texts)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if entities:
        write_csv(args.out_dir / "entities_summarized.csv", ["name", "type", "summary"], entities)
    if relationships:
        write_csv(
            args.out_dir / "relationships_summarized.csv",
            ["source", "target", "summary", "strength", "type"],
            relationships,
        )
    if claims:
        write_csv(
            args.out_dir / "claims.csv",
            [
                "subject",
                "object",
                "category",
                "status",
                "description",
                "evidence",
                "source_file",
                "start_date",
                "end_date",
            ],
            claims,
        )


if __name__ == "__main__":
    main()

