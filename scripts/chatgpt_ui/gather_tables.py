#!/usr/bin/env python3
"""Convert ChatGPT UI markdown tables and CSV files into CSVs for GraphRAG."""
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


# ---------------- CSV support ----------------

def _first_key(d: dict[str, str], *candidates: str) -> str:
    for k in candidates:
        if k in d:
            return k
    return ""


def collect_from_csv(paths: list[Path]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    entities: list[dict[str, str]] = []
    relationships: list[dict[str, str]] = []
    claims: list[dict[str, str]] = []

    for p in paths:
        if p.suffix.lower() != ".csv" or not p.is_file():
            continue
        with p.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                continue
            # normalise header keys once per file (lower + strip)
            field_map = {h: re.sub(r"\s+", " ", h.strip().lower()) for h in reader.fieldnames}
            inverse = {v: k for k, v in field_map.items()}

            def has_all(*need: str) -> bool:
                return all(n in inverse for n in need)

            # Try to classify this CSV by available columns.
            # 1) Entities
            ent_name = next((inverse[k] for k in ("entity_name", "name") if k in inverse), None)
            ent_type = next((inverse[k] for k in ("type", "entity_type") if k in inverse), None)
            ent_summary = next((inverse[k] for k in ("summary", "description", "description_list") if k in inverse), None)

            # 2) Relationships
            rel_src = next((inverse[k] for k in ("source", "subject", "from") if k in inverse), None)
            rel_dst = next((inverse[k] for k in ("target", "object", "to") if k in inverse), None)
            rel_summary = next((inverse[k] for k in ("summary", "description") if k in inverse), None)
            rel_strength = next((inverse[k] for k in ("strength", "weight", "confidence") if k in inverse), None)
            rel_type = next((inverse[k] for k in ("type", "predicate", "relationship") if k in inverse), None)

            # 3) Claims
            cl_subj = inverse.get("subject")
            cl_obj = inverse.get("object")
            cl_cat = next((inverse[k] for k in ("category", "claimtype") if k in inverse), None)
            cl_status = inverse.get("status")
            cl_desc = next((inverse[k] for k in ("description", "summary") if k in inverse), None)
            cl_evid = next((inverse[k] for k in ("evidence", "quote", "source text") if k in inverse), None)
            cl_src_file = next((inverse[k] for k in ("source", "provenance", "binder") if k in inverse), None)
            cl_start = next((inverse[k] for k in ("start", "date start") if k in inverse), None)
            cl_end = next((inverse[k] for k in ("end", "date end") if k in inverse), None)

            # Read rows once and dispatch
            for row in reader:
                # Entities row?
                if ent_name:
                    name = (row.get(ent_name) or "").strip()
                    if name:
                        entities.append({
                            "name": name,
                            "type": (row.get(ent_type) or "").strip() if ent_type else "",
                            "summary": (
                                (row.get(ent_summary) or "").strip()
                                if ent_summary else ""
                            ),
                        })
                        # continue checking others — same CSV may contain mixed columns

                # Relationships row?
                if rel_src and rel_dst:
                    src = (row.get(rel_src) or "").strip()
                    dst = (row.get(rel_dst) or "").strip()
                    if src and dst:
                        relationships.append({
                            "source": src,
                            "target": dst,
                            "summary": (row.get(rel_summary) or "").strip() if rel_summary else "",
                            "strength": (row.get(rel_strength) or "").strip() if rel_strength else "",
                            "type": (row.get(rel_type) or "").strip() if rel_type else "",
                        })

                # Claims row?
                if cl_subj and cl_obj and cl_cat and cl_status and cl_desc and cl_evid:
                    subj = (row.get(cl_subj) or "").strip()
                    obj = (row.get(cl_obj) or "").strip()
                    if subj and obj:
                        claims.append({
                            "subject": subj,
                            "object": obj,
                            "category": (row.get(cl_cat) or "").strip(),
                            "status": (row.get(cl_status) or "").strip(),
                            "description": (row.get(cl_desc) or "").strip(),
                            "evidence": (row.get(cl_evid) or "").strip(),
                            "source_file": (row.get(cl_src_file) or "").strip() if cl_src_file else "",
                            "start_date": (row.get(cl_start) or "").strip() if cl_start else "",
                            "end_date": (row.get(cl_end) or "").strip() if cl_end else "",
                        })

    return entities, relationships, claims


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory OR single file path containing .md/.txt/.markdown and/or .csv")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"--input-dir does not exist: {args.input_dir}")

    # Collect candidate paths
    if args.input_dir.is_dir():
        all_paths = sorted(args.input_dir.glob("**/*"))
    else:
        # allow single file
        all_paths = [args.input_dir]

    # 1) Markdown / text inputs
    texts: list[str] = []
    for path in all_paths:
        if path.is_dir():
            continue
        if path.suffix.lower() in {".txt", ".md", ".markdown"}:
            texts.append(path.read_text(encoding="utf-8"))

    md_entities, md_relationships, md_claims = collect_tables(texts)

    # 2) CSV inputs
    csv_entities, csv_relationships, csv_claims = collect_from_csv(all_paths)

    # Merge
    entities = md_entities + csv_entities
    relationships = md_relationships + csv_relationships
    claims = md_claims + csv_claims

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
