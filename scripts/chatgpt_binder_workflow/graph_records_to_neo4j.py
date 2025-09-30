#!/usr/bin/env python3
"""Convert GraphRAG tables into Neo4j import CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def norm(value: str) -> str:
    return " ".join(value.strip().lower().split())


def entity_key(name: str, entity_type: str) -> str:
    return f"{norm(entity_type)}:{norm(name)}"


def attrs_fingerprint(attrs: dict) -> str:
    data = json.dumps(attrs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, object]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_nodes(entities: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    node_rows: list[dict[str, object]] = []
    index: dict[str, int] = {}
    next_id = 1

    for entity in entities:
        name = entity.get("name", "")
        entity_type = entity.get("type", "Unknown")
        key = entity_key(name, entity_type)
        if key in index:
            continue
        index[key] = next_id
        node_rows.append(
            {
                "id": next_id,
                "name": name,
                "type": entity_type,
                "key": key,
                "description": entity.get("description", ""),
            }
        )
        next_id += 1

    return node_rows, index


def build_relationships(
    relationships: list[dict[str, str]],
    node_index: dict[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen = set()
    for rel in relationships:
        source_key = entity_key(rel.get("source", ""), rel.get("source_type", ""))
        target_key = entity_key(rel.get("target", ""), rel.get("target_type", ""))

        if source_key not in node_index or target_key not in node_index:
            continue

        attrs = {
            "description": rel.get("description", ""),
            "strength": rel.get("strength", ""),
        }
        fingerprint = (
            node_index[source_key],
            node_index[target_key],
            rel.get("relationship_type", "RELATED"),
            attrs_fingerprint(attrs),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        rows.append(
            {
                "start_id": node_index[source_key],
                "end_id": node_index[target_key],
                "type": rel.get("relationship_type", "RELATED"),
                "attrs_json": json.dumps(attrs, ensure_ascii=False),
            }
        )

    return rows


def build_claims(claims: list[dict[str, str]], node_index: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for claim in claims:
        subject_key = entity_key(claim.get("subject", ""), claim.get("subject_type", ""))
        object_key = entity_key(claim.get("object", ""), claim.get("object_type", ""))
        if subject_key not in node_index or object_key not in node_index:
            continue
        rows.append(
            {
                "subject_id": node_index[subject_key],
                "object_id": node_index[object_key],
                "claim_category": claim.get("claim_category", ""),
                "status": claim.get("status", ""),
                "date": claim.get("date", ""),
                "description": claim.get("description", ""),
                "evidence": claim.get("evidence", ""),
            }
        )
    return rows


def create_cypher(out_dir: Path, nodes_csv: Path, rels_csv: Path, claims_csv: Path | None) -> None:
    statements = [
        "// Load nodes",
        "USING PERIODIC COMMIT 500",
        f"LOAD CSV WITH HEADERS FROM 'file:///{nodes_csv.name}' AS row",
        "MERGE (n:Entity {id: toInteger(row.id)})",
        "  SET n.name = row.name, n.type = row.type, n.key = row.key, n.description = row.description;",
        "",
        "// Load relationships",
        "USING PERIODIC COMMIT 500",
        f"LOAD CSV WITH HEADERS FROM 'file:///{rels_csv.name}' AS row",
        "MATCH (a:Entity {id: toInteger(row.start_id)})",
        "MATCH (b:Entity {id: toInteger(row.end_id)})",
        "WITH a, b, row",
        "CALL apoc.create.relationship(a, row.type, apoc.convert.fromJsonMap(row.attrs_json), b) YIELD rel",
        "RETURN count(rel);",
    ]

    if claims_csv is not None:
        statements.extend(
            [
                "",
                "// Load claims",
                "USING PERIODIC COMMIT 500",
                f"LOAD CSV WITH HEADERS FROM 'file:///{claims_csv.name}' AS row",
                "MATCH (s:Entity {id: toInteger(row.subject_id)})",
                "MATCH (o:Entity {id: toInteger(row.object_id)})",
                "MERGE (s)-[c:CLAIM {category: row.claim_category, status: row.status}]->(o)",
                "SET c.date = row.date, c.description = row.description, c.evidence = row.evidence;",
            ]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "import.cypher").write_text("\n".join(statements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", required=True, help="Summarized entities CSV")
    parser.add_argument("--relationships", required=True, help="Summarized relationships CSV")
    parser.add_argument("--claims", help="Optional claims CSV")
    parser.add_argument("--out", required=True, help="Output directory for Neo4j CSVs")
    args = parser.parse_args()

    entities = load_csv(Path(args.entities))
    relationships = load_csv(Path(args.relationships))
    claims = load_csv(Path(args.claims)) if args.claims else []

    nodes, index = build_nodes(entities)

    for rel in relationships:
        rel.setdefault("source_type", "")
        rel.setdefault("target_type", "")
        rel.setdefault("relationship_type", rel.get("type", "RELATED").upper().replace(" ", "_"))

    rel_rows = build_relationships(relationships, index)

    if claims:
        for claim in claims:
            claim.setdefault("subject_type", "")
            claim.setdefault("object_type", "")
        claim_rows = build_claims(claims, index)
    else:
        claim_rows = []

    out_dir = Path(args.out)
    nodes_csv = out_dir / "nodes.csv"
    rels_csv = out_dir / "relationships.csv"
    claims_csv = out_dir / "claims.csv" if claim_rows else None

    write_csv(nodes_csv, nodes, ["id", "name", "type", "key", "description"])
    write_csv(rels_csv, rel_rows, ["start_id", "end_id", "type", "attrs_json"])
    if claim_rows and claims_csv is not None:
        write_csv(
            claims_csv,
            claim_rows,
            [
                "subject_id",
                "object_id",
                "claim_category",
                "status",
                "date",
                "description",
                "evidence",
            ],
        )

    create_cypher(out_dir, nodes_csv, rels_csv, claims_csv)

    print(
        json.dumps(
            {
                "nodes": len(nodes),
                "relationships": len(rel_rows),
                "claims": len(claim_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
