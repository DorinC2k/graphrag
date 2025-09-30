#!/usr/bin/env python3
"""Convert ChatGPT UI extraction tables into Neo4j import CSVs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


@dataclass
class Node:
    id: int
    name: str
    type: str
    summary: str


@dataclass
class Relationship:
    start_id: int
    end_id: int
    type: str
    summary: str
    strength: str


@dataclass
class Claim:
    id: int
    subject_id: int
    object_id: int
    category: str
    status: str
    description: str
    evidence: str
    source: str


def norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned: dict[str, str] = {}
            for key, value in row.items():
                clean_key = (key or "").strip()
                clean_value = value.strip() if isinstance(value, str) else ""
                cleaned[clean_key] = clean_value
            rows.append(cleaned)
        return rows


def node_key(name: str, typ: str) -> str:
    return f"{norm(typ)}::{norm(name)}"


def upsert_node(
    nodes: Dict[str, Node],
    name: str,
    typ: str,
    summary: str,
    next_id: int,
) -> tuple[int, int]:
    key = node_key(name, typ or "Unknown")
    if key in nodes:
        node = nodes[key]
        if summary and summary not in node.summary:
            parts = [node.summary, summary]
            node.summary = "; ".join(part for part in parts if part)
        return node.id, next_id
    node = Node(id=next_id, name=name, type=typ or "Unknown", summary=summary)
    nodes[key] = node
    return node.id, next_id + 1


def normalise_relationship_type(value: str) -> str:
    if not value:
        return "RELATED_TO"
    return "_".join(part for part in value.upper().split()) or "RELATED_TO"


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_graph(
    entities_csv: Path,
    relationships_csv: Path,
    claims_csv: Path | None,
    out_dir: Path,
) -> None:
    nodes: Dict[str, Node] = {}
    relationships: list[Relationship] = []
    claims: list[Claim] = []
    next_node_id = 1
    next_claim_id = 1

    # Entities
    for row in load_csv(entities_csv):
        name = row.get("name", "").strip()
        typ = row.get("type", "Unknown").strip()
        summary = row.get("summary", "").strip() or row.get("description", "").strip()
        if not name:
            continue
        _, next_node_id = upsert_node(nodes, name, typ, summary, next_node_id)

    # Relationships
    for row in load_csv(relationships_csv):
        source = row.get("source", "").strip()
        target = row.get("target", "").strip()
        if not source or not target:
            continue
        summary = row.get("summary", "").strip() or row.get("description", "").strip()
        strength = row.get("strength", "").strip() or row.get("weight", "").strip()
        rel_type = (
            row.get("type")
            or row.get("relationship_type")
            or row.get("predicate")
            or "RELATED_TO"
        )
        source_type = row.get("source_type", "Unknown")
        target_type = row.get("target_type", "Unknown")
        source_summary = row.get("source_summary", "").strip()
        target_summary = row.get("target_summary", "").strip()
        source_id, next_node_id = upsert_node(
            nodes, source, source_type, source_summary, next_node_id
        )
        target_id, next_node_id = upsert_node(
            nodes, target, target_type, target_summary, next_node_id
        )
        relationships.append(
            Relationship(
                start_id=source_id,
                end_id=target_id,
                type=normalise_relationship_type(rel_type),
                summary=summary,
                strength=strength,
            )
        )

    # Claims
    if claims_csv:
        for row in load_csv(claims_csv):
            subject = row.get("subject", "").strip() or row.get("source", "").strip()
            obj = row.get("object", "").strip() or row.get("target", "").strip()
            if not subject or not obj:
                continue
            subject_type = row.get("subject_type", "Unknown")
            object_type = row.get("object_type", "Unknown")
            subject_summary = row.get("subject_summary", "").strip()
            object_summary = row.get("object_summary", "").strip()
            subject_id, next_node_id = upsert_node(
                nodes, subject, subject_type, subject_summary, next_node_id
            )
            object_id, next_node_id = upsert_node(
                nodes, obj, object_type, object_summary, next_node_id
            )
            claims.append(
                Claim(
                    id=next_claim_id,
                    subject_id=subject_id,
                    object_id=object_id,
                    category=row.get("category", "").strip(),
                    status=row.get("status", "").strip(),
                    description=row.get("description", "").strip()
                    or row.get("summary", "").strip(),
                    evidence=row.get("evidence", "").strip()
                    or row.get("quote", "").strip(),
                    source=row.get("source_file", "").strip(),
                )
            )
            next_claim_id += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    node_rows = [
        {
            "id": str(node.id),
            "name": node.name,
            "type": node.type,
            "summary": node.summary,
        }
        for node in sorted(nodes.values(), key=lambda n: n.id)
    ]

    rel_seen: set[tuple[int, int, str, str, str]] = set()
    rel_rows: list[dict[str, str]] = []
    for rel in relationships:
        fingerprint = (
            rel.start_id,
            rel.end_id,
            rel.type,
            rel.summary,
            rel.strength,
        )
        if fingerprint in rel_seen:
            continue
        rel_seen.add(fingerprint)
        rel_rows.append(
            {
                "start_id": str(rel.start_id),
                "end_id": str(rel.end_id),
                "type": rel.type,
                "summary": rel.summary,
                "strength": rel.strength,
            }
        )

    claim_rows = [
        {
            "id": str(claim.id),
            "subject_id": str(claim.subject_id),
            "object_id": str(claim.object_id),
            "category": claim.category,
            "status": claim.status,
            "description": claim.description,
            "evidence": claim.evidence,
            "source": claim.source,
        }
        for claim in claims
    ]

    write_csv(out_dir / "nodes.csv", ["id", "name", "type", "summary"], node_rows)
    write_csv(
        out_dir / "relationships.csv",
        ["start_id", "end_id", "type", "summary", "strength"],
        rel_rows,
    )
    if claim_rows:
        write_csv(
            out_dir / "claims.csv",
            [
                "id",
                "subject_id",
                "object_id",
                "category",
                "status",
                "description",
                "evidence",
                "source",
            ],
            claim_rows,
        )

    cypher = f"""
// Load nodes
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///{(out_dir / 'nodes.csv').name}' AS row
MERGE (n:Entity {{id: toInteger(row.id)}})
  SET n.name = row.name, n.type = row.type, n.summary = row.summary;

// Load relationships
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///{(out_dir / 'relationships.csv').name}' AS row
MATCH (a:Entity {{id: toInteger(row.start_id)}})
MATCH (b:Entity {{id: toInteger(row.end_id)}})
CALL apoc.create.relationship(
  a,
  row.type,
  {{summary: row.summary, strength: row.strength}},
  b
) YIELD rel
RETURN count(rel);
"""

    if claim_rows:
        cypher += f"""
// Load claims
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///{(out_dir / 'claims.csv').name}' AS row
MATCH (a:Entity {{id: toInteger(row.subject_id)}})
MATCH (b:Entity {{id: toInteger(row.object_id)}})
MERGE (c:Claim {{id: toInteger(row.id)}})
  SET c.category = row.category,
      c.status = row.status,
      c.description = row.description,
      c.evidence = row.evidence,
      c.source = row.source;
MERGE (a)-[:ASSERTS]->(c)
MERGE (c)-[:ABOUT]->(b);
"""

    (out_dir / "import.cypher").write_text(cypher.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", required=True, type=Path)
    parser.add_argument("--relationships", required=True, type=Path)
    parser.add_argument("--claims", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    build_graph(args.entities, args.relationships, args.claims, args.out)


if __name__ == "__main__":
    main()
