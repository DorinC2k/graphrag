# ChatGPT UI Binder Pipeline

This optional toolkit mirrors the GraphRAG indexing prompts while routing every
LLM interaction through the ChatGPT UI.  It is useful when direct API access is
restricted or when you prefer to drive the workflow manually.

## Contents

The scripts live in `scripts/chatgpt_ui/` and can be mixed with the standard
`chatgpt_prepare_docs.py` preparation flow.

| Script | Purpose |
| --- | --- |
| `pack_and_manifest.py` | Packs source documents into ~9 MiB binder text files and writes a `manifest.csv` listing binder sizes, token estimates, and provenance. |
| `graph_records_to_tables.py` | Parses the tuple-formatted output of the graph extraction prompt into `entities_raw.csv` and `relationships_raw.csv`. |
| `gather_tables.py` | Converts markdown tables copied from ChatGPT UI into `entities_summarized.csv`, `relationships_summarized.csv`, and optional `claims.csv`. |
| `graph_records_to_neo4j.py` | Merges the curated CSV tables (and optional claim tuples) into Neo4j import CSVs plus a helper `import.cypher`. |

## Typical workflow

1. **Pack documents**
   ```bash
   python scripts/chatgpt_ui/pack_and_manifest.py \
     --src /path/to/corpus \
     --out /path/to/binders \
     --binder-bytes 9500000
   ```
   Upload each binder to ChatGPT and run the repository prompt templates.

2. **Convert tuple output**
   ```bash
   python scripts/chatgpt_ui/graph_records_to_tables.py \
     --input binder_00001.graph.txt \
     --record-delim $'\n---\n' \
     --tuple-delim '|||' \
     --completion '<END>' \
     --out-dir ./binder_00001
   ```

3. **Summarise descriptions** (via ChatGPT UI) and gather the resulting tables.
   Save each ChatGPT response (entities, relationships, optional claims) to a
   plain-text file and run:

   ```bash
   python scripts/chatgpt_ui/gather_tables.py \
     --input-dir ./chatgpt_tables \
     --out-dir ./prepared_tables
   ```

   The script reads every `.txt` / `.md` file in `--input-dir`, detects
   markdown-formatted tables, and writes the combined
   `entities_summarized.csv`, `relationships_summarized.csv`, and (if present)
   `claims.csv` into `--out-dir`.

4. **Export to Neo4j**
   ```bash
   python scripts/chatgpt_ui/graph_records_to_neo4j.py \
     --entities entities_summarized.csv \
     --relationships relationships_summarized.csv \
     --claims claims.csv \
     --out neo4j_export
   ```
   Copy the resulting CSVs into Neo4j's `import/` directory and execute the
   generated `import.cypher` script.

## Notes

* The scripts are pure-Python and rely only on the standard library, although
  `pack_and_manifest.py` will opportunistically use `pdfminer.six`, `PyPDF2`, or
  `python-docx` when installed.
* Delimiters must match the ones used in the prompt templates (`|||`, `---`,
  `<END>`, etc.).
* The Neo4j exporter deduplicates nodes and relationships by normalised names and
  types.  Empty fields are ignored gracefully.
