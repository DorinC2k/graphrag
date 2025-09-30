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

### Prompt order (run inside ChatGPT UI)

1. **Graph extraction** – copy the system, user, continue, and loop prompts from
   `graphrag/prompts/index/extract_graph.py`. Run this immediately after
   uploading each binder so you produce the tuple-formatted entity and
   relationship records (`("entity"|||...)`, `("relationship"|||...)`).
2. **Description summarisation** – once you have entity/relationship tables,
   switch to the prompt text in `graphrag/prompts/index/summarize_descriptions.py`
   to collapse each description list into a concise summary.
3. **Optional claim extraction** – when you need covariate/claim tables, use the
   template in `graphrag/prompts/index/extract_claims.py` against the relevant
   binder segments (same delimiters as graph extraction).
4. **Community reporting** – after summarisation (and optional claims), run the
   prompt from `graphrag/prompts/index/community_report.py` to generate the
   graph-based reports. If you are producing text-unit reports instead, use
   `graphrag/prompts/index/community_report_text_units.py`.

Each prompt file contains the full system/user messages. Copy them verbatim so
the downstream scripts can parse the outputs without changes.

### Typical workflow

1. **Pack documents**
   ```bash
   python scripts/chatgpt_ui/pack_and_manifest.py \
     --src /path/to/corpus \
     --out /path/to/binders \
     --binder-bytes 9500000
   ```
   Upload each binder to ChatGPT and immediately run the **graph extraction**
   prompt from `extract_graph.py`.

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
   After deduplicating the raw entities/relationships, open a new ChatGPT chat,
   paste the prompt from `summarize_descriptions.py`, and provide the description
   lists you want collapsed. Save each ChatGPT response (entities,
   relationships, optional claims) to a plain-text file and run:

   ```bash
   python scripts/chatgpt_ui/gather_tables.py \
     --input-dir ./chatgpt_tables \
     --out-dir ./prepared_tables
   ```

   The script reads every `.txt` / `.md` file in `--input-dir`, detects
   markdown-formatted tables, and writes the combined
   `entities_summarized.csv`, `relationships_summarized.csv`, and (if present)
   `claims.csv` into `--out-dir`.

4. **Optional claim extraction**
   If you captured binder excerpts for claim mining, feed them through the
   `extract_claims.py` prompt (reuse the same tuple delimiters), then parse the
   output with `graph_records_to_tables.py` or copy the resulting markdown table
   into `gather_tables.py` alongside the other summaries.

5. **Generate community reports (via ChatGPT UI)**
   With summarised entities/relationships (and any claims), open a new chat and
   paste the prompt from `community_report.py` to create the graph-based JSON
   reports. If you are using text-unit contexts, switch to
   `community_report_text_units.py` instead. Save each response for downstream
   analysis.

6. **Export to Neo4j**
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
