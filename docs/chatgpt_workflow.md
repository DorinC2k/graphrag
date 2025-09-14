# Build a GraphRAG index with the ChatGPT UI

The scripts in `scripts/` help you construct a GraphRAG graph when you cannot
call the OpenAI API directly.  They allow you to preprocess documents, run
prompts manually in the ChatGPT UI and then merge the results into the parquet
files expected by GraphRAG.

## 1. Chunk documents into binders

```bash
python scripts/chatgpt_prepare_docs.py \
    --source local --input-dir ./corpus --out-dir ./binders
```

* `--source` may be `local` or `azure`.
* The Azure mode reads text blobs from the container specified with
  `--container` (connection string via `AZURE_STORAGE_CONNECTION_STRING`).
* The script writes `binder_XXXX` folders, each containing at most 20 files and
  ~2M tokens, plus a `text_units.parquet` table.

## 2. Run ChatGPT prompts
Upload the text files inside a binder to the ChatGPT UI and execute the
GraphRAG prompts in the order:

1. `GRAPH_EXTRACTION_PROMPT` for each chunk.
2. `SUMMARIZE_PROMPT` to consolidate descriptions.
3. (Optional) `EXTRACT_CLAIMS_PROMPT`.

Save the JSONL responses from each binder as `entities.jsonl` and
`relationships.jsonl` in the corresponding binder directory.

## 3. Merge outputs into GraphRAG format

```bash
python scripts/merge_chatgpt_graph.py --binder-root ./binders --out-dir ./graph
```

This command combines the JSONL files from all binders and produces
`entities.parquet`, `relationships.parquet`, and copies `text_units.parquet`
into the `graph` directory.  You can then continue with GraphRAG's indexing
workflow using `graphrag index`.
