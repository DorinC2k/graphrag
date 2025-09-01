#!/usr/bin/env python3
import argparse
import sys
from typing import Optional
import lancedb

try:
    import pandas as pd  # ensure pandas installed
except ImportError:
    print("This script requires pandas. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

# Default LanceDB path
DEFAULT_DB_PATH = r"C:\development\graphrag\tests\fixtures\min-csv\tests\fixtures\min-csv\lancedb"


def choose_table_interactively(db) -> str:
    names = db.table_names()
    if not names:
        print("No tables found in this LanceDB.")
        sys.exit(0)

    print("Tables found:")
    for i, name in enumerate(names, start=1):
        print(f"  [{i}] {name}")

    while True:
        sel = input(f"Select a table (1-{len(names)}): ").strip()
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(names):
                return names[idx - 1]
        print("Invalid selection. Try again.")


def main():
    parser = argparse.ArgumentParser(description="View entries from a LanceDB table.")
    parser.add_argument(
        "--db",
        help=f"Path/URI to LanceDB (default: {DEFAULT_DB_PATH})",
        default=DEFAULT_DB_PATH
    )
    parser.add_argument("--table", help="Table name. If omitted, you'll be prompted to choose.")
    parser.add_argument("--limit", type=int, help="Show only the first N rows (optional).")
    parser.add_argument("--list", action="store_true", help="Just list tables and exit.")
    args = parser.parse_args()

    db = lancedb.connect(args.db)
    names = db.table_names()

    if args.list:
        if not names:
            print("No tables found.")
        else:
            print("Tables:")
            for n in names:
                print(f"- {n}")
        return

    table_name = args.table or choose_table_interactively(db)

    try:
        table = db.open_table(table_name)
    except Exception as e:
        print(f"Error opening table '{table_name}': {e}", file=sys.stderr)
        sys.exit(2)

    df = table.to_pandas()
    if args.limit is not None:
        df = df.head(args.limit)

    # Print metadata
    print(f"\nDB: {args.db}")
    print(f"Table: {table_name}")
    print(f"Rows: {len(df)} (showing {'all' if args.limit is None else args.limit})")
    print(f"Columns: {list(df.columns)}\n")

    # Pretty print DataFrame
    with pd.option_context(
        "display.max_rows", None if args.limit is None else args.limit,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 200
    ):
        print(df)


if __name__ == "__main__":
    main()
