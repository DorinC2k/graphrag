#!/usr/bin/env python3
"""Pack a corpus into ~N MiB binder text files for ChatGPT UI ingestion.

The script walks an input folder, converts supported documents into text, and
writes binder text files in the output directory.  A ``manifest.csv`` file is
also produced describing each binder along with its approximate token count and
source files.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# Rough fallback conversion between characters and tokens.
TOKENS_PER_CHAR_HEURISTIC = 1 / 4.0


@dataclass
class Args:
    src: Path
    out: Path
    binder_bytes: int
    min_file_bytes: int
    extensions: list[str]


def estimate_tokens(text: str) -> int:
    """Estimate token count for ``text`` using tiktoken when available."""
    try:
        import tiktoken  # type: ignore

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        collapsed = re.sub(r"\s+", " ", text)
        return int(len(collapsed) * TOKENS_PER_CHAR_HEURISTIC)


def read_txt(path: Path) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="ignore")
        except Exception:
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


def read_md(path: Path) -> str:
    return read_txt(path)


def strip_html(raw: str) -> str:
    raw = html.unescape(raw)
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def read_html(path: Path) -> str:
    return strip_html(read_txt(path))


def read_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(str(path)) or ""
    except Exception:
        try:
            import PyPDF2  # type: ignore

            text: list[str] = []
            with path.open("rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception:
            return ""


def read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore

        document = docx.Document(str(path))
        return "\n".join(para.text for para in document.paragraphs)
    except Exception:
        return ""


READERS: dict[str, Callable[[Path], str]] = {
    ".txt": read_txt,
    ".md": read_md,
    ".markdown": read_md,
    ".html": read_html,
    ".htm": read_html,
    ".pdf": read_pdf,
    ".docx": read_docx,
}

DEFAULT_EXTS = sorted(READERS)
SEPARATOR = "\n" + ("-" * 80) + "\n"


def gather_files(root: Path, extensions: Iterable[str], min_file_bytes: int) -> list[Path]:
    allowed = {ext.lower() for ext in extensions}
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < min_file_bytes:
            continue
        files.append(path)
    files.sort()
    return files


def to_text(path: Path) -> str:
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        return ""
    text = reader(path)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return text or ""


def flush_buffer(
    out_dir: Path,
    buffer: io.StringIO,
    records: list[Path],
    start: int,
    end: int,
    binder_index: int,
    manifest: list[dict[str, str | int]],
) -> int:
    content = buffer.getvalue()
    if not content.strip():
        return binder_index

    binder_name = f"binder_{binder_index:05d}.txt"
    binder_path = out_dir / binder_name
    binder_path.write_text(content, encoding="utf-8")
    approx_tokens = estimate_tokens(content)
    size_bytes = binder_path.stat().st_size
    manifest.append(
        {
            "binder_path": str(binder_path),
            "size_bytes": size_bytes,
            "approx_tokens": approx_tokens,
            "source_files": ";".join(str(p) for p in records[start:end]),
        }
    )
    return binder_index + 1


def pack(args: Args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    files = gather_files(args.src, args.extensions, args.min_file_bytes)

    by_parent: dict[Path, list[Path]] = {}
    for file in files:
        by_parent.setdefault(file.parent, []).append(file)

    manifest_rows: list[dict[str, str | int]] = []
    binder_index = 1

    for folder, file_list in sorted(by_parent.items()):
        buffer = io.StringIO()
        size_bytes = 0
        start_idx = 0

        for idx, path in enumerate(file_list):
            text = to_text(path)
            if not text.strip():
                continue
            if args.src in path.parents:
                relative_parent = path.parent.relative_to(args.src)
            else:
                relative_parent = path.parent
            header = (
                f"SOURCE_FILE: {path}\nRELATIVE_TO: {relative_parent}\n\n"
            )
            chunk = header + text.strip() + SEPARATOR
            chunk_bytes = len(chunk.encode("utf-8"))
            if size_bytes + chunk_bytes > args.binder_bytes and buffer.tell() > 0:
                binder_index = flush_buffer(
                    args.out,
                    buffer,
                    file_list,
                    start_idx,
                    idx,
                    binder_index,
                    manifest_rows,
                )
                buffer = io.StringIO()
                size_bytes = 0
                start_idx = idx
            buffer.write(chunk)
            size_bytes += chunk_bytes

        binder_index = flush_buffer(
            args.out,
            buffer,
            file_list,
            start_idx,
            len(file_list),
            binder_index,
            manifest_rows,
        )

    manifest_path = args.out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["binder_path", "size_bytes", "approx_tokens", "source_files"],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Wrote {len(manifest_rows)} binders -> {manifest_path}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Root of the corpus")
    parser.add_argument(
        "--out", required=True, type=Path, help="Directory for binder files"
    )
    parser.add_argument(
        "--binder-bytes",
        type=int,
        default=9_500_000,
        help="Target bytes per binder (default ≈9.5 MiB)",
    )
    parser.add_argument(
        "--min-file-bytes",
        type=int,
        default=256,
        help="Skip files smaller than this many bytes",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=DEFAULT_EXTS,
        help="File extensions to include",
    )
    ns = parser.parse_args()
    return Args(
        src=ns.src.expanduser().resolve(),
        out=ns.out.expanduser().resolve(),
        binder_bytes=ns.binder_bytes,
        min_file_bytes=ns.min_file_bytes,
        extensions=[ext if ext.startswith(".") else f".{ext}" for ext in ns.extensions],
    )


def main() -> None:
    args = parse_args()
    pack(args)


if __name__ == "__main__":
    main()
