#!/usr/bin/env python3
"""Pack a corpus into ~N MiB plain-text binders for ChatGPT UI ingestion."""
from __future__ import annotations

import argparse
import csv
import html
import io
import re
from pathlib import Path
from typing import Callable, Iterable

TOKENS_PER_CHAR_HEURISTIC = 1 / 4.0  # ≈4 chars per token
SEP = "\n" + ("-" * 80) + "\n"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate tokens for text using tiktoken when available."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        approx_chars = len(re.sub(r"\s+", " ", text))
        return int(approx_chars * TOKENS_PER_CHAR_HEURISTIC)


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

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
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_html(path: Path) -> str:
    return strip_html(read_txt(path))


def read_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(str(path)) or ""
    except Exception:
        try:
            import PyPDF2  # type: ignore

            text_parts: list[str] = []
            with path.open("rb") as handle:
                reader = PyPDF2.PdfReader(handle)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
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

DEFAULT_EXTENSIONS = tuple(READERS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gather_files(root: Path, extensions: Iterable[str], min_bytes: int) -> list[Path]:
    ext_set = {ext.lower() for ext in extensions}
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ext_set:
            continue
        if path.stat().st_size < min_bytes:
            continue
        files.append(path)
    files.sort()
    return files


def to_text(path: Path) -> str:
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        return ""
    text = reader(path)
    if not isinstance(text, str):
        try:
            text = text.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    return text


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def pack_binders(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = args.extensions or DEFAULT_EXTENSIONS
    files = gather_files(src, extensions, args.min_file_bytes)

    by_parent: dict[Path, list[Path]] = {}
    for file_path in files:
        by_parent.setdefault(file_path.parent, []).append(file_path)

    manifest_rows: list[dict[str, object]] = []
    binder_index = 1

    def flush_buffer(buffer: io.StringIO, folder_files: list[Path], start: int, end: int) -> None:
        nonlocal binder_index
        content = buffer.getvalue()
        if not content.strip():
            return
        binder_name = f"binder_{binder_index:05d}.txt"
        binder_path = out_dir / binder_name
        binder_path.write_text(content, encoding="utf-8")
        approx_tokens = estimate_tokens(content)
        manifest_rows.append(
            {
                "binder_path": str(binder_path),
                "size_bytes": binder_path.stat().st_size,
                "approx_tokens": approx_tokens,
                "source_files": ";".join(str(p) for p in folder_files[start:end]),
            }
        )
        binder_index += 1

    for folder, folder_files in by_parent.items():
        buffer = io.StringIO()
        size_bytes = 0
        start_idx = 0

        for idx, file_path in enumerate(folder_files):
            text = to_text(file_path)
            if not text:
                continue

            rel_folder = (
                folder.relative_to(src)
                if src in folder.parents or folder == src
                else folder
            )
            header = (
                f"SOURCE_FILE: {file_path}\n"
                f"RELATIVE_TO: {rel_folder}\n\n"
            )
            chunk = header + text.strip() + SEP
            chunk_bytes = len(chunk.encode("utf-8"))

            if size_bytes + chunk_bytes > args.binder_bytes and buffer.tell() > 0:
                flush_buffer(buffer, folder_files, start_idx, idx)
                buffer = io.StringIO()
                size_bytes = 0
                start_idx = idx

            buffer.write(chunk)
            size_bytes += chunk_bytes

        flush_buffer(buffer, folder_files, start_idx, len(folder_files))

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["binder_path", "size_bytes", "approx_tokens", "source_files"],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Wrote {len(manifest_rows)} binders -> {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Root folder of the corpus")
    parser.add_argument(
        "--out",
        required=True,
        help="Output folder for binder .txt files and manifest.csv",
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
        default=list(DEFAULT_EXTENSIONS),
        help="Extensions to include (default: common text formats)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="pack_and_manifest 1.0",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pack_binders(args)


if __name__ == "__main__":
    main()
