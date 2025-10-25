# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Print Progress Logger."""

from __future__ import annotations

import sys
from typing import TextIO

from graphrag.logger.base import Progress, ProgressLogger


class PrintProgressLogger(ProgressLogger):
    """A progress logger that prints progress to stdout."""

    prefix: str

    def __init__(self, prefix: str, stream: TextIO | None = None):
        """Create a new progress logger."""
        self.prefix = prefix
        self._stream = stream or sys.stdout
        self._emit(f"\n{self.prefix}", end="")

    def __call__(self, update: Progress) -> None:
        """Update progress."""
        self._emit(".", end="")

    def dispose(self) -> None:
        """Dispose of the progress logger."""

    def child(self, prefix: str, transient: bool = True) -> ProgressLogger:
        """Create a child progress bar."""
        return PrintProgressLogger(prefix, stream=self._stream)

    def stop(self) -> None:
        """Stop the progress logger."""

    def force_refresh(self) -> None:
        """Force a refresh."""

    def error(self, message: str) -> None:
        """Log an error."""
        self._emit(f"\n{self.prefix}ERROR: {message}")

    def warning(self, message: str) -> None:
        """Log a warning."""
        self._emit(f"\n{self.prefix}WARNING: {message}")

    def info(self, message: str) -> None:
        """Log information."""
        self._emit(f"\n{self.prefix}INFO: {message}")

    def success(self, message: str) -> None:
        """Log success."""
        self._emit(f"\n{self.prefix}SUCCESS: {message}")

    def _emit(self, text: str, end: str = "\n") -> None:
        """Write text to the configured stream using UTF-8 encoding."""
        stream = self._stream
        full_text = f"{text}{end}"

        try:
            buffer = getattr(stream, "buffer", None)
            if buffer is not None:
                buffer.write(full_text.encode("utf-8", errors="replace"))
                buffer.flush()
            else:
                stream.write(full_text)
                stream.flush()
        except Exception:
            fallback = getattr(sys, "__stdout__", None)
            if fallback is not None:
                print(full_text, end="", file=fallback)
