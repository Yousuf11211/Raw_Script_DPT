"""Chunk sizing and progress utilities.

This module exists to enforce consistent, memory-safe chunking rules:
- Chunk count must be pre-calculated from file size & chunk size.
- Scripts should print and use progress of the form:
  [Chunk 12 / 240] – 5.0% complete

We support both:
- Pandas streaming (row-chunks; caller decides chunk_rows), and
- Dask streaming (bytes-based blocksize).

Important: Do NOT compute full DataFrames; chunking must remain out-of-core.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkPlan:
    file_size_bytes: int
    chunk_size_bytes: int
    total_chunks: int


def compute_chunk_plan(input_file: str, chunk_size_mb: int) -> ChunkPlan:
    """Compute a file-size-based chunk plan.

    This does not read the CSV; it only uses file size.
    """
    if not input_file:
        raise ValueError("input_file is required")
    if chunk_size_mb <= 0:
        raise ValueError("chunk_size_mb must be > 0")

    file_size_bytes = os.path.getsize(input_file)
    chunk_size_bytes = int(chunk_size_mb) * 1024 * 1024
    total_chunks = int(math.ceil(file_size_bytes / float(chunk_size_bytes))) if file_size_bytes else 1

    return ChunkPlan(
        file_size_bytes=int(file_size_bytes),
        chunk_size_bytes=int(chunk_size_bytes),
        total_chunks=int(total_chunks),
    )


def format_progress(chunk_index: int, total_chunks: int) -> str:
    """Return a standard progress string.

    Args:
        chunk_index: 1-based chunk number.
        total_chunks: total chunks predicted.
    """
    total = max(1, int(total_chunks))
    idx = max(1, int(chunk_index))
    pct = min(100.0, (idx / total) * 100.0)
    return f"[Chunk {idx} / {total}] – {pct:.1f}% complete"


def print_chunk_plan(plan: ChunkPlan) -> None:
    """Print a human-readable chunk plan summary."""
    print(
        "Chunk plan: "
        f"file_size={plan.file_size_bytes:,} bytes, "
        f"chunk_size={plan.chunk_size_bytes:,} bytes, "
        f"total_chunks={plan.total_chunks:,}"
    )

