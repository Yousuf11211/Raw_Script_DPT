"""Disk-backed helpers for cross-chunk duplicate detection.

Why this exists:
- Keeping a Python `set()` of seen hashes grows linearly with data size.
- For large CSVs, that will eventually OOM.

This module provides an *exact* dedup membership store using SQLite.
It is slower than in-memory sets but is safe and works out-of-core.

Usage pattern (pandas chunk loop):
    with SQLiteHashStore(db_path) as store:
        for chunk in pd.read_csv(..., chunksize=...):
            hashes = pd.util.hash_pandas_object(chunk, index=False)
            keep_mask = store.keep_mask(hashes)
            chunk = chunk.loc[keep_mask]

Important:
- This is exact dedup across the entire run.
- It does not require collecting all hashes in RAM.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, List, Sequence


class SQLiteHashStore:
    """SQLite-backed membership store for integer row hashes."""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "SQLiteHashStore":
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        # Fast-ish settings; still durable enough for a local job.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("CREATE TABLE IF NOT EXISTS seen (h INTEGER PRIMARY KEY);")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
        self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteHashStore is not open. Use it as a context manager.")
        return self._conn

    def _insert_ignore_many(self, hashes: Sequence[int]) -> None:
        # `INSERT OR IGNORE` is membership test + insert.
        self.conn.executemany("INSERT OR IGNORE INTO seen(h) VALUES (?);", [(int(h),) for h in hashes])

    def keep_mask(self, hashes: Iterable[int], *, batch_size: int = 50_000) -> List[bool]:
        """Return a boolean keep-mask where True means 'first time seen'.

        Implementation notes:
        - We commit in batches to keep memory and transaction size bounded.
        - For each batch:
            - we query which hashes already exist
            - then insert all hashes (ignoring existing)
        """
        seen_before: List[bool] = []
        buf: List[int] = []

        def flush() -> None:
            nonlocal buf, seen_before
            if not buf:
                return

            # Query existing.
            placeholders = ",".join(["?"] * len(buf))
            existing = set()
            for row in self.conn.execute(f"SELECT h FROM seen WHERE h IN ({placeholders});", buf):
                existing.add(int(row[0]))

            # keep if not existing
            seen_before.extend([h not in existing for h in buf])

            # insert all
            self._insert_ignore_many(buf)
            self.conn.commit()
            buf = []

        for h in hashes:
            buf.append(int(h))
            if len(buf) >= batch_size:
                flush()

        flush()
        return seen_before

