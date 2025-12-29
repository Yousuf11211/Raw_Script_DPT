"""Small IO helpers used by multiple scripts.

Keeping these in utils/ reduces copy/paste and makes the repo refactor safer.
"""

from __future__ import annotations

import os


def make_unique_path(path: str) -> str:
    """Return a non-overwriting path by appending _runN before the extension."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 2
    while True:
        candidate = f"{base}_run{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1

