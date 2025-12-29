"""Path resolution helpers.

Rules:
- Input paths can be absolute or relative to the repo root.
- Output paths are resolved relative to (1) user provided output path OR
  (2) DEFAULT_OUTPUT_DIR under repo root.
- Never assume the current working directory is the script folder.
"""

from __future__ import annotations

import os
from typing import Optional

import config.global_config as global_config


def _repo_root() -> str:
    # This file lives at <repo_root>/utils/path_utils.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def resolve_input_path(path: str) -> str:
    """Return an absolute path for an input file/folder.

    If `path` is already absolute it's returned as-is.
    Otherwise it's treated as relative to the repo root.
    """
    if not path:
        raise ValueError("Input path is empty")
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(_repo_root(), path))


def resolve_output_path(path: Optional[str]) -> str:
    """Return an absolute output directory.

    - If `path` is provided: absolute paths are used, relative paths are resolved
      from repo root.
    - If not provided: defaults to <repo_root>/<DEFAULT_OUTPUT_DIR>.

    Directories are created.
    """
    if path:
        out = resolve_input_path(path)
    else:
        out = os.path.abspath(os.path.join(_repo_root(), global_config.DEFAULT_OUTPUT_DIR))

    os.makedirs(out, exist_ok=True)
    return out
