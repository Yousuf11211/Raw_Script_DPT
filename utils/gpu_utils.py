"""GPU detection and user-consent helpers.

Non-negotiable rules enforced here:
- GPU is NEVER used without explicit user consent.
- In non-interactive (no TTY) mode, default selection is CPU.

We intentionally check for Dask-CUDA (dask_cuda + CUDA visibility) rather than
Torch/TensorFlow presence. Some scripts may still use torch/tensorflow for
"GPU detected" messaging today; new engine code should use this module.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def gpu_available() -> bool:
    """Return True if a CUDA GPU looks usable for Dask-CUDA.

    This is a best-effort check that avoids importing heavy GPU frameworks.
    We consider GPU available only if:
      - dask_cuda can be imported, and
      - CUDA_VISIBLE_DEVICES is not set to an empty value.

    Note: This does not guarantee the GPU has enough memory for a workload.
    """
    try:
        import dask_cuda  # noqa: F401
    except Exception:
        return False

    # Respect explicit disabling.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and str(cvd).strip() == "":
        return False

    return True


def _is_interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def ask_user_gpu_choice(
    *,
    prompt: str = "GPU detected (CUDA). Do you want to use GPU acceleration? [y/N]: ",
    default: bool = False,
    assume_yes: Optional[bool] = None,
) -> bool:
    """Ask the user whether to use GPU.

    Args:
        prompt: Prompt text.
        default: Default choice if user presses Enter.
        assume_yes: If set, bypass the prompt and return this value.

    Returns:
        True if the user explicitly opts into GPU, otherwise False.

    Behavior:
        - If not interactive (no TTY): returns False.
        - Default answer is No.
    """
    if assume_yes is not None:
        return bool(assume_yes)

    if not _is_interactive():
        return False

    while True:
        raw = input(prompt).strip().lower()
        if raw == "":
            return bool(default)
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")

