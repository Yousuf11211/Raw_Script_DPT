"""Execution engine selection and (optional) Dask cluster initialization.

Every standalone script must accept:
  --engine pandas|dask|dask-gpu

Strict rules enforced:
- GPU is NEVER used silently.
- If CUDA + dask_cuda are available, and the run is interactive, we ask:
    GPU detected (CUDA).
    Do you want to use GPU acceleration? [y/N]:
  Default is No.
- If non-interactive (no TTY), default is CPU.
- Flags override prompts:
    --use-gpu : Force GPU or fail
    --no-gpu  : Force CPU

Notes:
- We do NOT import dask or dask_cuda unless required.
- Many scripts are still pandas-only; they can accept --engine now and
  continue to run in pandas mode until a full dask migration is applied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from utils.gpu_utils import ask_user_gpu_choice, gpu_available

SUPPORTED_ENGINES = ("pandas", "dask", "dask-gpu")


@dataclass(frozen=True)
class EngineSelection:
    """Final engine decision resolved under the strict rules."""

    engine: str
    use_gpu: bool


def select_engine(*, engine: str, use_gpu_flag: bool = False, no_gpu_flag: bool = False) -> EngineSelection:
    """Resolve final engine choice under the repo's strict rules."""
    eng = (engine or "").strip().lower()
    if eng not in SUPPORTED_ENGINES:
        raise ValueError(f"Unsupported --engine '{engine}'. Choose one of {SUPPORTED_ENGINES}.")

    if use_gpu_flag and no_gpu_flag:
        raise ValueError("--use-gpu and --no-gpu are mutually exclusive")

    cuda_ok = gpu_available()

    # dask-gpu explicitly requests GPU execution.
    if eng == "dask-gpu":
        if no_gpu_flag:
            raise ValueError("--engine dask-gpu cannot be combined with --no-gpu")
        if not cuda_ok:
            raise RuntimeError(
                "--engine dask-gpu requested, but GPU/Dask-CUDA not available. "
                "Install dask-cuda and ensure CUDA is configured."
            )

        if use_gpu_flag:
            return EngineSelection(engine=eng, use_gpu=True)

        # No explicit flag: ask the user in interactive mode.
        if ask_user_gpu_choice():
            return EngineSelection(engine=eng, use_gpu=True)

        # User declined => safe fallback.
        return EngineSelection(engine="dask", use_gpu=False)

    # pandas/dask engines: GPU is optional.
    if no_gpu_flag:
        return EngineSelection(engine=eng, use_gpu=False)

    if use_gpu_flag:
        if not cuda_ok:
            raise RuntimeError("--use-gpu set but GPU/Dask-CUDA not available")
        # If engine is pandas, we keep pandas execution; caller must still not use GPU silently.
        return EngineSelection(engine=eng, use_gpu=True)

    # No flags: only ask if engine=='dask' (GPU doesn't apply to pandas here).
    if eng == "dask" and cuda_ok:
        if ask_user_gpu_choice():
            return EngineSelection(engine=eng, use_gpu=True)

    return EngineSelection(engine=eng, use_gpu=False)


def init_dask_cluster(*, use_gpu: bool, local_directory: Optional[str] = None, memory_limit: str | int | None = "auto"):
    """Initialize a Dask cluster consistent with the chosen engine.

    Returns:
        (client, cluster)

    Important:
        Call this ONLY after select_engine() indicates GPU is allowed.
    """

    # Lazy import so pandas-only scripts don't need dask installed.
    from dask.distributed import Client, LocalCluster

    if local_directory:
        os.makedirs(local_directory, exist_ok=True)

    if use_gpu:
        # Only import dask_cuda if we truly plan to use GPU.
        from dask_cuda import LocalCUDACluster

        cluster = LocalCUDACluster(local_directory=local_directory, memory_limit=memory_limit)
    else:
        cluster = LocalCluster(local_directory=local_directory, memory_limit=memory_limit, processes=True)

    client = Client(cluster)

    # Memory-safe eviction hook: force worker-side GC.
    try:
        import gc

        client.run(gc.collect)
    except Exception:
        pass

    return client, cluster
