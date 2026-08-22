"""Small PyTorch runtime compatibility helpers for RAMBO.

The CUDA 12.8 PyTorch wheel lazily opens ``libtorch_cuda_linalg.so`` when a
linear-algebra operation is first used.  On this host the system
``LD_LIBRARY_PATH`` contains only NVIDIA driver directories, so that late load
cannot discover the sibling library under ``torch/lib``.  Loading the absolute
wheel path once, globally, preserves the official wheel unchanged and makes
qpth's CUDA linear algebra usable from the ordinary RAMBO entry points.
"""

from __future__ import annotations

import ctypes
from pathlib import Path


def ensure_cuda_linalg_loaded() -> Path | None:
    """Preload PyTorch's CUDA linear-algebra library when CUDA is available.

    Returns the loaded library path, or ``None`` on CPU-only installations.
    The helper deliberately imports only PyTorch; it has no Isaac Sim or Kit
    side effects and is safe to call immediately after ``AppLauncher`` starts.
    """

    import torch

    if not torch.cuda.is_available():
        return None

    library_path = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cuda_linalg.so"
    if not library_path.is_file():
        raise RuntimeError(
            "PyTorch CUDA is available but its linalg library is missing: "
            f"{library_path}"
        )
    try:
        ctypes.CDLL(str(library_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    except OSError as error:
        raise RuntimeError(
            "Unable to preload PyTorch CUDA linear algebra from "
            f"{library_path}. Run through scripts/rambo/run.sh or ensure the "
            "wheel's torch/lib directory is visible to the dynamic loader."
        ) from error
    return library_path


__all__ = ["ensure_cuda_linalg_loaded"]
