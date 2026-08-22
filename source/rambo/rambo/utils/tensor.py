"""Tensor conversion helpers without a hard-coded CUDA device."""

from __future__ import annotations

from typing import Any

import torch


def to_torch(
    value: Any,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create a detached tensor on an explicit or safely inferred device."""

    if isinstance(value, torch.Tensor):
        target_device = value.device if device is None else device
        return value.detach().clone().to(device=target_device, dtype=dtype).requires_grad_(requires_grad)
    target_device = "cpu" if device is None else device
    return torch.as_tensor(value, dtype=dtype, device=target_device).clone().detach().requires_grad_(requires_grad)
