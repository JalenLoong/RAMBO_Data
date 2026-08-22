"""Numerical acceptance checks for RAMBO's pinned qpth solver."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
qpth_qp = pytest.importorskip("qpth.qp")
QPFunction = qpth_qp.QPFunction

from rambo.torch_runtime import ensure_cuda_linalg_loaded  # noqa: E402


def _solve(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve a strictly convex, interior two-variable QP and differentiate it."""

    dtype = torch.float64
    q = torch.eye(2, dtype=dtype, device=device).unsqueeze(0).requires_grad_(True)
    p = torch.tensor([[-0.2, 0.3]], dtype=dtype, device=device, requires_grad=True)
    # x >= 0, represented as -x <= 0.
    g = -torch.eye(2, dtype=dtype, device=device).unsqueeze(0)
    h = torch.zeros((1, 2), dtype=dtype, device=device)
    # An equality makes the KKT stationarity residual directly observable.
    a = torch.ones((1, 1, 2), dtype=dtype, device=device)
    b = torch.ones((1, 1), dtype=dtype, device=device)
    solution = QPFunction(eps=1.0e-12, maxIter=50, verbose=False)(q, p, g, h, a, b)
    solution.square().sum().backward()
    return solution.detach(), q.grad.detach(), p.grad.detach(), q.detach(), p.detach(), a.detach()


def _assert_qp_acceptance(device: torch.device) -> torch.Tensor:
    solution, q_grad, p_grad, q, p, a = _solve(device)
    expected = torch.tensor([[0.75, 0.25]], dtype=solution.dtype, device=device)
    torch.testing.assert_close(solution, expected, atol=2.0e-5, rtol=2.0e-5)
    assert torch.isfinite(q_grad).all()
    assert torch.isfinite(p_grad).all()

    equality_residual = torch.bmm(a, solution.unsqueeze(-1)).squeeze(-1) - 1.0
    inequality_residual = -solution
    # The inequality is strictly inactive; with A=[1, 1], the remaining
    # stationarity multiplier is a scalar shared by both coordinates.
    stationarity = torch.bmm(q, solution.unsqueeze(-1)).squeeze(-1) + p
    lagrange_multiplier = -stationarity.mean(dim=-1, keepdim=True)
    stationarity_residual = stationarity + lagrange_multiplier * a.squeeze(1)
    assert float(equality_residual.abs().max()) < 2.0e-5
    assert float(inequality_residual.max()) <= 2.0e-5
    assert float(stationarity_residual.abs().max()) < 2.0e-5
    return solution


def test_qpth_cpu_forward_backward_kkt_and_determinism() -> None:
    first = _assert_qp_acceptance(torch.device("cpu"))
    second = _assert_qp_acceptance(torch.device("cpu"))
    torch.testing.assert_close(first, second, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.cuda
def test_qpth_cuda_forward_backward_kkt_and_determinism() -> None:
    require_cuda = os.environ.get("RAMBO_REQUIRE_CUDA_QPTH", "0") == "1"
    if not torch.cuda.is_available():
        if require_cuda:
            pytest.fail("RAMBO_REQUIRE_CUDA_QPTH=1 but CUDA is not available")
        pytest.skip("CUDA is unavailable")

    loaded_library = ensure_cuda_linalg_loaded()
    assert loaded_library is not None
    assert loaded_library.name == "libtorch_cuda_linalg.so"
    first = _assert_qp_acceptance(torch.device("cuda"))
    second = _assert_qp_acceptance(torch.device("cuda"))
    torch.testing.assert_close(first, second, atol=1.0e-10, rtol=1.0e-10)
