"""``RowAdagrad`` vs. the row-wise Adagrad update formula
(``torchbiggraph/row_adagrad.py``'s dense-gradient branch)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omicverse.ge._optim import RowAdagrad


def _manual_step(p, grad, state_sum, lr, eps=1e-10):
    delta_g = (grad * grad).mean(1)
    state_sum = state_sum + delta_g
    std = state_sum.sqrt() + eps
    p_new = p - lr * grad / std.unsqueeze(1)
    return p_new, state_sum


def test_single_step_matches_formula():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(5, 4))
    opt = RowAdagrad([p], lr=0.3)
    grad = torch.randn(5, 4)
    p_before = p.detach().clone()
    p.grad = grad.clone()
    opt.step()

    expected, _ = _manual_step(p_before, grad, torch.zeros(5), lr=0.3)
    assert torch.allclose(p.detach(), expected, atol=1e-6)


def test_accumulator_is_per_row_not_per_coordinate():
    """The defining property of *row* Adagrad: the accumulator is one
    scalar per row (mean squared grad across the embedding dim), not one
    per coordinate as in plain Adagrad."""
    p = torch.nn.Parameter(torch.zeros(2, 3))
    opt = RowAdagrad([p], lr=1.0)
    # row 0: uniform gradient -> mean == any coordinate's square
    # row 1: lopsided gradient -> mean squared differs a lot from any single coord
    grad = torch.tensor([[2.0, 2.0, 2.0], [10.0, 0.0, 0.0]])
    p.grad = grad
    opt.step()
    state_sum = opt.state[p]["sum"]
    assert torch.allclose(state_sum, torch.tensor([4.0, 100.0 / 3]))
    # row 1's two update steps use the SAME std (not per-coordinate), so the
    # coordinate that had zero gradient still doesn't move (addcdiv with 0
    # numerator), but the ratio between row-0 and row-1 step sizes reflects
    # the shared per-row denominator.
    assert p[1, 1].item() == 0.0 and p[1, 2].item() == 0.0


def test_multiple_steps_accumulate():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(3, 2))
    opt = RowAdagrad([p], lr=0.1)
    p_ref = p.detach().clone()
    state_sum = torch.zeros(3)
    for _ in range(5):
        grad = torch.randn(3, 2)
        p.grad = grad.clone()
        opt.step()
        p_ref, state_sum = _manual_step(p_ref, grad, state_sum, lr=0.1)
    assert torch.allclose(p.detach(), p_ref, atol=1e-5)
    assert torch.allclose(opt.state[p]["sum"], state_sum, atol=1e-6)


def test_rejects_non_2d_params():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError):
        RowAdagrad([p])


def test_none_grad_is_skipped():
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = RowAdagrad([p], lr=0.1)
    before = p.detach().clone()
    opt.step()  # p.grad is None
    assert torch.equal(p.detach(), before)
