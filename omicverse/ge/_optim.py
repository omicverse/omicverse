"""``RowAdagrad``: PyTorch-BigGraph's row-wise Adagrad for embedding tables.

Ported from ``torchbiggraph/row_adagrad.py``. Each embedding table is a 2-D
parameter ``(n_entities, dim)`` with **one Adagrad accumulator per row**
(not per coordinate, as in ``torch.optim.Adagrad``): the squared gradient is
averaged across the embedding dimension before being added to that row's
running sum, so every dimension of a row shares one effective learning
rate. This is what lets PBG/SIMBA update one giant table with plain SGD-like
cost per step regardless of how many *other* rows it holds.

The HOGWILD-safety dance in the original (reading ``state['sum']`` into a
local copy before adding, so concurrent processes' updates can't shrink the
accumulator below what any one process has already seen) is irrelevant here
-- this training loop is single-process -- so it has been dropped, but the
arithmetic is otherwise identical, including the dense-gradient branch
(which is what all ``ov.ge`` training actually uses: fancy-indexing an
``nn.Parameter`` produces a dense gradient, not a sparse one).
"""
from __future__ import annotations

from torch.optim import Optimizer


class RowAdagrad(Optimizer):
    """Row-wise Adagrad; every parameter must be a 2-D tensor."""

    def __init__(self, params, lr: float = 1e-2, eps: float = 1e-10):
        defaults = {"lr": lr, "eps": eps}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError("RowAdagrad only works on 2D tensors "
                                     f"(got shape {tuple(p.shape)})")
                self.state[p]["sum"] = p.new_zeros((p.shape[0],))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            eps = group["eps"]
            clr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if grad.is_sparse:
                    grad = grad.coalesce()
                    if grad._indices().numel() == 0:
                        continue
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    delta_g = (grad_values * grad_values).mean(1)
                    state["sum"].index_add_(0, grad_indices, delta_g)
                    local_g = state["sum"][grad_indices]
                    std_values = local_g.sqrt().add_(eps).unsqueeze(1)
                    p.data.index_add_(0, grad_indices, -clr * grad_values / std_values)
                else:
                    delta_g = (grad * grad).mean(1)
                    state["sum"].add_(delta_g)
                    std = state["sum"].sqrt().add(eps)
                    p.data.addcdiv_(grad, std.unsqueeze(1), value=-clr)
        return loss


__all__ = ["RowAdagrad"]
