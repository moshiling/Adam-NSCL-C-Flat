"""Adam-NSCL: Adam optimizer with SVD null-space projection for continual learning.

Reference: Adam-NSCL (NeurIPS 2021)

Core idea:
  After completing task t, compute the SVD of the accumulated gradient matrix
  to obtain the principal subspace U.  During training on task t+1, project
  the gradient into the null-space of U so that updates do not interfere with
  previously learned representations.

NOTE: The core optimizer step is intentionally left unchanged; C-Flat only
acts as a gradient-generation frontend that feeds into this class.
"""

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamSVD(Optimizer):
    """Adam with SVD null-space projection (Adam-NSCL).

    Args:
        params:        Iterable of parameters or parameter groups.
        lr:            Learning rate. Default: 1e-3.
        betas:         Adam β₁, β₂. Default: (0.9, 0.999).
        eps:           Adam ε. Default: 1e-8.
        weight_decay:  L2 penalty. Default: 0.
        amsgrad:       Use AMSGrad variant. Default: False.

    Attributes:
        transforms (dict):  Mapping from ``id(param)`` to an orthonormal basis
                            matrix U of shape ``[n_features, n_components]``.
                            Set externally after each task via
                            :meth:`update_transforms`.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
        super(AdamSVD, self).__init__(params, defaults)

        # SVD transforms: param_id -> U  (orthonormal column basis of previous
        # task gradient subspace).  Populated externally after each task.
        self.transforms: Dict[int, Optional[Tensor]] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def update_transforms(self, new_transforms: Dict[int, Optional[Tensor]]) -> None:
        """Merge ``new_transforms`` into :attr:`transforms`."""
        self.transforms.update(new_transforms)

    def set_transforms(self, transforms: Dict[int, Optional[Tensor]]) -> None:
        """Replace :attr:`transforms` with ``transforms``."""
        self.transforms = transforms

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Performs a single optimization step.

        Applies SVD null-space projection to each parameter's gradient before
        executing the standard Adam update.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamSVD does not support sparse gradients.")

                # ---- SVD null-space projection --------------------------------
                param_id = id(p)
                if param_id in self.transforms:
                    U = self.transforms[param_id]
                    if U is not None and U.numel() > 0 and U.shape[0] == grad.numel():
                        g = grad.view(-1)
                        # g_projected = g - U (U^T g)
                        g = g - U.mv(U.t().mv(g))
                        grad = g.view_as(grad)

                # ---- Adam state ----------------------------------------------
                amsgrad = group["amsgrad"]
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Momentum updates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if amsgrad:
                    max_sq: Tensor = state["max_exp_avg_sq"]
                    torch.maximum(max_sq, exp_avg_sq, out=max_sq)
                    denom = (max_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                step_size = group["lr"] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
