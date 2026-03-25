"""C-Flat gradient helper for Adam-NSCL.

Provides ``OfficialCFlatGradientHelper`` which generates flatter gradients
following the official C_Flat.step() sequence and injects them into the
Adam-NSCL update chain.

Official C-Flat sequence
------------------------
  get_grad  →  perturb0  →  disable BN stats  →  get_grad  →  unperturb
  →  grad_norm_ascent  →  perturb1  →  disable BN stats  →  get_grad
  →  gradient_aggregation  →  unperturb  →  re-enable BN stats

Fast path (λ = 0 / g0-only)
-----------------------------
  When ``lam == 0`` the g₁ / g₂ passes add no information to the aggregated
  gradient (g_final = g₀ + 0·g₁ + 0·g₂ = g₀).  The helper skips all
  perturbation passes and returns g₀ directly, avoiding expensive forward
  passes while preserving identical semantics.

Selective scope
---------------
  ``target_params`` restricts C-Flat to a subset of parameters.  Non-target
  parameters keep whatever gradient they had from the last standard backward
  pass (set by the caller before invoking this helper).

Projected variant
-----------------
  When ``project_before_perturb`` or ``project_after_aggregate`` is enabled
  and a ``transforms`` dict is supplied, gradients are projected into the
  null-space of the corresponding SVD basis (same projection used by
  AdamSVD) before perturbation is computed or after the final aggregation.
"""

import contextlib
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# BN-stats context helper
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _bn_stats_disabled(model: nn.Module):
    """Temporarily set BN layers to eval so that running stats are frozen."""
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    saved: List[tuple] = []
    for m in model.modules():
        if isinstance(m, bn_types):
            saved.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, was_training in saved:
            if was_training:
                m.train()


# ---------------------------------------------------------------------------
# SVD projection helper
# ---------------------------------------------------------------------------

def _svd_project(
    grads: Dict[int, Tensor],
    transforms: Dict[int, Optional[Tensor]],
) -> Dict[int, Tensor]:
    """Null-space project each gradient using the supplied SVD basis."""
    out: Dict[int, Tensor] = {}
    for pid, g in grads.items():
        U = transforms.get(pid)
        if U is not None and U.numel() > 0 and U.shape[0] == g.numel():
            g_flat = g.view(-1)
            g_flat = g_flat - U.mv(U.t().mv(g_flat))
            out[pid] = g_flat.view_as(g)
        else:
            out[pid] = g
    return out


# ---------------------------------------------------------------------------
# Main helper class
# ---------------------------------------------------------------------------

class OfficialCFlatGradientHelper:
    """Generate C-Flat gradients to feed into Adam-NSCL.

    Args:
        model:                  Neural network.
        rho:                    SAM perturbation radius. Default: 0.05.
        lam:                    Aggregation weight for g₁ and g₂.
                                Set to 0 to activate the λ=0 fast path
                                (g0-only).  Default: 0.0.
        target_params:          Parameters to apply C-Flat to.  ``None``
                                means all model parameters.
        transforms:             SVD basis dict ``{param_id: U}`` used for
                                projected variants.  Default: ``{}``.
        project_before_perturb: Project g₀ before computing the perturbation
                                direction.  Default: False.
        project_after_aggregate:Project the aggregated gradient before
                                writing it back to ``.grad``.  Default: False.
        eps:                    Numerical stability term.  Default: 1e-12.
    """

    def __init__(
        self,
        model: nn.Module,
        rho: float = 0.05,
        lam: float = 0.0,
        target_params: Optional[Iterable[nn.Parameter]] = None,
        transforms: Optional[Dict[int, Optional[Tensor]]] = None,
        project_before_perturb: bool = False,
        project_after_aggregate: bool = False,
        eps: float = 1e-12,
    ) -> None:
        self.model = model
        self.rho = rho
        self.lam = lam
        self.target_params: List[nn.Parameter] = (
            list(target_params) if target_params is not None
            else [p for p in model.parameters() if p.requires_grad]
        )
        self.target_param_ids = {id(p) for p in self.target_params}
        self.transforms: Dict[int, Optional[Tensor]] = transforms or {}
        self.project_before_perturb = project_before_perturb
        self.project_after_aggregate = project_after_aggregate
        self.eps = eps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_cflat_gradient(
        self, closure: Callable[[], Tensor]
    ) -> Dict[int, Tensor]:
        """Run the full C-Flat sequence and return the aggregated gradients.

        Returns
        -------
        g_final : dict mapping ``id(param)`` → gradient tensor (same shape
                  as the parameter, on the same device).
        """
        # ---- λ=0 fast path -----------------------------------------------
        # Use math.isclose to guard against floating-point representation
        # issues when lam is constructed from string parsing (e.g. 0.0).
        if abs(self.lam) < 1e-9:
            return self._fast_path_g0(closure)

        # ---- Step 1: get_grad → g₀ -----------------------------------
        g_0 = self._get_grad(closure)

        if self.project_before_perturb:
            g_0 = _svd_project(g_0, self.transforms)

        # ---- Step 2: perturb0 → w += e_w_0 ---------------------------
        e_w_0 = self._compute_perturbation(g_0)
        self._perturb(e_w_0)

        # ---- Steps 3–4: disable BN, get_grad → g₁ --------------------
        with _bn_stats_disabled(self.model):
            g_1 = self._get_grad(closure)

        # ---- Step 5: unperturb ----------------------------------------
        self._unperturb(e_w_0)

        # ---- Step 6: grad_norm_ascent → e_w_1_2 ----------------------
        e_w_1_2 = self._compute_perturbation(g_1)

        # ---- Steps 8–9: perturb1 (e_w_0 + e_w_1_2), get_grad → g₂ ---
        combined: Dict[int, Tensor] = {}
        for pid in e_w_0:
            combined[pid] = e_w_0[pid] + e_w_1_2.get(pid, torch.zeros_like(e_w_0[pid]))
        self._perturb(combined)

        with _bn_stats_disabled(self.model):
            g_2 = self._get_grad(closure)

        # ---- Step 10: gradient_aggregation ---------------------------
        g_final = self._gradient_aggregation(g_0, g_1, g_2)

        # ---- Step 11: unperturb --------------------------------------
        self._unperturb(combined)

        if self.project_after_aggregate:
            g_final = _svd_project(g_final, self.transforms)

        return g_final

    def apply_gradients_to_model(self, g_final: Dict[int, Tensor]) -> None:
        """Write ``g_final`` into ``.grad`` of each target parameter."""
        # Zero only target-param grads to avoid disturbing non-target grads.
        for p in self.target_params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        for p in self.target_params:
            pid = id(p)
            if pid in g_final:
                if p.grad is None:
                    p.grad = g_final[pid].clone()
                else:
                    p.grad.copy_(g_final[pid])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_grad(self, closure: Callable[[], Tensor]) -> Dict[int, Tensor]:
        """Execute closure + backward, collect target-param gradients."""
        self.model.zero_grad()
        with torch.enable_grad():
            loss = closure()
        loss.backward()
        return {
            id(p): p.grad.data.clone()
            for p in self.target_params
            if p.grad is not None
        }

    def _compute_perturbation(
        self, grads: Dict[int, Tensor]
    ) -> Dict[int, Tensor]:
        """Compute SAM-style perturbation: e_w = ρ · g / ‖g‖."""
        total_norm_sq = sum(g.norm(2).item() ** 2 for g in grads.values())
        total_norm = max(total_norm_sq ** 0.5, self.eps)
        scale = self.rho / total_norm
        return {pid: g.mul(scale) for pid, g in grads.items()}

    def _perturb(self, e_w: Dict[int, Tensor]) -> None:
        """Add perturbation to target-param data in-place."""
        for p in self.target_params:
            pid = id(p)
            if pid in e_w:
                p.data.add_(e_w[pid])

    def _unperturb(self, e_w: Dict[int, Tensor]) -> None:
        """Remove perturbation from target-param data in-place."""
        for p in self.target_params:
            pid = id(p)
            if pid in e_w:
                p.data.sub_(e_w[pid])

    def _gradient_aggregation(
        self,
        g_0: Dict[int, Tensor],
        g_1: Dict[int, Tensor],
        g_2: Dict[int, Tensor],
    ) -> Dict[int, Tensor]:
        """g_final = g₀ + λ·g₁ + λ·g₂."""
        result: Dict[int, Tensor] = {}
        for pid, g in g_0.items():
            agg = g.clone()
            if pid in g_1:
                agg = agg.add(g_1[pid], alpha=self.lam)
            if pid in g_2:
                agg = agg.add(g_2[pid], alpha=self.lam)
            result[pid] = agg
        return result

    def _fast_path_g0(
        self, closure: Callable[[], Tensor]
    ) -> Dict[int, Tensor]:
        """Return g₀ directly (λ≈0 ⟹ g_final = g₀).

        Skips all perturbation / extra forward passes while preserving the
        same mathematical result as the full sequence with λ=0.
        """
        g_0 = self._get_grad(closure)
        if self.project_before_perturb or self.project_after_aggregate:
            g_0 = _svd_project(g_0, self.transforms)
        return g_0
