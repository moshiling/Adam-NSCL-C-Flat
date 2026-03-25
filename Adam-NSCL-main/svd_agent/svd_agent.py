"""SVD agent: scope parsing and SVD-transform computation for Adam-NSCL + C-Flat.

Scope parsing (``get_cflat_target_params``)
-------------------------------------------
Translates a human-readable ``scope`` string into the list of
``nn.Parameter`` objects that C-Flat should act on, along with the
matching subset of ``optimizer.transforms`` for projected variants.

Supported ``scope`` values
  * ``"all"``                    – every model parameter
  * ``"classifier"``             – the final linear head
  * ``"deep"``                   – deep feature layers (rule controlled by
                                   ``deep_layer_rule``)
  * ``"deep_plus_classifier"``   – deep layers + classifier head
  * ``"last_block_plus_classifier"`` – alias for
                                   ``deep_plus_classifier`` with
                                   ``deep_layer_rule="last_block"``

Supported ``deep_layer_rule`` values
  * ``"last_block"``   – last residual block (e.g. layer4 in ResNet-18)
  * ``"last_stage"``   – last top-level feature module
  * ``"last_third"``   – final one-third of feature modules

SVD-transform computation (``SVDTransformComputer``)
-----------------------------------------------------
After each task, collects per-parameter gradient vectors over a replay
loader, stacks them into a matrix, runs truncated SVD to extract the
principal subspace U, and returns a ``{param_id: U}`` dict suitable for
setting on ``AdamSVD.transforms``.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Public API: scope → target params + transforms
# ---------------------------------------------------------------------------

def get_cflat_target_params(
    model: nn.Module,
    optimizer,
    scope: str = "last_block_plus_classifier",
    deep_layer_rule: str = "last_block",
) -> Tuple[List[nn.Parameter], Dict[int, Optional[Tensor]]]:
    """Return (target_params, transforms) for C-Flat based on ``scope``.

    Args:
        model:            Neural network.
        optimizer:        ``AdamSVD`` instance whose ``.transforms`` dict is
                          used to build the projected-variant transforms.
        scope:            Which parameters C-Flat should act on.
        deep_layer_rule:  Sub-rule for the ``"deep"`` family of scopes.

    Returns:
        target_params: Parameters that C-Flat perturbations apply to.
        transforms:    Subset of ``optimizer.transforms`` for those params.
    """
    scope = scope.lower()

    if scope == "all":
        target_params = [p for p in model.parameters() if p.requires_grad]

    elif scope == "classifier":
        target_params = _get_classifier_params(model)

    elif scope == "deep":
        target_params = _get_deep_params(model, deep_layer_rule)

    elif scope in ("deep_plus_classifier", "last_block_plus_classifier"):
        # For "last_block_plus_classifier" always use last_block rule.
        rule = "last_block" if scope == "last_block_plus_classifier" else deep_layer_rule
        deep_params = _get_deep_params(model, rule)
        classifier_params = _get_classifier_params(model)
        deep_ids = {id(p) for p in deep_params}
        extra = [p for p in classifier_params if id(p) not in deep_ids]
        target_params = deep_params + extra

    else:
        raise ValueError(
            f"Unknown scope '{scope}'.  Must be one of: 'all', 'classifier', "
            f"'deep', 'deep_plus_classifier', 'last_block_plus_classifier'."
        )

    # Extract matching transforms from optimizer.transforms
    transforms: Dict[int, Optional[Tensor]] = {}
    if hasattr(optimizer, "transforms"):
        target_ids = {id(p) for p in target_params}
        transforms = {
            k: v
            for k, v in optimizer.transforms.items()
            if k in target_ids
        }

    return target_params, transforms


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------

def _get_classifier_params(model: nn.Module) -> List[nn.Parameter]:
    """Return parameters of the classification head."""
    for attr in ("classifier", "fc", "head", "linear", "output"):
        if hasattr(model, attr):
            params = [p for p in getattr(model, attr).parameters() if p.requires_grad]
            if params:
                return params
    # Fallback: last named child
    children = list(model.named_children())
    if children:
        params = [p for p in children[-1][1].parameters() if p.requires_grad]
        return params
    return [p for p in model.parameters() if p.requires_grad]


def _get_feature_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return named top-level modules that are *not* the classifier head."""
    head_names = {"classifier", "fc", "head", "linear", "output"}
    return [
        (name, m)
        for name, m in model.named_children()
        if name not in head_names
    ]


def _get_deep_params(model: nn.Module, rule: str) -> List[nn.Parameter]:
    """Dispatch to the appropriate deep-layer rule."""
    rule = rule.lower()
    if rule == "last_block":
        return _last_block_params(model)
    if rule == "last_stage":
        return _last_stage_params(model)
    if rule == "last_third":
        return _last_third_params(model)
    raise ValueError(
        f"Unknown deep_layer_rule '{rule}'.  Must be one of: "
        "'last_block', 'last_stage', 'last_third'."
    )


def _last_block_params(model: nn.Module) -> List[nn.Parameter]:
    """Parameters of the last residual block (e.g. layer4[-1] in ResNet)."""
    # ResNet-style: prefer layer4 > layer3
    for attr in ("layer4", "layer3", "layer2", "layer1"):
        stage = getattr(model, attr, None)
        if stage is not None and isinstance(stage, nn.Sequential):
            children = list(stage.children())
            if children:
                params = [p for p in children[-1].parameters() if p.requires_grad]
                if params:
                    return params

    # Generic fallback: look for Sequential feature modules
    feature_mods = _get_feature_modules(model)
    for _, m in reversed(feature_mods):
        if isinstance(m, nn.Sequential):
            children = list(m.children())
            if children:
                params = [p for p in children[-1].parameters() if p.requires_grad]
                if params:
                    return params
        # Non-sequential module
        params = [p for p in m.parameters() if p.requires_grad]
        if params:
            return params

    # Ultimate fallback: all parameters
    return [p for p in model.parameters() if p.requires_grad]


def _last_stage_params(model: nn.Module) -> List[nn.Parameter]:
    """Parameters of the last top-level feature module that has parameters."""
    feature_mods = _get_feature_modules(model)
    # Iterate in reverse to find the last module that actually has parameters.
    for _, m in reversed(feature_mods):
        params = [p for p in m.parameters() if p.requires_grad]
        if params:
            return params
    return [p for p in model.parameters() if p.requires_grad]


def _last_third_params(model: nn.Module) -> List[nn.Parameter]:
    """Parameters in the last one-third of feature modules."""
    feature_mods = _get_feature_modules(model)
    n = len(feature_mods)
    start = max(0, n - max(1, n // 3))
    params: List[nn.Parameter] = []
    for _, m in feature_mods[start:]:
        params.extend(p for p in m.parameters() if p.requires_grad)
    return params or [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# SVD transform computer
# ---------------------------------------------------------------------------

class SVDTransformComputer:
    """Compute truncated-SVD null-space transforms after each CL task.

    After training task *t*, feed the task's training loader through the
    frozen model to collect gradient vectors, stack them into a matrix G,
    then compute G = U S V^T and store U[:, :k] as the null-space basis that
    Adam-NSCL will project away from when training task *t+1*.

    Args:
        model:        Neural network (should be the same object as used in
                      training).
        optimizer:    ``AdamSVD`` instance; transforms are set on it via
                      :meth:`update_and_apply`.
        n_components: Maximum rank of the basis to keep per parameter.
        device:       Device on which to accumulate gradient tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        n_components: int = 50,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.n_components = n_components
        self.device = device

    @torch.no_grad()
    def compute_transforms(
        self,
        dataloader: Iterable,
        criterion: nn.Module,
        n_batches: Optional[int] = None,
    ) -> Dict[int, Optional[Tensor]]:
        """Collect gradients and return ``{param_id: U}`` transforms.

        Args:
            dataloader: Yields ``(inputs, targets)`` batches.
            criterion:  Loss function for gradient collection.
            n_batches:  Number of batches to use.  ``None`` = all.

        Returns:
            Mapping from ``id(param)`` to orthonormal basis U of shape
            ``[n_features, n_components]``.
        """
        was_training = self.model.training
        self.model.eval()

        param_list = [p for p in self.model.parameters() if p.requires_grad]
        grad_lists: Dict[int, List[Tensor]] = {id(p): [] for p in param_list}

        for batch_idx, batch in enumerate(dataloader):
            if n_batches is not None and batch_idx >= n_batches:
                break
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

            self.model.zero_grad()
            with torch.enable_grad():
                out = self.model(inputs)
                if isinstance(out, dict):
                    out = out["logits"]
                loss = criterion(out, targets)
            loss.backward()

            for p in param_list:
                if p.grad is not None:
                    grad_lists[id(p)].append(p.grad.data.view(-1).clone())

        self.model.train(was_training)

        transforms: Dict[int, Optional[Tensor]] = {}
        for p in param_list:
            pid = id(p)
            g_list = grad_lists[pid]
            if not g_list:
                transforms[pid] = None
                continue
            G = torch.stack(g_list, dim=1)  # [n_features, n_samples]
            k = min(self.n_components, G.shape[0], G.shape[1])
            try:
                U, _, _ = torch.linalg.svd(G, full_matrices=False)
                transforms[pid] = U[:, :k].to(self.device)
            except Exception:
                transforms[pid] = None

        return transforms

    def update_and_apply(
        self,
        dataloader: Iterable,
        criterion: nn.Module,
        n_batches: Optional[int] = None,
    ) -> None:
        """Compute new transforms and merge them into ``optimizer.transforms``.

        For tasks > 0 the existing and new bases are concatenated and
        re-orthogonalized via SVD so that the null-space accumulates across
        all seen tasks.
        """
        new_T = self.compute_transforms(dataloader, criterion, n_batches)

        if not self.optimizer.transforms:
            self.optimizer.transforms = new_T
            return

        merged: Dict[int, Optional[Tensor]] = {}
        all_pids = set(self.optimizer.transforms) | set(new_T)
        for pid in all_pids:
            old_U = self.optimizer.transforms.get(pid)
            new_U = new_T.get(pid)
            if old_U is None:
                merged[pid] = new_U
            elif new_U is None:
                merged[pid] = old_U
            else:
                combined = torch.cat([old_U, new_U], dim=1)
                k = min(self.n_components, combined.shape[0], combined.shape[1])
                try:
                    U, _, _ = torch.linalg.svd(combined, full_matrices=False)
                    merged[pid] = U[:, :k]
                except Exception:
                    merged[pid] = old_U
        self.optimizer.transforms = merged
