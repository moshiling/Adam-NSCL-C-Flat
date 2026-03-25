"""ContinualLearningAgent: Adam-NSCL training loop with optional C-Flat frontend.

Design principles
-----------------
* ``use_cflat``      – apply the full-model (non-selective) C-Flat gradient
                       frontend before every Adam-NSCL step.
* ``use_pls_cflat``  – apply the *Parameter-Layer Selective* C-Flat variant,
                       restricting perturbations to the parameters selected by
                       ``cflat_target_scope`` / ``deep_layer_rule``.
* The ``closure`` passed to C-Flat contains **only** the classification loss;
  Adam-NSCL's null-space projection / regularisation operates through the
  optimizer itself, keeping the two concerns cleanly separated.
* Loss and accuracy are counted **once** per batch regardless of how many
  extra forward passes C-Flat performs internally.
* Non-target parameters always receive gradients from a standard backward pass
  (the last ``_get_grad`` call inside C-Flat leaves them intact for ``lam > 0``;
  the fast-path restores them explicitly).
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from optim.adam_svd import AdamSVD
from optim.cflat_utils import OfficialCFlatGradientHelper
from svd_agent.svd_agent import SVDTransformComputer, get_cflat_target_params


class ContinualLearningAgent:
    """Continual-learning agent combining Adam-NSCL and (optionally) C-Flat.

    Args:
        config (dict): Flat configuration dict.  See ``_default_config`` for
                       supported keys.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device: str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Adam-NSCL hyper-parameters
        self.lr: float = config.get("lr", 1e-3)
        self.betas: Tuple[float, float] = config.get("betas", (0.9, 0.999))
        self.eps_adam: float = config.get("eps", 1e-8)
        self.weight_decay: float = config.get("weight_decay", 0.0)
        self.epochs: int = config.get("epochs", 50)
        self.n_svd_components: int = config.get("n_svd_components", 50)
        self.svd_n_batches: Optional[int] = config.get("svd_n_batches", None)

        # C-Flat flags & hyper-parameters
        self.use_cflat: bool = config.get("use_cflat", False)
        self.use_pls_cflat: bool = config.get("use_pls_cflat", False)
        self.cflat_rho: float = config.get("cflat_rho", 0.05)
        self.cflat_lam: float = config.get("cflat_lam", 0.0)
        self.cflat_target_scope: str = config.get(
            "cflat_target_scope", "last_block_plus_classifier"
        )
        self.deep_layer_rule: str = config.get("deep_layer_rule", "last_block")
        self.project_before_perturb: bool = config.get("project_before_perturb", False)
        self.project_after_aggregate: bool = config.get("project_after_aggregate", False)

        # Misc
        self.n_tasks: int = config.get("n_tasks", 10)
        self.verbose: bool = config.get("verbose", True)

        # Will be set in setup()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[AdamSVD] = None
        self.svd_computer: Optional[SVDTransformComputer] = None
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, model: nn.Module) -> None:
        """Bind ``model`` and create optimizer + SVD computer."""
        self.model = model.to(self.device)
        self.optimizer = AdamSVD(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps_adam,
            weight_decay=self.weight_decay,
        )
        self.svd_computer = SVDTransformComputer(
            self.model,
            self.optimizer,
            n_components=self.n_svd_components,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Dict:
        """Train the model on ``task_id`` for ``self.epochs`` epochs.

        Returns a dict with ``task_id``, ``final_loss``, ``final_acc``,
        ``time``, and per-epoch ``losses`` / ``accs`` lists.
        """
        assert self.model is not None and self.optimizer is not None, \
            "Call setup() before train_task()."

        self.model.train()

        # Build (or rebuild) the C-Flat helper once per task so that the
        # target-param list always reflects the current optimizer.transforms.
        cflat_helper: Optional[OfficialCFlatGradientHelper] = None
        if self.use_cflat or self.use_pls_cflat:
            cflat_helper = self._build_cflat_helper()

        epoch_losses: List[float] = []
        epoch_accs: List[float] = []
        t0 = time.time()

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if cflat_helper is not None:
                    # ---- C-Flat + Adam-NSCL path ---------------------------------
                    loss_val, acc = self._train_step_cflat(
                        inputs, targets, cflat_helper
                    )
                else:
                    # ---- Standard Adam-NSCL path ------------------------------
                    loss_val, acc = self._train_step_standard(inputs, targets)

                running_loss += loss_val
                running_correct += int(acc * inputs.size(0))
                running_total += inputs.size(0)

            avg_loss = running_loss / max(len(train_loader), 1)
            avg_acc = running_correct / max(running_total, 1)
            epoch_losses.append(avg_loss)
            epoch_accs.append(avg_acc)

            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                print(
                    f"  Task {task_id} | Epoch {epoch + 1:3d}/{self.epochs} | "
                    f"Loss {avg_loss:.4f} | Acc {avg_acc:.4f}"
                )

        elapsed = time.time() - t0
        return {
            "task_id": task_id,
            "final_loss": epoch_losses[-1],
            "final_acc": epoch_accs[-1],
            "losses": epoch_losses,
            "accs": epoch_accs,
            "time": elapsed,
        }

    # ------------------------------------------------------------------
    # Standard Adam-NSCL step
    # ------------------------------------------------------------------

    def _train_step_standard(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float]:
        """One Adam-NSCL gradient step without C-Flat."""
        self.optimizer.zero_grad()
        outputs = self._forward(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            _, predicted = outputs.max(1)
            acc = predicted.eq(targets).float().mean().item()

        return loss.item(), acc

    # ------------------------------------------------------------------
    # C-Flat + Adam-NSCL step
    # ------------------------------------------------------------------

    def _train_step_cflat(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        helper: OfficialCFlatGradientHelper,
    ) -> Tuple[float, float]:
        """One C-Flat + Adam-NSCL gradient step.

        The closure passed to C-Flat contains **only** the classification
        loss.  Adam-NSCL's null-space projection is applied automatically
        inside ``optimizer.step()`` via ``optimizer.transforms``.

        Non-target parameters also need gradients for the Adam update.
        We detect whether they are missing and perform a lightweight
        backward pass if necessary, then restore the C-Flat gradients so
        they are not overwritten.
        """
        # Closure: classification loss only (no regularisation)
        def closure() -> torch.Tensor:
            return self.criterion(self._forward(inputs), targets)

        # ---- C-Flat gradient frontend ------------------------------------
        g_final = helper.compute_cflat_gradient(closure)
        helper.apply_gradients_to_model(g_final)

        # ---- Ensure non-target params have gradients ---------------------
        all_req_grad = {id(p) for p in self.model.parameters() if p.requires_grad}
        target_ids = {id(p) for p in helper.target_params}
        non_target_need_grad = any(
            p.grad is None
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in target_ids
        )
        if non_target_need_grad and (all_req_grad - target_ids):
            # Zero non-target grads only; preserve the C-Flat target grads.
            for p in self.model.parameters():
                if p.requires_grad and id(p) not in target_ids:
                    if p.grad is not None:
                        p.grad.zero_()
            with torch.enable_grad():
                loss_nt = closure()
            loss_nt.backward()
            # Restore C-Flat gradients in case backward overwrote them.
            helper.apply_gradients_to_model(g_final)

        # ---- Adam-NSCL step (null-space projection inside) ---------------
        self.optimizer.step()

        # ---- Compute loss & accuracy **once** (no extra forward) --------
        with torch.no_grad():
            out = self._forward(inputs)
            loss_val = self.criterion(out, targets).item()
            _, predicted = out.max(1)
            acc = predicted.eq(targets).float().mean().item()

        return loss_val, acc

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        task_id: int,
        test_loader: torch.utils.data.DataLoader,
    ) -> Dict:
        """Evaluate on ``test_loader``; return ``{task_id, loss, acc}``."""
        assert self.model is not None
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self._forward(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

        self.model.train()
        return {
            "task_id": task_id,
            "loss": total_loss / max(len(test_loader), 1),
            "acc": total_correct / max(total_samples, 1),
        }

    # ------------------------------------------------------------------
    # Post-task SVD update
    # ------------------------------------------------------------------

    def after_task(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
    ) -> None:
        """Compute and merge SVD transforms after finishing task ``task_id``.

        The transforms are stored in ``self.optimizer.transforms`` and
        will be used by Adam-NSCL in subsequent tasks.
        """
        if self.verbose:
            print(f"  [SVD] computing transforms after task {task_id} …")
        assert self.svd_computer is not None
        self.svd_computer.update_and_apply(
            train_loader,
            self.criterion,
            n_batches=self.svd_n_batches,
        )
        if self.verbose:
            n_transforms = len(self.optimizer.transforms)
            print(f"  [SVD] transforms stored: {n_transforms} parameter entries.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.model(inputs)
        if isinstance(out, dict):
            return out["logits"]
        return out

    def _build_cflat_helper(self) -> OfficialCFlatGradientHelper:
        """Instantiate C-Flat helper with current scope / transform settings."""
        if self.use_pls_cflat:
            # PLS-CFlat: selective application
            target_params, transforms = get_cflat_target_params(
                self.model,
                self.optimizer,
                scope=self.cflat_target_scope,
                deep_layer_rule=self.deep_layer_rule,
            )
        else:
            # Standard C-Flat: all parameters, no projection
            target_params = [p for p in self.model.parameters() if p.requires_grad]
            transforms = (
                self.optimizer.transforms
                if (self.project_before_perturb or self.project_after_aggregate)
                else {}
            )

        return OfficialCFlatGradientHelper(
            model=self.model,
            rho=self.cflat_rho,
            lam=self.cflat_lam,
            target_params=target_params,
            transforms=transforms,
            project_before_perturb=self.project_before_perturb,
            project_after_aggregate=self.project_after_aggregate,
        )
