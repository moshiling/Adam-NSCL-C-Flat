import contextlib
from collections import defaultdict

import torch
from torch.distributed import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class OfficialCFlatGradientHelper:
    def __init__(self, model, optimizer, rho=0.2, lamb=0.2, adaptive=False,
                 perturb_eps=1e-12, bn_mode='disable_running_stats',
                 grad_reduce='mean', target_params=None,
                 projection_transforms=None, project_before_perturb=False,
                 project_after_aggregate=False):
        self.model = model
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.rho = rho
        self.lamb = lamb
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.bn_mode = bn_mode
        self.target_param_ids = (
            {id(param) for param in target_params} if target_params is not None else None
        )
        self.projection_transforms = projection_transforms or {}
        self.project_before_perturb = project_before_perturb
        self.project_after_aggregate = project_after_aggregate
        self._configure_grad_reduce(grad_reduce)

    def _configure_grad_reduce(self, grad_reduce):
        grad_reduce = grad_reduce.lower()
        if grad_reduce == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError(
                '"grad_reduce" should be one of ["mean", "sum"].')

    def _iter_params(self):
        for group in self.param_groups:
            for param in group['params']:
                yield param

    def _is_target_param(self, param):
        if self.target_param_ids is None:
            return True
        return id(param) in self.target_param_ids

    @torch.no_grad()
    def _project_tensor(self, param, tensor):
        transform = self.projection_transforms.get(param)
        if transform is None or tensor is None:
            return tensor
        if tensor.ndim == 4:
            flat = tensor.view(tensor.size(0), -1)
            if flat.size(1) != transform.size(0):
                return tensor
            return torch.mm(flat, transform).view_as(tensor)
        if tensor.ndim == 2:
            if tensor.size(1) != transform.size(0):
                return tensor
            return torch.mm(tensor, transform)
        return tensor

    @torch.no_grad()
    def _prepare_direction(self, param, grad_tensor):
        if grad_tensor is None:
            return None
        if self.project_before_perturb and self._is_target_param(param):
            return self._project_tensor(param, grad_tensor)
        return grad_tensor

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        return contextlib.ExitStack()

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive=False, target_only=False):
        norms = []
        device = None
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                if target_only and not self._is_target_param(param):
                    continue
                device = param.grad.device
                if weight_adaptive:
                    norms.append(
                        (torch.abs(param.data) * param.grad).norm(p=2))
                else:
                    norms.append(param.grad.norm(p=2))
        if not norms:
            return torch.zeros((), device=device or torch.device('cpu'))
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def perturb_weights(self, perturb_idx):
        grad_norm = self._grad_norm(
            weight_adaptive=self.adaptive, target_only=True)
        scale = self.rho / (grad_norm + self.perturb_eps)

        if perturb_idx == 0:
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    if not self._is_target_param(param):
                        continue
                    param_state = self.state[param]
                    param_state['g_0'] = self._prepare_direction(
                        param, param.grad.detach().clone())
                    e_w = param_state['g_0'] * scale.to(param)
                    param.add_(e_w)
                    param_state['e_w_0'] = e_w.detach().clone()
        elif perturb_idx == 1:
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    if not self._is_target_param(param):
                        continue
                    param_state = self.state[param]
                    param_state['g_2'] = self._prepare_direction(
                        param, param.grad.detach().clone())
                    e_w = param_state['g_2'] * scale.to(param)
                    param.add_(e_w)
                    param_state['e_w_1_2'] = param_state['e_w_1_2'] + \
                        e_w.detach().clone()
        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                if not self._is_target_param(param):
                    continue
                param_state = self.state[param]
                param_state['g_1'] = self._prepare_direction(
                    param, param.grad.detach().clone())
                param.grad.data = param_state['g_1'] - param_state['g_0']

        grad_norm = self._grad_norm(
            weight_adaptive=self.adaptive, target_only=True)
        scale = self.rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                if not self._is_target_param(param):
                    continue
                e_w = param.grad * scale.to(param)
                if self.adaptive:
                    e_w *= torch.pow(param, 2)
                param.add_(e_w)
                self.state[param]['e_w_1_2'] = e_w.detach().clone()

    @torch.no_grad()
    def unperturb(self, perturb_key):
        for group in self.param_groups:
            for param in group['params']:
                if perturb_key in self.state[param]:
                    param.data.sub_(self.state[param][perturb_key])

    @torch.no_grad()
    def gradient_aggregation(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                if not self._is_target_param(param):
                    continue
                param_state = self.state[param]
                current_grad = self._prepare_direction(
                    param, param.grad.detach().clone())
                final_grad = param_state['g_1'] + self.lamb * (
                    current_grad - param_state['g_2']
                )
                if self.project_after_aggregate:
                    final_grad = self._project_tensor(param, final_grad)
                param.grad.data = final_grad

    @torch.no_grad()
    def set_final_grad_from_g1(self):
        for group in self.param_groups:
            for param in group['params']:
                if not self._is_target_param(param):
                    continue
                param_state = self.state[param]
                if 'g_1' not in param_state:
                    continue
                final_grad = param_state['g_1']
                if self.project_after_aggregate:
                    final_grad = self._project_tensor(param, final_grad)
                if param.grad is None:
                    param.grad = final_grad.clone()
                else:
                    param.grad.data = final_grad

    @torch.no_grad()
    def _sync_grad(self):
        if not torch.distributed.is_initialized():
            return
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                if self.manual_average:
                    torch.distributed.all_reduce(param.grad, op=self.grad_reduce)
                    param.grad.div_(float(torch.distributed.get_world_size()))
                else:
                    torch.distributed.all_reduce(param.grad, op=self.grad_reduce)

    def _collect_debug_stats(self):
        def stacked_norm(key):
            tensors = [
                param_state[key].norm(p=2)
                for param_state in self.state.values()
                if key in param_state
            ]
            if not tensors:
                return 0.0
            return torch.norm(torch.stack(tensors), p=2).item()

        current_grad_norm = self._grad_norm(
            weight_adaptive=False, target_only=False)
        projected_raw_norm = stacked_norm('g_0')
        target_numel = 0
        total_numel = 0
        for param in self._iter_params():
            total_numel += param.numel()
            if self._is_target_param(param):
                target_numel += param.numel()
        return {
            'raw_grad_norm': stacked_norm('raw_g_0'),
            'g_0_norm': stacked_norm('g_0'),
            'g_1_norm': stacked_norm('g_1'),
            'g_2_norm': stacked_norm('g_2'),
            'final_grad_norm': current_grad_norm.item(),
            'e_w_0_norm': stacked_norm('e_w_0'),
            'e_w_1_2_norm': stacked_norm('e_w_1_2'),
            'projected_target_grad_norm': projected_raw_norm,
            'target_param_numel': float(target_numel),
            'total_param_numel': float(total_numel),
            'target_param_ratio': float(target_numel) / float(total_numel or 1),
            'project_before_perturb': float(self.project_before_perturb),
            'project_after_aggregate': float(self.project_after_aggregate),
        }

    def run(self, closure, cflat=True):
        def get_grad():
            self.optimizer.zero_grad()
            with torch.enable_grad():
                outputs, loss_list = closure()
                total_loss = sum(loss_list)
            total_loss.backward()
            return outputs, loss_list

        with self.maybe_no_sync():
            outputs, loss_list = get_grad()
            for param in self._iter_params():
                if param.grad is None:
                    continue
                self.state[param]['raw_g_0'] = param.grad.detach().clone()

            if cflat:
                bn_disabled = self.bn_mode == 'disable_running_stats'
                perturb0_active = False
                perturb12_active = False
                try:
                    self.perturb_weights(perturb_idx=0)
                    perturb0_active = True
                    if bn_disabled:
                        disable_running_stats(self.model)
                    get_grad()
                    self.unperturb(perturb_key='e_w_0')
                    perturb0_active = False
                    self.grad_norm_ascent()
                    perturb12_active = True
                    if abs(float(self.lamb)) <= self.perturb_eps:
                        self.set_final_grad_from_g1()
                    else:
                        get_grad()
                        self.perturb_weights(perturb_idx=1)
                        get_grad()
                        self.gradient_aggregation()
                    self.unperturb(perturb_key='e_w_1_2')
                    perturb12_active = False
                finally:
                    if perturb12_active:
                        self.unperturb(perturb_key='e_w_1_2')
                    if perturb0_active:
                        self.unperturb(perturb_key='e_w_0')
                    if bn_disabled:
                        enable_running_stats(self.model)
            else:
                enable_running_stats(self.model)

        for param in self._iter_params():
            if not self._is_target_param(param):
                if 'raw_g_0' in self.state[param]:
                    param.grad = self.state[param]['raw_g_0'].clone()

        self._sync_grad()
        return outputs, loss_list, self._collect_debug_stats()


def apply_cflat_gradients(model, optimizer, closure, cflat=True, rho=0.2,
                          lamb=0.2, adaptive=False, perturb_eps=1e-12,
                          bn_mode='disable_running_stats', grad_reduce='mean',
                          target_params=None, projection_transforms=None,
                          project_before_perturb=False,
                          project_after_aggregate=False):
    helper = OfficialCFlatGradientHelper(
        model=model,
        optimizer=optimizer,
        rho=rho,
        lamb=lamb,
        adaptive=adaptive,
        perturb_eps=perturb_eps,
        bn_mode=bn_mode,
        grad_reduce=grad_reduce,
        target_params=target_params,
        projection_transforms=projection_transforms,
        project_before_perturb=project_before_perturb,
        project_after_aggregate=project_after_aggregate,
    )
    return helper.run(closure=closure, cflat=cflat)
