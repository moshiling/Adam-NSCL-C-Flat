from .agent import Agent
import optim
import torch
import re
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F


class SVDAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.fea_in_hook = {}
        self.fea_in = defaultdict(dict)
        self.fea_in_count = defaultdict(int)

        self.drop_num = 0

        self.regularization_terms = {}
        self.reg_params = {n: p for n,
                           p in self.model.named_parameters() if 'bn' in n}
        self.empFI = False
        self.svd_lr = self.config['model_lr']  # first task
        self.init_model_optimizer()

        self.params_json = {p: n for n, p in self.model.named_parameters()}
        self._scope_cache = {}

    
    def init_model_optimizer(self):
        fea_params = [p for n, p in self.model.named_parameters(
        ) if not bool(re.match('last', n)) and 'bn' not in n]
        cls_params_all = list(
            p for n, p in self.model.named_children() if bool(re.match('last', n)))[0]
        cls_params = list(cls_params_all[str(self.task_count+1)].parameters())
        bn_params = [p for n, p in self.model.named_parameters() if 'bn' in n]
        model_optimizer_arg = {'params': [{'params': fea_params, 'svd': True, 'lr': self.svd_lr,
                                            'thres': self.config['svd_thres']},
                                          {'params': cls_params, 'weight_decay': 0.0,
                                              'lr': self.config['head_lr']},
                                          {'params': bn_params, 'lr': self.config['bn_lr']}],
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad']:
            if self.config['model_optimizer'] == 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(
            optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=self.config['gamma'])

    def train_task(self, train_loader, val_loader=None):
        # 1.Learn the parameters for current task
        self.train_model(train_loader, val_loader)
        self.task_count += 1
        if self.task_count < self.num_task or self.num_task is None:
            if self.reset_model_optimizer:  # Reset model optimizer before learning each task
                self.log('Classifier Optimizer is reset!')
                self.svd_lr = self.config['svd_lr']
                self.init_model_optimizer()
                self.model.zero_grad()
                
            with torch.no_grad():
                # end = time.time()
                self.update_optim_transforms(train_loader)
                # print('update trans: {}'.format(time.time() - end))

            if self.reg_params:
                if len(self.regularization_terms) == 0:
                    self.regularization_terms = {'importance': defaultdict(
                        list), 'task_param': defaultdict(list)}
                importance = self.calculate_importance(train_loader)
                for n, p in self.reg_params.items():
                    self.regularization_terms['importance'][n].append(
                        importance[n].unsqueeze(0))
                    self.regularization_terms['task_param'][n].append(
                        p.unsqueeze(0).clone().detach())
            # Use a new slot to store the task-specific information

    def update_optim_transforms(self, train_loader):
        modules = [m for n, m in self.model.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        handles = []
        for m in modules:
            handles.append(m.register_forward_hook(hook=self.compute_cov))

        
        for i, (inputs, target, task) in enumerate(train_loader):
            if self.config['gpu']:
                inputs = inputs.cuda()
            self.model.forward(inputs)
            
        self.model_optimizer.get_eigens(self.fea_in)
        

        self.model_optimizer.get_transforms()
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

    def calculate_importance(self, dataloader):
        self.log('computing EWC')
        importance = {}
        for n, p in self.reg_params.items():
            importance[n] = p.clone().detach().fill_(0)

        mode = self.model.training
        self.model.eval()
        for _, (inputs, targets, task) in enumerate(dataloader):
            if self.config['gpu']:
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = self.model.forward(inputs)

            if self.empFI:
                ind = targets
            else:
                task_name = task[0] if self.multihead else 'ALL'
                pred = output[task_name] if not isinstance(self.valid_out_dim, int) else output[task_name][:,
                                                                                                           :self.valid_out_dim]
                ind = pred.max(1)[1].flatten()

            loss = self.criterion(output, ind, task, regularization=False)
            self.model.zero_grad()
            loss.backward()

            for n, p in importance.items():
                if self.reg_params[n].grad is not None:
                    p += ((self.reg_params[n].grad ** 2)
                          * len(inputs) / len(dataloader))

        return importance

    def reg_loss(self, log=True):
        if log:
            self.reg_step += 1
        reg_loss = 0
        for n, p in self.reg_params.items():
            importance = torch.cat(
                self.regularization_terms['importance'][n], dim=0)
            old_params = torch.cat(
                self.regularization_terms['task_param'][n], dim=0)
            new_params = p.unsqueeze(0).expand(old_params.shape)
            reg_loss += (importance * (new_params - old_params) ** 2).sum()

        if log:
            self.summarywritter.add_scalar(
                'reg_loss', reg_loss, self.reg_step)
        return reg_loss

    def _active_optimizer_named_params(self):
        active_param_ids = {
            id(param)
            for group in self.model_optimizer.param_groups
            for param in group['params']
        }
        return [
            (name, param)
            for name, param in self.model.named_parameters()
            if id(param) in active_param_ids
        ]

    def _resolve_deep_named_params(self, named_active_params, deep_layer_rule):
        classifier_prefix = 'last.'
        feature_params = [
            (name, param)
            for name, param in named_active_params
            if not name.startswith(classifier_prefix)
        ]
        if not feature_params:
            return []

        if deep_layer_rule == 'last_third':
            start_index = max(0, (2 * len(feature_params)) // 3)
            return feature_params[start_index:]

        stage_ids = []
        for name, _ in feature_params:
            match = re.match(r'^stage(\d+)\.', name)
            if match:
                stage_ids.append(int(match.group(1)))
        if not stage_ids:
            start_index = max(0, (2 * len(feature_params)) // 3)
            return feature_params[start_index:]

        last_stage = max(stage_ids)
        last_stage_prefix = f'stage{last_stage}.'
        last_stage_params = [
            (name, param)
            for name, param in feature_params
            if name.startswith(last_stage_prefix) or name.startswith('bn_last')
        ]
        if deep_layer_rule == 'last_stage':
            return last_stage_params

        if deep_layer_rule == 'last_block':
            block_ids = []
            for name, _ in last_stage_params:
                match = re.match(r'^stage{}\.(\d+)\.'.format(last_stage), name)
                if match:
                    block_ids.append(int(match.group(1)))
            if block_ids:
                last_block = max(block_ids)
                block_prefix = f'stage{last_stage}.{last_block}.'
                block_params = [
                    (name, param)
                    for name, param in last_stage_params
                    if name.startswith(block_prefix) or name.startswith('bn_last')
                ]
                if block_params:
                    return block_params

        start_index = max(0, (2 * len(feature_params)) // 3)
        return feature_params[start_index:]

    def get_cflat_scope_info(self, scope='all', deep_layer_rule='last_stage'):
        cache_key = (
            self.task_count,
            scope,
            deep_layer_rule,
            tuple(len(group['params']) for group in self.model_optimizer.param_groups),
        )
        if cache_key in self._scope_cache:
            return self._scope_cache[cache_key]

        named_active_params = self._active_optimizer_named_params()
        classifier_named_params = [
            (name, param)
            for name, param in named_active_params
            if name.startswith('last.')
        ]
        deep_named_params = self._resolve_deep_named_params(
            named_active_params, deep_layer_rule)

        named_active_map = {name: param for name, param in named_active_params}
        target_named_map = {}

        if scope == 'all':
            target_named_map = dict(named_active_params)
        elif scope == 'classifier':
            target_named_map = dict(classifier_named_params)
        elif scope == 'deep':
            target_named_map = dict(deep_named_params)
        elif scope == 'deep_plus_classifier':
            target_named_map = dict(deep_named_params)
            target_named_map.update(dict(classifier_named_params))
        else:
            raise ValueError(f'Unsupported cflat target scope: {scope}')

        total_numel = sum(param.numel() for _, param in named_active_params)
        target_numel = sum(param.numel() for param in target_named_map.values())
        scope_info = {
            'named_active_params': named_active_params,
            'named_target_params': list(target_named_map.items()),
            'target_params': set(target_named_map.values()),
            'total_numel': total_numel,
            'target_numel': target_numel,
            'target_ratio': float(target_numel) / float(total_numel or 1),
            'deep_layer_rule': deep_layer_rule,
            'scope': scope,
            'classifier_param_names': [name for name, _ in classifier_named_params],
            'deep_param_names': [name for name, _ in deep_named_params],
            'active_param_names': list(named_active_map.keys()),
        }
        self._scope_cache[cache_key] = scope_info
        return scope_info

    def get_projection_transforms(self, target_params=None):
        projection_map = {}
        target_param_ids = None
        if target_params is not None:
            target_param_ids = {id(param) for param in target_params}
        for param, transform in self.model_optimizer.transforms.items():
            if target_param_ids is not None and id(param) not in target_param_ids:
                continue
            projection_map[param] = transform
        return projection_map
