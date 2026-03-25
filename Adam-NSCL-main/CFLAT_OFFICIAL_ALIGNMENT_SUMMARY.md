# C-Flat Official Alignment Summary

## Goal

- Align the Adam-NSCL C-Flat integration with the official `/home/moshiling/papers/C-Flat/optims/c_flat.py`.
- Keep Adam-NSCL's original null-space / SVD / custom Adam update path.
- Restrict C-Flat to gradient generation only, then feed the final `p.grad` into the original Adam-NSCL optimizer step.

## Official Step Alignment

The repaired helper now follows the official `C_Flat.step()` order directly:

1. `get_grad()`
2. `perturb_weights(perturb_idx=0)`
3. `disable_running_stats(model)`
4. `get_grad()`
5. `unperturb("e_w_0")`
6. `grad_norm_ascent()`
7. `get_grad()`
8. `perturb_weights(perturb_idx=1)`
9. `get_grad()`
10. `gradient_aggregation()`
11. `unperturb("e_w_1_2")`
12. `_sync_grad()` compatibility kept as a no-op unless distributed is initialized
13. outer Adam-NSCL `optimizer.step()`
14. `enable_running_stats(model)`

## What Was Wrong Before

- The previous `optim/cflat_utils.py` used a custom `delta_grads / second_point_grads / final_point_grads / first_order_grads` formulation that did not match the official `g_0 / g_1 / g_2 / gradient_aggregation()` semantics.
- The previous integration mixed Adam-NSCL regularization into the C-Flat closure by default, which deviated from the requested repair direction and made the coupling harder to interpret.
- The previous helper did not mirror the official naming and update order closely enough, which made verification against the official file difficult.

## What Was Changed

- `optim/cflat_utils.py`
  Rewritten around an `OfficialCFlatGradientHelper` whose methods directly mirror the official code:
  `perturb_weights`, `grad_norm_ascent`, `unperturb`, `gradient_aggregation`, `_grad_norm`, `_sync_grad`.
- `svd_agent/agent.py`
  Reworked the training loop so `use_cflat=True` does not do an extra normal `loss.backward()` before the helper.
  The helper now owns the C-Flat forward/backward sequence.
  Metrics are updated once per batch.
- `svd_agent/svd_agent.py`
  `reg_loss()` now accepts `log=True/False` so repeated closure calls can avoid repeatedly logging the same regularizer when needed.
- `main.py`
  Added `--cflat_on_total_loss` and passed it into the agent config.

## Why The New Version Is Closer To Official

- The helper now stores and uses the exact official state variables:
  `g_0`, `g_1`, `g_2`, `e_w_0`, `e_w_1_2`.
- `grad_norm_ascent()` now matches the official semantics:
  save current grad as `g_1`,
  do `p.grad -= g_0`,
  compute the new norm,
  build `e_w_1_2`.
- `gradient_aggregation()` now matches the official formula exactly:
  `p.grad = g_1 + lambda * (current_grad - g_2)`
- BatchNorm running stats are disabled only during the perturbed phase and restored afterward.
- Parameters are restored even on exceptional paths.

## Loss Coupling Choice

- Default repaired behavior:
  C-Flat is applied to the main task classification loss only.
- Adam-NSCL regularization is added afterward through the original gradient path at the current weights.
- Optional switch:
  `--cflat_on_total_loss`
- Default is `False`.

This choice was made because the user explicitly requested that the default repair path should avoid mixing the regularizer into the C-Flat closure.

## Files Modified

- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/optim/cflat_utils.py`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/svd_agent/agent.py`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/svd_agent/svd_agent.py`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py`

## Static Checks

- Syntax/import:
  `/home/moshiling/miniconda3/envs/torch118/bin/python -m py_compile main.py svd_agent/agent.py svd_agent/svd_agent.py optim/cflat_utils.py optim/adam_svd.py`
- Arg parsing:
  `main.py --help`
- Baseline startup:
  passed
- Repaired C-Flat startup:
  passed
- One-batch dry run:
  passed

## Sanity Check Stats

Example debug lines from the repaired implementation:

- `raw_grad: 2.806322 | g_0: 2.806322 | g_1: 2.854342 | g_2: 3.062590 | final: 2.869094 | e_w_0: 0.050000 | e_w_1_2: 0.067420`
- `raw_grad: 2.437734 | g_0: 2.437734 | g_1: 2.459922 | g_2: 2.510628 | final: 2.465071 | e_w_0: 0.050000 | e_w_1_2: 0.068683`

Observed properties:

- final `p.grad` stayed non-zero and numerically stable
- perturb norms matched the configured `rho`
- BN stats did not show evidence of perturbed-phase contamination
- task transitions, SVD updates, and post-task evaluation all remained functional

## Experiments Run

Output root:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment`

### Smoke setting

- CIFAR100 10-split
- first 2 tasks
- 1 epoch
- batch size 128
- seed 0

### Formal minimum setting

- CIFAR100 10-split
- first 3 tasks
- 15 epochs via `--schedule 5 10 15`
- batch size 32
- seed 0

## Commands

### Smoke baseline

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u main.py \
  --schedule 1 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 \
  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based \
  --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 \
  --first_split_size 10 --other_split_size 10 --batch_size 128 --model_name resnet18 \
  --model_type resnet --workers 0 --print_freq 20 --max_tasks 2 --seed 0 \
  --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment/smoke/baseline
```

### Smoke repaired C-Flat

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u main.py \
  --schedule 1 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 \
  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based \
  --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 \
  --first_split_size 10 --other_split_size 10 --batch_size 128 --model_name resnet18 \
  --model_type resnet --workers 0 --print_freq 20 --max_tasks 2 --seed 0 \
  --use_cflat --cflat_rho 0.05 --cflat_lambda 0.05 \
  --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment/smoke/cflat
```

### Formal minimum baseline

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u main.py \
  --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 \
  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based \
  --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 \
  --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 \
  --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 \
  --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment/formal_3task_15ep/baseline
```

### Formal minimum repaired C-Flat

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u main.py \
  --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 \
  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based \
  --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 \
  --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 \
  --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 \
  --use_cflat --cflat_rho 0.05 --cflat_lambda 0.05 \
  --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment/formal_3task_15ep/cflat
```

## Results

### Smoke comparison

| Method | Final Avg Acc | Final BWT | Wall Clock |
|---|---:|---:|---:|
| Baseline | 39.75 | 7.00 | 91.69s |
| Repaired C-Flat | 37.00 | -3.40 | 88.13s |

### Formal minimum comparison

| Method | Final Avg Acc | Final BWT | Wall Clock |
|---|---:|---:|---:|
| Baseline | 66.73 | -0.95 | 359.13s |
| Repaired C-Flat | 66.90 | -2.10 | 811.32s |

## Comparison Against The Previous Broken Integration

Previous integrated C-Flat results from the earlier summary:

- old smoke C-Flat:
  `acc=28.90`, `bwt=-3.20`
- old formal-minimum C-Flat:
  `acc=54.47`, `bwt=-1.95`, `time=952.47s`

Repaired official-aligned C-Flat:

- new smoke C-Flat:
  `acc=37.00`, `bwt=-3.40`
- new formal-minimum C-Flat:
  `acc=66.90`, `bwt=-2.10`, `time=811.32s`

Observed change:

- Smoke avg acc improved by `+8.10`
- Formal-minimum avg acc improved by `+12.43`
- Formal-minimum runtime decreased by about `141.15s`
- BWT is still worse than baseline

## Conclusion

- `use_cflat=False` baseline:
  unchanged in behavior.
- Repaired C-Flat vs official implementation:
  much closer semantically.
- Repaired C-Flat vs previous broken integration:
  clearly more reasonable.
- Repaired C-Flat vs baseline:
  mixed.

Final judgment:

- Engineering correctness:
  valid.
- Official semantic alignment:
  substantially improved.
- Evidence that the repair fixed the abnormal previous integration:
  yes.
- Evidence that C-Flat is now a better method than baseline in this coupled Adam-NSCL setting:
  no.

The repaired version no longer shows the previous "obviously wrong" degradation pattern, and its final average accuracy is now essentially back to the baseline level on the 3-task formal minimum run. However, BWT remains worse and runtime is still much higher. That points more toward method-coupling limitations in this Adam-NSCL setting than toward a remaining obvious implementation bug in the C-Flat step semantics.

## Next Suggestions

- Test `cflat_lambda=0` as a g0-only ablation under the repaired implementation.
- Sweep smaller `rho` and `lambda` around the repaired default validation point, especially `0.02`, `0.05`, `0.1`.
- If more GPU time is available, extend the repaired comparison from 3 tasks to all 10 tasks before making a stronger method conclusion.
