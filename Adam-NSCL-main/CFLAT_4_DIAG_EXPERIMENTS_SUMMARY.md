# C-Flat 4 Diagnostic Experiments Summary

## Goal

Based on the current repaired, officially aligned C-Flat integration, run 4 small diagnostic experiments to distinguish between these hypotheses:

1. `g1` is the main problem in Adam-NSCL.
2. `rho` / `lambda` are too large.
3. The repaired implementation is correct, but direct C-Flat coupling is only weakly compatible with Adam-NSCL.

Code base used for all runs:

- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main`

Environment:

- `torch118`

GPU selection:

- Used a single mostly idle card: physical `GPU 3`
- Commands were run with `CUDA_VISIBLE_DEVICES=3` and `--gpuid 0`

Output root:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp`

## Pre-Run Checks

- Argument parsing already supported:
  - `--use_cflat`
  - `--cflat_rho`
  - `--cflat_lambda`
  - `--cflat_debug`
- `lambda=0` dry run passed.
- `cflat_debug=True` logs confirmed these fields:
  - `raw_grad`
  - `g_0`
  - `g_1`
  - `g_2`
  - `final`
  - `e_w_0`
  - `e_w_1_2`

Short `lambda=0` dry-run evidence:

- `final_grad_norm` collapsed to the `g_1` norm as expected under `lambda=0`
- perturbation and gradient magnitudes were finite
- no NaN, no autograd breakage, no startup issues

Static check output:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/static_d1_lambda0/run.log`

## Shared Experiment Setting

All 4 diagnostics kept the same formal minimum setup as the repaired formal comparison:

- CIFAR100 10-split
- first 3 tasks only
- `--schedule 5 10 15`
- batch size `32`
- seed `0`
- same optimizer / model / SVD / regularization settings as the repaired formal run

Only `cflat_rho` and `cflat_lambda` changed.

## Commands

### D1: g0-only

Directory:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d1_g0_only_r005`

Command:

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u /home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py \
  --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 \
  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based \
  --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 \
  --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 \
  --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 \
  --use_cflat --cflat_rho 0.05 --cflat_lambda 0 --cflat_debug \
  --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d1_g0_only_r005
```

### D2: smaller rho + smaller lambda

Directory:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d2_r002_l002`

Command:

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u /home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py \
  --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 \
  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based \
  --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 \
  --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 \
  --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 \
  --use_cflat --cflat_rho 0.02 --cflat_lambda 0.02 --cflat_debug \
  --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d2_r002_l002
```

### D3: same rho, smaller lambda

Directory:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d3_r005_l002`

Command:

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u /home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py \
  --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 \
  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based \
  --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 \
  --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 \
  --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 \
  --use_cflat --cflat_rho 0.05 --cflat_lambda 0.02 --cflat_debug \
  --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d3_r005_l002
```

### D4: repaired default conservative pair

Directory:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d4_r005_l005`

Command:

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u /home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py \
  --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 \
  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based \
  --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 \
  --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 \
  --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 \
  --use_cflat --cflat_rho 0.05 --cflat_lambda 0.05 --cflat_debug \
  --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d4_r005_l005
```

## Result Table

Reference runs:

- baseline from:
  `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment/formal_3task_15ep/baseline/final_metrics.json`
- prior repaired formal C-Flat from:
  `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment/formal_3task_15ep/cflat/final_metrics.json`

| Run | rho | lambda | Meaning | Final Avg Acc | Final BWT | Wall Clock |
|---|---:|---:|---|---:|---:|---:|
| Baseline | - | - | Adam-NSCL | 66.73 | -0.95 | 359.13s |
| Repaired formal previous | 0.05 | 0.05 | earlier repaired C-Flat reference | 66.90 | -2.10 | 811.32s |
| D1 | 0.05 | 0.00 | g0-only | 67.33 | -1.75 | 1029.58s |
| D2 | 0.02 | 0.02 | smaller rho + smaller lambda | 68.07 | -1.35 | 859.29s |
| D3 | 0.05 | 0.02 | same rho, smaller lambda | 66.93 | -1.70 | 1075.56s |
| D4 | 0.05 | 0.05 | repaired default re-check | 66.90 | -2.10 | 1426.05s |

## Interpretation

### 1. Is `g1` the main problem?

Evidence:

- D1 (`lambda=0`) improved over D4:
  - acc `67.33` vs `66.90`
  - bwt `-1.75` vs `-2.10`

Judgment:

- Yes, `g1` is at least part of the problem.
- But it is not the whole story, because D2 outperformed D1 even while keeping `g1` active.

### 2. Are `rho` / `lambda` too large?

Evidence:

- D2 (`0.02 / 0.02`) was the best of the 4 diagnostics:
  - acc `68.07`
  - bwt `-1.35`
- D3 (`0.05 / 0.02`) improved over D4, but clearly underperformed D2:
  - acc `66.93` vs `68.07`
  - bwt `-1.70` vs `-1.35`
- This isolates a strong `rho` effect:
  - lowering only `lambda` is not enough
  - lowering both `rho` and `lambda` helps most

Judgment:

- Yes, the repaired default `rho=0.05, lambda=0.05` is too aggressive for this Adam-NSCL setting.
- `rho` looks especially important.

### 3. Is the repaired C-Flat merely “correct but not adapted”?

Evidence:

- D4 exactly reproduced the previous repaired formal metrics:
  - previous repaired formal: `66.90 / -2.10`
  - D4: `66.90 / -2.10`
- That means the repaired implementation is stable and reproducible.
- Even the best diagnostic run, D2, still did not beat the baseline on BWT:
  - baseline bwt `-0.95`
  - D2 bwt `-1.35`

Judgment:

- Yes, this now looks more like a compatibility limitation than a remaining implementation bug.
- The direct “外挂 C-Flat before Adam-NSCL constraints” route can be made less harmful, but it still does not show clear continual-learning superiority over baseline in this 3-task test.

## Direct Answers To The Diagnostic Questions

### D1 (`g0-only`) better than full C-Flat?

- Yes.
- D1 outperformed D4 on both final avg acc and BWT.
- This suggests the first-order flatness part is not reliably helping Adam-NSCL in the current coupling.

### Do smaller `rho` / `lambda` improve BWT?

- Yes.
- D2 improved BWT from `-2.10` to `-1.35`.
- D3 improved BWT from `-2.10` to `-1.70`.
- The strongest improvement happened when both `rho` and `lambda` were reduced together.

### Is the current issue more like “`g1` mismatch” or “too much flatness strength”?

- Both matter, but the data points more strongly to overly strong flatness perturbation.
- Ranking by evidence:
  1. flatness strength too large
  2. `g1` is additionally somewhat mismatched

Reason:

- If `g1` were the sole issue, D1 should dominate D2.
- Instead, D2 beat D1.
- That means a mild `g1` can still be usable, as long as the perturbation scale is reduced.

### Is the direct repaired C-Flat route worth continuing as the main line?

- Not as the main line in its current form.
- It is worth a small, bounded follow-up around conservative settings.
- It is not worth heavy further investment on the aggressive `0.05 / 0.05` route.

## Recommendation

More worth continuing:

- a narrow conservative line around D2
- especially small `rho` and small `lambda`
- candidate region:
  - `rho` in `0.01` to `0.03`
  - `lambda` in `0.0` to `0.03`

Less worth continuing:

- the repaired default `rho=0.05, lambda=0.05`
- high-perturbation direct外挂 C-Flat as the primary method path

Practical next step if we continue:

- treat D2 as the temporary best candidate
- run one more confirmatory seed or a slightly longer task count
- only if D2 keeps its advantage should we consider further integration work

## Final Conclusion

The 4 diagnostics support the following overall conclusion:

- The repaired implementation is reproducible and no longer looks buggy.
- The current repaired default C-Flat setting is too aggressive for Adam-NSCL.
- `g1` is somewhat harmful in this coupling, but not categorically unusable.
- The stronger problem is that flatness strength, especially `rho`, is too large.
- Even after making it milder, direct C-Flat coupling still does not beat baseline on BWT in this minimum formal test.

Therefore:

- `D2 (rho=0.02, lambda=0.02)` is the most promising continuation line.
- `D4 (rho=0.05, lambda=0.05)` is not worth further investment as the main direction.
- The current evidence favors the statement:
  “implementation is correct, but direct C-Flat-to-Adam-NSCL coupling is only partially compatible, and gains are limited unless the flatness signal is made much more conservative.”
