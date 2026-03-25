# PLS-CFlat (D2-Based) Summary

## 1. This round's goal

This round did **not** continue the high-strength direct-attach C-Flat line.
It started from the repaired, conservative D2 setting:

- `cflat_rho=0.02`
- `cflat_lambda=0.02`

and tested whether a more Adam-NSCL-aware design can improve **BWT** under comparable accuracy:

- V1: Layer-Selective only
- V2: Projected all
- V3: Projected + Layer-Selective

The main comparison target in this round is therefore:

- baseline: original Adam-NSCL
- D2: strongest repaired direct-attach baseline

## 2. Code changes

Modified files:

- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/optim/cflat_utils.py`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/svd_agent/agent.py`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/svd_agent/svd_agent.py`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py`

What changed:

- Added `use_pls_cflat` branch while keeping both original baseline and repaired direct-attach `use_cflat=True` path intact.
- Added `pls_cflat_mode` with:
  - `projected_all`
  - `layer_selective`
  - `projected_layer_selective`
- Added `cflat_target_scope` with:
  - `all`
  - `classifier`
  - `deep`
  - `deep_plus_classifier`
- Added projection controls:
  - `project_before_perturb`
  - `project_after_aggregate`
- Added deep-layer rule:
  - `deep_layer_rule`
- Added debug switch:
  - `pls_cflat_debug`
- Changed default C-Flat strength to the D2 line:
  - `--cflat_rho 0.02`
  - `--cflat_lambda 0.02`

## 3. Layer-Selective rule

Implemented in `svd_agent/svd_agent.py`.

For the current ResNet setup:

- `classifier` scope selects active `last.*` head parameters.
- `deep` with `last_stage` selects the highest `stageN` plus `bn_last`.
- `deep_plus_classifier` is the union of:
  - final stage
  - final BN
  - current classifier head

Fallback behavior:

- if stage naming is irregular, `last_third` uses the last third of active non-classifier parameters by optimizer order.

For the formal V1 and V3 runs:

- target scope: `deep_plus_classifier`
- selected parameter ratio: `0.7517`
- selected numel: `8,398,346 / 11,172,170`

Evidence from logs:

- V1/V3 scope line reports `target_numel: 8398346 | total_numel: 11172170 | ratio: 0.7517`

## 4. Projected implementation

This round used a conservative P1-style implementation rather than inventing a new optimizer.

Implementation choice:

- Read the existing Adam-NSCL transform matrix from `optimizer.transforms[p]`.
- Reuse the same tensor geometry as `optim/adam_svd.py`:
  - conv weights: flatten to `(out, -1)`, right-multiply by transform, then reshape back
  - linear weights: `grad @ transform`
- Apply projection:
  - before perturbation direction construction when `project_before_perturb=True`
  - after official C-Flat gradient aggregation when `project_after_aggregate=True`

This keeps PLS-CFlat aligned with Adam-NSCL at the level of the **same transform family** already used during constrained updates, while still letting Adam-NSCL perform the final optimizer step.

Important limitation:

- this is an engineering approximation to Adam-NSCL's allowed update space
- it is not a full theoretical re-derivation of the optimizer's final constrained step

## 5. Static and numerical checks

Completed checks:

- baseline still runs
- repaired direct-attach D2 still runs
- V1, V2, V3 all run end-to-end
- no BN/stat restoration issue observed
- no parameter restoration issue observed
- no NaN / empty grad issue observed

Debug evidence:

- V2 projection becomes active after earlier tasks:
  - `transform_count: 20`
- V3 projection becomes active after earlier tasks:
  - `transform_count: 5`
- projected norms are meaningfully smaller than raw norms once transforms exist

Examples:

- V2 sample:
  - `raw_grad: 16.913279 -> projected_target: 3.672632`
- V3 sample:
  - `raw_grad: 31.136131 -> projected_target: 3.854398`

## 6. Formal experiment commands

Common setup:

- dataset: CIFAR100 10-split
- seed: 0
- tasks: 3
- schedule: `5 10 15`
- same Adam-NSCL training hyperparameters as the prior formal comparison

Baseline:

```bash
python main.py --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment/formal_3task_15ep/baseline
```

D2 direct-attach baseline:

```bash
python main.py --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 --use_cflat --cflat_rho 0.02 --cflat_lambda 0.02 --cflat_debug --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d2_r002_l002
```

V1:

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u /home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 --use_pls_cflat --pls_cflat_mode layer_selective --cflat_target_scope deep_plus_classifier --deep_layer_rule last_stage --cflat_rho 0.02 --cflat_lambda 0.02 --pls_cflat_debug --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/pls_cflat_d2_experiments/formal/v1_layer_selective
```

V2:

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u /home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 --use_pls_cflat --pls_cflat_mode projected_all --cflat_target_scope all --deep_layer_rule last_stage --project_before_perturb --project_after_aggregate --cflat_rho 0.02 --cflat_lambda 0.02 --pls_cflat_debug --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/pls_cflat_d2_experiments/formal/v2_projected_all
```

V3:

```bash
CUDA_VISIBLE_DEVICES=3 /home/moshiling/miniconda3/envs/torch118/bin/python -u /home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/main.py --schedule 5 10 15 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid 0 --repeat 1 --model_optimizer Adam --force_out_dim 0 --first_split_size 10 --other_split_size 10 --batch_size 32 --model_name resnet18 --model_type resnet --workers 0 --print_freq 100 --max_tasks 3 --seed 0 --use_pls_cflat --pls_cflat_mode projected_layer_selective --cflat_target_scope deep_plus_classifier --deep_layer_rule last_stage --project_before_perturb --project_after_aggregate --cflat_rho 0.02 --cflat_lambda 0.02 --pls_cflat_debug --output_dir /home/moshiling/Adam-NSCL-C-Flat/outputs/pls_cflat_d2_experiments/formal/v3_projected_layer_selective
```

## 7. Result table

| Method | Key setting | Acc | BWT | Time (s) | Delta vs D2 |
|---|---|---:|---:|---:|---|
| Baseline | Adam-NSCL | 66.73 | -0.95 | 359.13 | reference |
| D2 direct-attach | all params, direct attach, `0.02/0.02` | 68.07 | -1.35 | 859.29 | strongest direct baseline |
| V1 Layer-Selective | `deep_plus_classifier`, no projection | 66.77 | **-0.75** | 1527.06 | BWT +0.60, Acc -1.30 |
| V2 Projected all | all params + projection | **68.17** | -1.75 | 2135.59 | Acc +0.10, BWT -0.40 |
| V3 Projected + Selective | `deep_plus_classifier` + projection | 66.77 | -1.15 | 962.25 | BWT +0.20, Acc -1.30 |

Notes:

- V1 beats both D2 and baseline on BWT.
- V2 has the best accuracy in this round, but the worst BWT among the new variants.
- V3 improves BWT over D2 while keeping runtime much lower than V1 and far lower than V2, but it does not beat V1 on forgetting.

## 8. Interpretation

### 8.1 Which variant is most worth continuing?

**V1 Layer-Selective** is the best next-step candidate.

Why:

- It gives the strongest BWT result of the whole table: `-0.75`
- It improves over baseline BWT: `-0.95 -> -0.75`
- It improves over D2 BWT by a clear margin: `-1.35 -> -0.75`
- It validates the main structural idea:
  - indiscriminate flatness over all layers is not necessary
  - limiting C-Flat to more plastic layers is much more compatible with Adam-NSCL's stability objectives

### 8.2 Did the projected idea help?

Only partially.

- V2 `projected_all` is a negative signal for the projected idea as a main line:
  - accuracy is fine
  - BWT gets worse than D2
  - runtime becomes the worst of all methods
- V3 `projected_layer_selective` is better than D2 on BWT:
  - `-1.35 -> -1.15`
  - runtime stays close to D2
- But V3 still does **not** beat V1:
  - same final accuracy as V1
  - worse BWT than V1

So the current evidence says:

- projection alone is not enough
- projection over the whole model is likely misaligned with CL retention
- projection may still be useful as a secondary refinement on top of selective targeting, but it is not yet the leading innovation signal

### 8.3 Is this more like a paper-worthy innovation than a direct optimizer attach?

Yes, but only for the **Layer-Selective** direction so far.

The strongest evidence from this round is not "project everything into Adam-NSCL geometry".
It is:

- C-Flat should **not** act uniformly across the network
- selective application to high-plasticity layers can recover or even improve BWT
- this is more method-specific to continual learning than a plain optimizer add-on

## 9. Final judgment

### Recommended main line

Continue with:

- **Layer-Selective C-Flat**, centered on `deep_plus_classifier`

Optional secondary line:

- a lighter projected refinement on top of the selective line

### Not recommended as the main line

Do **not** continue with:

- `projected_all`

Reason:

- no BWT benefit
- runtime cost is too high
- it behaves more like over-constraining the whole network than aligning with Adam-NSCL's retention geometry

## 10. Most likely explanation if results are still not ideal

The current evidence suggests:

1. Direct flatness pressure over all layers conflicts with CL retention, even when projection is added.
2. Adam-NSCL already applies a strong geometric constraint at optimizer step time, so adding another global geometric bias during gradient generation can become redundant or harmful.
3. The useful signal seems to come from **where** flatness is applied, more than from making all-layer flatness more geometrically faithful.

## 11. Output locations

Formal results:

- baseline:
  - `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_official_alignment/formal_3task_15ep/baseline/final_metrics.json`
- D2:
  - `/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d2_r002_l002/final_metrics.json`
- V1:
  - `/home/moshiling/Adam-NSCL-C-Flat/outputs/pls_cflat_d2_experiments/formal/v1_layer_selective/final_metrics.json`
- V2:
  - `/home/moshiling/Adam-NSCL-C-Flat/outputs/pls_cflat_d2_experiments/formal/v2_projected_all/final_metrics.json`
- V3:
  - `/home/moshiling/Adam-NSCL-C-Flat/outputs/pls_cflat_d2_experiments/formal/v3_projected_layer_selective/final_metrics.json`

Logs and commands:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/pls_cflat_d2_experiments/`
