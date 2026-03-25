# V1 Full Scope V2 Summary

## 1. Summary status

- Check time: `2026-03-25 Asia/Shanghai`
- Checked scope:
  - `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_a_v1_3seed`
  - `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_b_scope_v2`
- Verification basis:
  - `command.sh`
  - `config_resolved.json`
  - `run.log`
  - `final_metrics.json`
  - `queue_status.log`
  - `heartbeat.log`
  - live `ps`
  - live `nvidia-smi`

Current overall state:

- **Partially completed**
- Group B is complete enough for a stable ranking
- Group A `v1_repeat3` is still running, but 2 out of 3 repeats have already produced end-of-repeat summaries in `run.log`
- Those 2 repeats are already strong enough to support a stage-level judgment

## 2. Full protocol used

This round uses the repo's canonical full CIFAR100-10 protocol, selected from:

- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/our_results/rehearsal_org/cifar100-10/svd_epoch_80_bn32_lr1e-4_headlr1e-3_bnlr5e-4_svdlr5e-5_wdecay5e-5_regcoef_100_eigvec_gt_ada10_combine0.log`

Active settings:

- dataset: `CIFAR100`
- tasks: `10`
- split: `10 + 10`
- schedule: `30 60 80`
- batch size: `32`
- model: `resnet18`
- `model_lr 1e-4`
- `head_lr 1e-3`
- `svd_lr 5e-5`
- `bn_lr 5e-4`
- `svd_thres 10`
- `reg_coef 100`
- `model_weight_decay 5e-5`
- `workers 0`
- `print_freq 10`

## 3. Restart policy actually used

This codebase still does not provide a reliable native checkpoint/resume path for the full Adam-NSCL agent state.

So the practical policy is:

- completed runs are reused directly
- interrupted runs without `final_metrics.json` are restarted from the same run directory
- queue and heartbeat logs are used to track restarts and GPU migrations

For `v1_repeat3`, the main engineering failure was an incorrect working directory during restart. The corrected launch form is:

```bash
cd /home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main && ...
```

This preserves the original experiment definition while making `dataroot=../data` resolve correctly to `/home/moshiling/data`.

## 4. Current run table

| Run | Group | Status | final_metrics | Latest usable state | Next action |
|---|---|---|---|---|---|
| `baseline_repeat3` | A | Completed | Yes | 3-seed full result available | Reuse |
| `v1_repeat3` | A | In progress | No | 2 repeats finished in log; repeat 3 still running | Continue until final |
| `v2_projected_all_seed0` | B | Completed | Yes | Full result available | Reuse |
| `classifier_only_seed0` | B | Completed | Yes | Full result available | Reuse |
| `last_block_plus_classifier_seed0` | B | Completed | Yes | Full result available | Reuse |
| `deep_last_stage_seed0` | B | Completed | Yes | Full result available | Reuse |
| `deep_last_block_seed0` | B | Completed | Yes | Full result available | Reuse |

## 5. Group A results

### 5.1 Baseline full 3-seed result

Source:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_a_v1_3seed/baseline_repeat3/final_metrics.json`

Baseline mean/std:

- acc mean/std: `73.20 / 0.20`
- bwt mean/std: `-1.31 / 0.35`
- wall-clock mean/std: `7853.01s / 419.28s`

Per-seed baseline:

- seed 0: `73.37 / -1.64 / 8396.37s`
- seed 1: `73.30 / -0.82 / 7375.75s`
- seed 2: `72.92 / -1.47 / 7786.89s`

### 5.2 V1 original current status

Run directory:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_a_v1_3seed/v1_repeat3`

Configuration:

- mode: `layer_selective`
- scope: `deep_plus_classifier`
- deep rule: `last_stage`
- `repeat 3`, seeds `0/1/2`
- target ratio: `0.7517`
- `rho=0.02`
- `lambda=0.02`

Current run state:

- `final_metrics.json` is still missing
- the run is currently active
- the latest checked live process is the intended full command on GPU0
- the corrected cwd/data-path launch is working

Evidence that the old `task1 -> task2` engineering issue has been fixed:

- `Files already downloaded and verified` appears after the corrected restart
- later task transitions are visible
- `Classifier Optimizer is reset!` and `====================== 10 =======================` both appear in the current run log
- the latest visible training state has already advanced into task 10 and reached `Epoch:16`

### 5.3 V1 partial result already available

Even though `final_metrics.json` does not exist yet, `run.log` already contains two completed repeat summaries:

```text
The last avg acc of all repeats: [72.34 71.91  0.  ]
The last bwt of all repeats: [-2.28888889 -1.53333333  0.        ]
```

Source:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_a_v1_3seed/v1_repeat3/run.log`

This means:

- repeat 0: `72.34 / -2.29`
- repeat 1: `71.91 / -1.53`
- repeat 2: not finished yet

Current partial mean across the first two repeats:

- partial acc mean: `72.13`
- partial bwt mean: `-1.91`

Comparison against baseline mean:

- baseline acc mean: `73.20`
- baseline bwt mean: `-1.31`

This is already a strong negative signal for `V1 original`.

What repeat 3 would need in order for V1 to merely match the baseline mean:

- required repeat-3 acc: about `75.34`
- required repeat-3 bwt: about `-0.11`

That combination is highly unlikely under the current trend.

Stage-level judgment for V1:

- `V1 original` should no longer be treated as the leading main-line candidate
- before the 3rd repeat finishes, the fairest label is:
  - **弱继续**
- if repeat 3 follows the same range as repeats 0 and 1, the final label will likely become:
  - **停止推进**

## 6. Group B full results

| Run | Scope / mode | target_param_ratio | Acc | BWT | Time |
|---|---|---:|---:|---:|---:|
| `v2_projected_all_seed0` | projected all | 1.0000 | 74.20 | -2.80 | 19560.38s |
| `classifier_only_seed0` | classifier only | 0.0005 | 73.29 | -2.04 | 9502.66s |
| `last_block_plus_classifier_seed0` | deep plus classifier, last block | 0.4231 | 73.37 | -1.69 | 9547.58s |
| `deep_last_stage_seed0` | deep only, last stage | 0.7513 | 72.74 | -2.23 | 13600.42s |
| `deep_last_block_seed0` | deep only, last block | 0.4226 | 73.25 | -1.66 | 13579.92s |

Key result files:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_b_scope_v2/v2_projected_all_seed0/final_metrics.json`
- `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_b_scope_v2/classifier_only_seed0/final_metrics.json`
- `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_b_scope_v2/last_block_plus_classifier_seed0/final_metrics.json`
- `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_b_scope_v2/deep_last_stage_seed0/final_metrics.json`
- `/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_b_scope_v2/deep_last_block_seed0/final_metrics.json`

## 7. Unified comparison

### 7.1 Full-protocol results currently available

| Method | Scope | Acc | BWT | Time |
|---|---|---:|---:|---:|
| baseline | full Adam-NSCL, 3-seed mean | 73.20 | -1.31 | 7853.01s |
| V1 original | deep_plus_classifier, last_stage, 2/3 repeats so far | 72.13 partial | -1.91 partial | running |
| V2 projected_all | all params, projected | 74.20 | -2.80 | 19560.38s |
| classifier only | selective | 73.29 | -2.04 | 9502.66s |
| last_block + classifier | selective | 73.37 | -1.69 | 9547.58s |
| deep only, last stage | selective | 72.74 | -2.23 | 13600.42s |
| deep only, last block | selective | 73.25 | -1.66 | 13579.92s |

### 7.2 Cross-round context

Earlier rounds already established two stable facts:

1. Direct all-model C-Flat is not the right main line.
2. Lower-strength selective flatness is more promising than stronger or broader flatness.

Important earlier references:

- 3-task baseline: `66.73 / -0.95 / 359.13s`
- 3-task D2 direct-attach: `68.07 / -1.35 / 859.29s`
- 3-task V1 last-stage selective: `66.77 / -0.75 / 1527.06s`

What the full protocol adds is:

- the earlier short-horizon BWT win of `last_stage` selective does **not** carry cleanly into the long-horizon 10-task setting
- once the horizon is long enough, `last_block`-level selective becomes more credible than `last_stage`

## 8. What is already clear

### 8.1 V2 projected_all

Full result:

- acc: `74.20`
- bwt: `-2.80`
- time: `19560.38s`

Interpretation:

- acc is high
- forgetting is much worse than baseline
- runtime cost is by far the worst

Decision:

- **停止推进**

### 8.2 Scope narrowing

Completed scope variants now show:

- `classifier_only` is too narrow
- `deep_last_stage` is too broad and too slow
- `deep_last_block` is the best completed narrowed variant on BWT so far
- `last_block_plus_classifier` is the strongest cheaper narrowed variant

Current narrowed-scope ranking:

1. `deep_last_block_seed0`
2. `last_block_plus_classifier_seed0`
3. `classifier_only_seed0`
4. `deep_last_stage_seed0`
5. `v2_projected_all_seed0`

However, the top two are very close, and they win on different axes:

- BWT-first candidate: `deep_last_block_seed0`
- efficiency-aware candidate: `last_block_plus_classifier_seed0`

Current status judgment:

- `deep_last_block_seed0`: **弱继续**
- `last_block_plus_classifier_seed0`: **弱继续**

### 8.3 V1 original

Current evidence no longer supports treating `deep_plus_classifier + last_stage` as the preferred main configuration.

Why:

- two completed repeats are already both worse than baseline on BWT
- partial mean is also worse than baseline on acc
- the third repeat would need an unrealistically strong outcome to recover the overall mean
- the long-horizon full result trend disagrees with the earlier 3-task signal

Current status judgment:

- `V1 original`: **弱继续**

Likely final outcome after repeat 3:

- if repeat 3 lands in the same neighborhood as repeats 0 and 1, `V1 original` should be downgraded to:
  - **停止推进**

## 9. Best current main-line candidate

At this stage, the best next-step family is no longer `last_stage` selective.

The most credible main direction is now:

- **last_block-level Layer-Selective C-Flat**

Preferred ordering:

1. `last_block_plus_classifier`
2. `deep_last_block`
3. `V1 original`

Reasoning:

- `last_block_plus_classifier` is more balanced on acc and time
- `deep_last_block` is slightly better on BWT, but only marginally and at much higher time cost
- `V1 original` currently looks over-broad for the full 10-task horizon

If a single default configuration must be chosen now, before `v1_repeat3` finishes:

- choose **`last_block_plus_classifier`**

If the next round is strictly BWT-first and compute is less important:

- keep **`deep_last_block`** as the secondary candidate

## 10. Most plausible modification directions

Based on all data sources currently available, the most promising modification lines are:

### 10.1 Stop using `last_stage` as the main selective scope

This is the clearest design-level conclusion from the full runs.

Short-horizon experiments made `last_stage` look attractive, but the full 10-task horizon does not support it.

### 10.2 Move the main line to `last_block`

Best immediate options:

- `deep_plus_classifier + last_block`
- `deep_only + last_block`

This preserves the selective idea while reducing unnecessary disturbance to a wider stage.

### 10.3 Weaken the flatness term further inside selective runs

The older direct-attach diagnostics already showed:

- large `rho` is harmful
- `g1` is not consistently helpful

So the next rational ablations are:

- keep `rho` in `0.01` to `0.02`
- reduce `lambda` toward `0.00` to `0.01`

### 10.4 Prioritize selective `g0-only`

This is the most targeted next experiment direction.

Recommended candidates:

- `last_block_plus_classifier` with `lambda=0`
- `deep_last_block` with `lambda=0`

Why this is attractive:

- selective scope already proved more compatible than all-model flatness
- earlier diagnostics showed that `g1` carries negative side effects
- combining narrower scope with `g0-only` is the most natural next conservative move

### 10.5 Do not continue `projected_all`

This line already has enough negative evidence:

- poor BWT
- highest runtime
- no sign of a worthwhile CL-specific advantage

If projection is kept at all, it should only be considered as a lightweight refinement on top of narrowed selective scope, not as the main idea.

## 11. Stage conclusion

Current completion state:

- **Partially completed**

Completed:

- Group A `baseline_repeat3`
- Group B `v2_projected_all_seed0`
- Group B `classifier_only_seed0`
- Group B `last_block_plus_classifier_seed0`
- Group B `deep_last_stage_seed0`
- Group B `deep_last_block_seed0`

Still running:

- Group A `v1_repeat3`

Current stage-level labels:

- `V2 projected_all`: **停止推进**
- `V1 original`: **弱继续**
- `deep_last_block`: **弱继续**
- `last_block_plus_classifier`: **弱继续**

Current best practical recommendation:

- shift the main line from `last_stage` selective to **`last_block_plus_classifier`**

Current best BWT-first recommendation:

- keep **`deep_last_block`** as the stronger forgetting-oriented backup line

Main unresolved item:

- the third repeat of `v1_repeat3`

But even before that final repeat finishes, the current evidence already suggests that the center of gravity should move away from `V1 original` and toward **last-block selective variants**.
