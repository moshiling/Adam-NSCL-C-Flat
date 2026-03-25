# Selective g0 / Small-Lambda Round Summary

## 1. Round goal

This round is a **converged main-line experiment round**, not a new branching round.

It follows directly from:

- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/V1_FULL_SCOPE_V2_SUMMARY.md`

The current historical conclusions carried into this round are:

1. `projected_all` is no longer worth continuing.
2. `last_stage` is no longer the preferred selective scope under the full 10-task protocol.
3. The remaining credible selective lines are:
   - practical line: `last_block_plus_classifier`
   - BWT-first backup line: `deep_last_block`
4. The most promising next refinement is:
   - `lambda = 0`
   - or very small `lambda` in `0.005` to `0.01`

So this round only tests:

- `last_block_plus_classifier`
- `deep_last_block`

with:

- `lambda = 0`
- `lambda = 0.005`
- `lambda = 0.01`

and keeps:

- `rho = 0.02`

## 2. Why these choices

### 2.1 Why not `projected_all`

Because the full protocol already showed:

- high accuracy can coexist with very poor BWT
- runtime is the worst
- the method is not aligned with the CL objective we care about

So `projected_all` is not part of this formal round.

### 2.2 Why not `last_stage`

Because the full 10-task results already showed:

- `last_stage` is too wide
- forgetting is not stably improved
- runtime cost is high

So this round narrows the search to `last_block`.

### 2.3 Why focus on `lambda = 0 / 0.005 / 0.01`

Because earlier evidence already indicated:

- broad flatness is harmful
- `g1` is not reliably helpful
- smaller `lambda` is safer

So this round tests whether **selective g0-only** or **very small g1 contribution** is enough.

## 3. Full protocol used

This round uses the same full CIFAR100-10 protocol as the earlier full runs:

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

Protocol source:

- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/our_results/rehearsal_org/cifar100-10/svd_epoch_80_bn32_lr1e-4_headlr1e-3_bnlr5e-4_svdlr5e-5_wdecay5e-5_regcoef_100_eigvec_gt_ada10_combine0.log`

## 4. Minimal implementation refinement

Modified file:

- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/optim/cflat_utils.py`

This round added a **minimal lambda-zero fast path**:

- if `lambda == 0`, the helper now directly writes the equivalent final gradient from the stored C-Flat term
- it skips the later two gradient-generation steps that no longer affect the result

Important note:

- this is an **engineering optimization**
- it does **not** change the intended `lambda=0` result
- it only removes unnecessary extra work for the selective g0-only case

Quick sanity check:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/static_check_lambda0`

Observed debug signature:

- `g_2: 0.000000`
- final gradient is still non-zero and finite
- no startup failure in the selective lambda-zero branch

## 5. Stage 1 design

Stage 1 runs 6 seed-0 full experiments:

- A1 `last_block_plus_classifier_lambda0`
- A2 `last_block_plus_classifier_lambda0005`
- A3 `last_block_plus_classifier_lambda001`
- B1 `deep_last_block_lambda0`
- B2 `deep_last_block_lambda0005`
- B3 `deep_last_block_lambda001`

Output root:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round`

Stage 1 output root:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/stage1_seed0`

Stage 1 queue status:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/stage1_seed0/queue_status.log`

Stage 1 aggregate file:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/stage1_seed0/aggregate_results.json`

## 6. Stage 2 design

Stage 2 is automated:

- once all 6 stage-1 seed-0 full results are available
- the stage-1 aggregator ranks them by:
  1. BWT
  2. acc
  3. time
- then the best config is selected
- then a 3-seed full confirmation is launched automatically

Stage 2 output root:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/stage2_best_3seed`

Stage 2 queue status:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/stage2_best_3seed/queue_status.log`

## 7. Reference results carried into this round

### Full-protocol references

| Method | Acc | BWT | Time |
|---|---:|---:|---:|
| baseline full 3-seed | 73.20 | -1.31 | 7853.01s |
| `last_block_plus_classifier`, old `lambda=0.02` | 73.37 | -1.69 | 9547.58s |
| `deep_last_block`, old `lambda=0.02` | 73.25 | -1.66 | 13579.92s |

### Historical reference only

This run is retained as context, but it is **not** directly full-protocol comparable:

| Method | Protocol | Acc | BWT | Time |
|---|---|---:|---:|---:|
| D2 direct-attach | earlier 3-task protocol | 68.07 | -1.35 | 859.29s |

## 8. Current launch status

Current state at this checkpoint:

- **Stage 1 has been launched**
- Group A and Group B are both active
- Stage 2 waiting logic has been prepared and can be started/continued after stage-1 completion

Current group allocation:

- Group A on physical `GPU 2`
- Group B on physical `GPU 5`

### Confirmed active run A

Run:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/stage1_seed0/last_block_plus_classifier_lambda0`

Files already present:

- `command.sh`
- `run.log`

Confirmed behavior from `run.log`:

- training entered normal epoch loop
- selective scope line is correct:
  - `scope: deep_plus_classifier`
  - `deep_rule: last_block`
  - `ratio: 0.4231`
- `lambda=0` fast path is active:
  - `g_2: 0.000000`
- the run has already advanced beyond startup and into later training epochs

### Confirmed active run B

Run:

- `/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/stage1_seed0/deep_last_block_lambda0`

Files already present:

- `command.sh`
- `run.log`

Confirmed behavior from `run.log`:

- training entered normal epoch loop
- selective scope line is correct:
  - `scope: deep`
  - `deep_rule: last_block`
  - `ratio: 0.4226`
- `lambda=0` fast path is active:
  - `g_2: 0.000000`
- the run has already advanced beyond startup and into later training epochs

## 9. Stage 1 selection rule

Selection priority for stage 1 is:

1. BWT
2. acc
3. time

A new config should be preferred if:

- it improves BWT over the old `lambda=0.02` version of the same selective scope
- while keeping acc acceptable
- and without unreasonable extra runtime

The most important practical question is:

- is `lambda=0` already good enough
- or does `lambda=0.005` / `0.01` give a better BWT-acc tradeoff

## 10. Current working hypothesis

Before stage 1 finishes, the most plausible outcomes are:

1. `last_block` remains the right scope family
2. `lambda=0` or `lambda=0.005` is more likely than `lambda=0.01` to become the new default
3. there is a real possibility that default selective C-Flat should become effectively **g0-only**

But this is still a hypothesis until the 6 seed-0 full runs finish.

## 11. Current status labels

At this checkpoint:

- best current scope family: **last_block**
- best practical line under test: **last_block_plus_classifier**
- best BWT-first backup under test: **deep_last_block**
- projected line: **停止推进**
- last-stage line: **停止推进**

The unresolved question of this round is now narrower:

- does the default selective line keep any useful `g1`, or should the default move to `lambda=0` or near-zero `lambda`

