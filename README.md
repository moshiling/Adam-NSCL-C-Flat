# Adam-NSCL-C-Flat

This repository is a full workspace snapshot for integrating **C-Flat / PLS-CFlat** into **Adam-NSCL** and validating the resulting continual-learning behavior on CIFAR100-10.

It contains:

- the modified Adam-NSCL codebase
- a local reference copy of C-Flat
- experiment outputs and logs
- round-by-round summaries

The main conclusion so far is simple:

- **full-model direct C-Flat is not a good main line for Adam-NSCL**
- **`projected_all` is not worth continuing**
- **the most promising direction is a narrow `last_block` selective scope**
- **the current best completed configuration is `last_block_plus_classifier + lambda=0`**

## Current best result

Under the full CIFAR100-10 protocol currently used in this project:

| Method | Acc | BWT | Time |
|---|---:|---:|---:|
| Adam-NSCL baseline (3-seed mean) | 73.20 | -1.31 | 7853.01s |
| `last_block_plus_classifier`, old `lambda=0.02` | 73.37 | -1.69 | 9547.58s |
| `deep_last_block`, old `lambda=0.02` | 73.25 | -1.66 | 13579.92s |
| **`last_block_plus_classifier`, `lambda=0`** | **73.53** | **-1.62** | **7587.29s** |
| `deep_last_block`, `lambda=0` | 73.51 | -1.70 | 12719.72s |

Interpretation:

- `last_block_plus_classifier + lambda=0` is the **best completed selective variant so far**
- it improves over the old `lambda=0.02` version on:
  - accuracy
  - BWT
  - runtime
- however, it still does **not** beat the Adam-NSCL baseline on BWT

So the current research direction is:

- keep the scope narrow
- keep flatness weak
- prefer `g0-only` or very small `lambda`

## Main findings across rounds

### 1. Direct C-Flat integration

After repairing the implementation to align with the official `c_flat.py` logic:

- engineering correctness improved a lot
- but direct full-model C-Flat still did not become a good CL method for Adam-NSCL

Relevant summaries:

- [Adam-NSCL-main/CFLAT_INTEGRATION_SUMMARY.md](Adam-NSCL-main/CFLAT_INTEGRATION_SUMMARY.md)
- [Adam-NSCL-main/CFLAT_OFFICIAL_ALIGNMENT_SUMMARY.md](Adam-NSCL-main/CFLAT_OFFICIAL_ALIGNMENT_SUMMARY.md)

### 2. Diagnostic sweep

The 4-diagnostic experiments showed:

- too much flatness is harmful
- `rho=0.05` is too aggressive
- `g1` has side effects
- a smaller `rho/lambda` pair is much safer than the high-strength setting

Relevant summary:

- [Adam-NSCL-main/CFLAT_4_DIAG_EXPERIMENTS_SUMMARY.md](Adam-NSCL-main/CFLAT_4_DIAG_EXPERIMENTS_SUMMARY.md)

### 3. PLS-CFlat

The PLS-CFlat round showed:

- `projected_all` is not a good main line
- `last_stage` is too wide under the full 10-task protocol
- selective application is more meaningful than global projection

Relevant summaries:

- [Adam-NSCL-main/PLS_CFLAT_D2_SUMMARY.md](Adam-NSCL-main/PLS_CFLAT_D2_SUMMARY.md)
- [Adam-NSCL-main/V1_FULL_SCOPE_V2_SUMMARY.md](Adam-NSCL-main/V1_FULL_SCOPE_V2_SUMMARY.md)

### 4. Current selective g0 / small-lambda round

This is the current converged main-line round:

- scope restricted to `last_block`
- only two selective families retained:
  - `last_block_plus_classifier`
  - `deep_last_block`
- only weak flatness settings retained:
  - `lambda=0`
  - `lambda=0.005`
  - `lambda=0.01`

Current round summary:

- [Adam-NSCL-main/SELECTIVE_G0_SMALLLAMBDA_SUMMARY.md](Adam-NSCL-main/SELECTIVE_G0_SMALLLAMBDA_SUMMARY.md)

## Repository structure

### Main modified code

- [Adam-NSCL-main](Adam-NSCL-main)
- entry point: [Adam-NSCL-main/main.py](Adam-NSCL-main/main.py)
- C-Flat helper: [Adam-NSCL-main/optim/cflat_utils.py](Adam-NSCL-main/optim/cflat_utils.py)
- training loop integration: [Adam-NSCL-main/svd_agent/agent.py](Adam-NSCL-main/svd_agent/agent.py)
- selective scope logic: [Adam-NSCL-main/svd_agent/svd_agent.py](Adam-NSCL-main/svd_agent/svd_agent.py)

### C-Flat reference snapshot

- [C-Flat-ref](C-Flat-ref)
- official step reference used during alignment:
  [C-Flat-ref/optims/c_flat.py](C-Flat-ref/optims/c_flat.py)

### Experiment outputs

- [outputs/cflat_integration](outputs/cflat_integration)
- [outputs/cflat_official_alignment](outputs/cflat_official_alignment)
- [outputs/cflat_diag_4exp](outputs/cflat_diag_4exp)
- [outputs/pls_cflat_d2_experiments](outputs/pls_cflat_d2_experiments)
- [outputs/full_v1_scope_v2](outputs/full_v1_scope_v2)
- [outputs/selective_g0_smalllambda_round](outputs/selective_g0_smalllambda_round)

## Most useful summary files

- [Adam-NSCL-main/ALL_EXPERIMENTS_SUMMARY.md](Adam-NSCL-main/ALL_EXPERIMENTS_SUMMARY.md)
- [Adam-NSCL-main/CFLAT_INTEGRATION_SUMMARY.md](Adam-NSCL-main/CFLAT_INTEGRATION_SUMMARY.md)
- [Adam-NSCL-main/CFLAT_OFFICIAL_ALIGNMENT_SUMMARY.md](Adam-NSCL-main/CFLAT_OFFICIAL_ALIGNMENT_SUMMARY.md)
- [Adam-NSCL-main/CFLAT_4_DIAG_EXPERIMENTS_SUMMARY.md](Adam-NSCL-main/CFLAT_4_DIAG_EXPERIMENTS_SUMMARY.md)
- [Adam-NSCL-main/PLS_CFLAT_D2_SUMMARY.md](Adam-NSCL-main/PLS_CFLAT_D2_SUMMARY.md)
- [Adam-NSCL-main/V1_FULL_SCOPE_V2_SUMMARY.md](Adam-NSCL-main/V1_FULL_SCOPE_V2_SUMMARY.md)
- [Adam-NSCL-main/SELECTIVE_G0_SMALLLAMBDA_SUMMARY.md](Adam-NSCL-main/SELECTIVE_G0_SMALLLAMBDA_SUMMARY.md)

## Practical takeaway

If someone only wants the current bottom line from this repo:

- do **not** continue `projected_all`
- do **not** continue wide `last_stage` selective scope
- the best current line is:
  - `layer_selective`
  - `deep_plus_classifier`
  - `deep_layer_rule=last_block`
  - `rho=0.02`
  - `lambda=0`
- if further refinement is needed, test:
  - even smaller `rho`
  - `lambda=0.005`
  - but do not go back to strong full-model flatness
