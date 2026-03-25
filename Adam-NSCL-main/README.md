# Adam-NSCL-C-Flat

This repository starts from the original Adam-NSCL codebase and adds a series of C-Flat and PLS-CFlat integrations for continual learning experiments.

The core design choice is:

- keep Adam-NSCL's original `Adam + SVD/null-space + regularization` update path
- use C-Flat only in the gradient-generation stage
- then hand the final gradient back to Adam-NSCL for the constrained optimizer step

The later experiments in this repo further show that:

- full-model direct C-Flat is not a good main line for Adam-NSCL
- wide `last_stage` selective scope is also not ideal under the full 10-task setting
- the most promising direction is a narrower `last_block` selective scope with very small `lambda`, or even `g0-only`

## Base paper

Original Adam-NSCL paper:

- [Training Networks in Null Space of Feature Covariance for Continual Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Training_Networks_in_Null_Space_of_Feature_Covariance_for_Continual_CVPR_2021_paper.pdf)

Original code authors:

- Shipeng Wang, Xiaorong Li, Jian Sun, Zongben Xu

## Main code changes in this repo

- official-aligned C-Flat gradient helper
- selective C-Flat over configurable parameter scopes
- projected / layer-selective PLS-CFlat variants
- experiment summaries and result aggregation for:
  - direct C-Flat repair
  - 4-diagnostic runs
  - PLS-CFlat ablations
  - full CIFAR100-10 comparisons
  - selective g0 / small-lambda follow-up round

## Key summaries

- `CFLAT_INTEGRATION_SUMMARY.md`
- `CFLAT_OFFICIAL_ALIGNMENT_SUMMARY.md`
- `CFLAT_4_DIAG_EXPERIMENTS_SUMMARY.md`
- `PLS_CFLAT_D2_SUMMARY.md`
- `V1_FULL_SCOPE_V2_SUMMARY.md`
- `SELECTIVE_G0_SMALLLAMBDA_SUMMARY.md`
- `ALL_EXPERIMENTS_SUMMARY.md`

## Original Adam-NSCL usage

```bash
sh scripts_svd/adamnscl.sh
```

## Requirements

- Python 3.7+
- PyTorch
- tensorboardX

## Citation

If you use the original Adam-NSCL method, please cite:

```bibtex
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Shipeng and Li, Xiaorong and Sun, Jian and Xu, Zongben},
    title     = {Training Networks in Null Space of Feature Covariance for Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {184-193}
}
```
