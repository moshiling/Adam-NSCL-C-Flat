# /home/moshiling 全局实验总表

## 1. 整理范围与口径

- 更新时间：`2026-03-25 Asia/Shanghai`
- 本次范围：`/home/moshiling` 下所有可识别实验工件，不再只限于 `Adam-NSCL-C-Flat`
- 已扫描并确认有实验痕迹的主项目：
  - `Adam-NSCL-C-Flat`
  - `Adam-resnet32-depth`
  - `Adam-resnet32-part1`
  - `Adam-resnet32-part2`
  - `Adam-Depth + Spectrum`
  - `Adam-NSCL-resnet18+AdNS `
  - `Adam-NSCL-main-test`
- 结果列统一写法：`Acc / BWT / Time`
- 若项目未导出 wall-clock，则写 `Time NA`
- 若只有阶段日志而没有完整收尾，则写成 `partial`
- 若是工程 smoke / determinism / resume 校验，结果列优先写可观测的 `ValAcc`，并在说明列明确用途

## 2. Adam-NSCL + C-Flat 线（`Adam-NSCL-C-Flat`）

| 使用的网络 | 方法简称 | 细分的试验参数 | 实验结果 | 试验方法说明 | 补充说明 |
|---|---|---|---|---|---|
| ResNet-18 | Base-Static | `1 task`，`schedule=1`，`bs=128`，`max_train_batches=2`，`max_eval_batches=1`，`seed=0` | `Acc 3.91 / BWT 0.00 / 45.22s` | 原始 Adam-NSCL 静态对照 | 工程静态检查 |
| ResNet-18 | CFlat-Zero | `1 task`，`schedule=1`，`bs=128`，`seed=0`，`rho=0`，`lambda=0` | `Acc 3.91 / BWT 0.00 / 54.25s` | C-Flat 退化零扰动等价检查 | 等价性通过 |
| ResNet-18 | CFlat-Static-v0 | `1 task`，`schedule=1`，`bs=128`，`seed=0`，`rho=0.2`，`lambda=0.2` | `Acc 0.00 / BWT 0.00 / 52.28s` | 初版 direct-attach C-Flat 静态跑通 | 仅工程验证 |
| ResNet-18 | Base-Smoke-v0 | `2 tasks`，`schedule=1`，`bs=128`，`seed=0` | `Acc 39.75 / BWT 7.00 / 110.45s` | 初版短程 baseline | 冒烟实验 |
| ResNet-18 | CFlat-Smoke-v0 | `2 tasks`，`schedule=1`，`bs=128`，`seed=0`，`rho=0.2`，`lambda=0.2` | `Acc 28.90 / BWT -3.20 / 111.80s` | 初版 direct-attach C-Flat | 相比 baseline 明显劣化 |
| ResNet-18 | Base-Formal-v0 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`seed=0` | `Acc 66.73 / BWT -0.95 / 434.77s` | 初版阶段正式 baseline | 后续多轮参考点 |
| ResNet-18 | CFlat-Formal-v0 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`seed=0`，`rho=0.2`，`lambda=0.2` | `Acc 54.47 / BWT -1.95 / 952.47s` | 初版正式 direct-attach C-Flat | Acc 与 BWT 都偏负面 |
| ResNet-18 | Align-Base-Static | `1 task`，`schedule=1`，`bs=128`，`seed=0` | `Acc 3.91 / BWT 0.00 / 37.71s` | 官方语义对齐版 baseline 静态检查 | 修复后 baseline 未受破坏 |
| ResNet-18 | Align-CFlat-Static | `1 task`，`schedule=1`，`bs=128`，`seed=0`，`rho=0.05`，`lambda=0.05` | `Acc 0.78 / BWT 0.00 / 36.39s` | 官方语义对齐版 C-Flat 静态检查 | 更接近官方实现 |
| ResNet-18 | Align-Base-Smoke | `2 tasks`，`schedule=1`，`bs=128`，`seed=0` | `Acc 39.75 / BWT 7.00 / 91.66s` | 官方对齐版短程 baseline | 冒烟实验 |
| ResNet-18 | Align-CFlat-Smoke | `2 tasks`，`schedule=1`，`bs=128`，`seed=0`，`rho=0.05`，`lambda=0.05` | `Acc 37.00 / BWT -3.40 / 89.41s` | 官方对齐版短程 C-Flat | 比初版合理，但 BWT 仍差 |
| ResNet-18 | Align-Base-Formal | `3 tasks`，`schedule=5/10/15`，`bs=32`，`seed=0` | `Acc 66.73 / BWT -0.95 / 359.13s` | 官方对齐版正式 baseline | D1-D4 与 PLS 的主 baseline |
| ResNet-18 | Align-CFlat-Formal | `3 tasks`，`schedule=5/10/15`，`bs=32`，`seed=0`，`rho=0.05`，`lambda=0.05` | `Acc 66.90 / BWT -2.10 / 811.32s` | 官方对齐后的 direct C-Flat | Acc 回升，但 BWT 仍差 |
| ResNet-18 | D1 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`rho=0.05`，`lambda=0` | `Acc 67.33 / BWT -1.75 / 1029.58s` | `g0-only` direct-attach | 验证 `g1` 有不匹配问题 |
| ResNet-18 | D2 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`rho=0.02`，`lambda=0.02` | `Acc 68.07 / BWT -1.35 / 859.29s` | 保守 direct-attach 基线 | D1-D4 中最稳健 |
| ResNet-18 | D3 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`rho=0.05`，`lambda=0.02` | `Acc 66.93 / BWT -1.70 / 1075.56s` | 只减小 `lambda` | 说明 `rho` 更关键 |
| ResNet-18 | D4 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`rho=0.05`，`lambda=0.05` | `Acc 66.90 / BWT -2.10 / 1426.05s` | repaired default 复核 | 与对齐 formal C-Flat 一致 |
| ResNet-18 | V1 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`mode=layer_selective`，`scope=deep_plus_classifier`，`deep_rule=last_stage`，`ratio=0.7517`，`rho=0.02`，`lambda=0.02` | `Acc 66.77 / BWT -0.75 / 1527.06s` | PLS-CFlat Layer-Selective | 本轮最佳 BWT |
| ResNet-18 | V2 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`mode=projected_all`，`scope=all`，`ratio=1.0000`，`project_before/after=True` | `Acc 68.17 / BWT -1.75 / 2135.59s` | 全参数投影 PLS-CFlat | Acc 最好，但 BWT 差 |
| ResNet-18 | V3 | `3 tasks`，`schedule=5/10/15`，`bs=32`，`mode=projected_layer_selective`，`scope=deep_plus_classifier`，`ratio=0.7517` | `Acc 66.77 / BWT -1.15 / 962.25s` | selective + projection | 优于 D2，但不如 V1 |
| ResNet-18 | Full-Base-3seed | `10 tasks`，`schedule=30/60/80`，`bs=32`，`repeat=3`，`seeds=0/1/2` | `Acc 73.20±0.20 / BWT -1.31±0.35 / 7853.01±419.28s` | full protocol 原始 Adam-NSCL 基线 | full line 主参考 |
| ResNet-18 | Full-V1-ongoing | `10 tasks`，`repeat=3`，`mode=layer_selective`，`scope=deep_plus_classifier`，`deep_rule=last_stage`，`ratio=0.7517`，`rho=0.02`，`lambda=0.02` | `partial: Acc 72.13(2/3) / BWT -1.91(2/3) / Time running` | 把 3-task 最强 V1 扩到 full 10-task | 当前信号偏负面 |
| ResNet-18 | Full-V2-All | `10 tasks`，`mode=projected_all`，`scope=all`，`ratio=1.0000`，`project_before/after=True` | `Acc 74.20 / BWT -2.80 / 19560.38s` | full 全参数投影版 | Acc 最高，但 forgetting 最差 |
| ResNet-18 | Full-CLS | `10 tasks`，`mode=layer_selective`，`scope=classifier`，`ratio=0.0005` | `Acc 73.29 / BWT -2.04 / 9502.66s` | 只动分类头 | Acc 接近 baseline，BWT 不佳 |
| ResNet-18 | Full-LB+CLS | `10 tasks`，`mode=layer_selective`，`scope=deep_plus_classifier`，`deep_rule=last_block`，`ratio=0.4231` | `Acc 73.37 / BWT -1.69 / 9547.58s` | 最后一个 block + 分类头 | full 10-task 中较可信 |
| ResNet-18 | Full-Deep-LS | `10 tasks`，`mode=layer_selective`，`scope=deep`，`deep_rule=last_stage`，`ratio=0.7513` | `Acc 72.74 / BWT -2.23 / 13600.42s` | 最后整 stage selective | 长时程表现较弱 |
| ResNet-18 | Full-Deep-LB | `10 tasks`，`mode=layer_selective`，`scope=deep`，`deep_rule=last_block`，`ratio=0.4226` | `Acc 73.25 / BWT -1.66 / 13579.92s` | 最后一个 block selective | deep-only 中较均衡 |
| ResNet-18 | G0-SmallLambda-Stage1 | `10 tasks`，只测 `last_block_plus_classifier` 与 `deep_last_block`，`rho=0.02`，`lambda∈{0,0.005,0.01}` | `running / no final metrics yet / Time NA` | selective g0 / small-lambda 收敛轮 | 已确认 `lambda=0` fast-path 正常，Stage1 已启动 |

## 3. ResNet32 深度阈值线（`Adam-resnet32-depth`）

| 使用的网络 | 方法简称 | 细分的试验参数 | 实验结果 | 试验方法说明 | 补充说明 |
|---|---|---|---|---|---|
| ResNet-32 | Depth-Base-A10D45 | `10 tasks`，`schedule=30/60/80`，`bs=32`，`shallow_a=10`，`deep_a=45`，`deep_mapping=last_block(final residual block only)` | `Acc 67.65 / BWT -3.00 / Time NA` | 深浅分离阈值 baseline | 任务 10 仅 `37.7`，且经历 1 次 resume |
| ResNet-32 | Depth-Last2-A45 | `10 tasks`，`shallow_a=10`，`deep_a=45`，`deep_mapping=last_two_blocks` | `Acc 72.41 / BWT -2.07 / Time NA` | 自动深度优化候选 | 优于深度 baseline |
| ResNet-32 | Depth-Last1-A50 | `10 tasks`，`shallow_a=10`，`deep_a=50`，`deep_mapping=last_block(stage4.2)` | `Acc 72.44 / BWT -2.02 / Time NA` | 自动深度优化最佳单 seed | `auto_depth_opt` 中最佳 |
| ResNet-32 | Depth-Last1-A45S12 | `10 tasks`，`shallow_a=12`，`deep_a=45`，`deep_mapping=last_block` | `partial / BWT NA / Time NA` | 补充验证 run | 已 early-stop，因主胜者已明确 |
| ResNet-32 | Depth-A50-3Seed | `10 tasks`，`shallow_a=10`，`deep_a=50`，`deep_mapping=last_block`，`seed=0/1/2` | `Acc 72.30±0.23 / BWT -1.56±0.51 / Time NA` | 对最佳 `last_block a50` 做 3-seed 验证 | seed1/2 结果分别 `71.97/-0.84`、`72.49/-1.82` |
| ResNet-32 | Depth-Smoke-Last2 | `1 task`，`schedule=1`，`max_tasks=2`，`max_train_batches=2`，`stop_after_task_count=1` | `Acc 0.00 / BWT 0.00 / Time NA` | 深度优化 smoke_check | 只用于跑通与 checkpoint 检查 |

## 4. ResNet32 光谱预算线（`Adam-resnet32-part1`）

| 使用的网络 | 方法简称 | 细分的试验参数 | 实验结果 | 试验方法说明 | 补充说明 |
|---|---|---|---|---|---|
| ResNet-32 | SB-Baseline-A15 | `10 tasks`，历史 paper-aligned baseline | `Acc 73.18 / BWT -2.16 / Time NA` | 章节对照基线 | 来自 `result_summary.md` 的参考基线 |
| ResNet-32 | SB-Base | `ns_mode=spectral_budget`，`rho=0.001~0.03`，`gamma=2.5`，`svd_thres=15` | `Acc 56.91 / BWT -16.87 / Time NA` | 最早的 spectral budget 主线 | 明显不稳定 |
| ResNet-32 | SB-Alt1 | `rho=0.005~0.06`，`gamma=2.0` | `Acc 59.97 / BWT -10.91 / Time NA` | 第一轮强度调参 | 比基线好，但仍差 |
| ResNet-32 | SB-Alt2-FixProj | `deepcap=0.75`，固定投影 | `partial: task2 avg_acc 65.00 / BWT NA / Time NA` | Alt2 固定投影分支 | 仅跑到 task2 |
| ResNet-32 | SB-Alt2-NormProj | `deepcap=0.75`，归一化投影 | `Acc 61.40 / BWT -5.40 / Time NA` | Alt2 归一化投影 | 相比 Alt1 明显改善 |
| ResNet-32 | SB-Alt2-Rho002-04 | `rho=0.002~0.04`，`gamma=2.0` | `Acc 61.71 / BWT -5.57 / Time NA` | Alt2 强度修正 | 与 Alt2-NormProj 接近 |
| ResNet-32 | SB-Alt3-DeepCap | `deepcap=0.75 only` | `Acc 61.27 / BWT -3.07 / Time NA` | Alt3 去掉其他改动，仅保留 deepcap | BWT 明显改善 |
| ResNet-32 | SB-Alt3-FixProj-W2 | `warmup_tasks=2`，固定投影 | `partial: task2 avg_acc 65.45 / BWT NA / Time NA` | Alt3 固定投影 + warmup2 | 仅跑到 task2 |
| ResNet-32 | SB-Alt3-NormProj-W2 | `warmup_tasks=2`，归一化投影 | `Acc 61.67 / BWT -3.89 / Time NA` | Alt3 norm projection + warmup2 | 比 Alt2 更稳 |
| ResNet-32 | SB-Alt3-Rho001-03 | `rho=0.001~0.03`，`gamma=2.5` | `Acc 61.65 / BWT -3.12 / Time NA` | Alt3 主候选 | `old_alt3` 参考点 |
| ResNet-32 | SB-Alt3-Warmup2 | `warmup_tasks=2 only` | `partial: task3 avg_acc 65.73 / BWT NA / Time NA` | 只保留 warmup2 | 仅跑到 task3 |
| ResNet-32 | SB-SoftWeight | `ns_mode=soft_weighting`，`soft_tau=2.0`，`soft_floor=0.1`，`soft_depth_factor=0.15` | `partial: task2 Acc 65.65 / BWT -18.30 / Time NA` | SFCL-like soft weighting 消融 | 早停，稳定性差 |
| ResNet-32 | SB-LightSSL | `ssl_mode=augment_consistency`，`ssl_coef=0.05` | `partial: task5 Acc 60.88 / BWT -3.83 / Time NA` | 轻量 SSL 消融 | 早停到 task5，BWT 比 old alt3 更差 |

## 5. ResNet32 触发式 Adapter 线（`Adam-resnet32-part2`）

| 使用的网络 | 方法简称 | 细分的试验参数 | 实验结果 | 试验方法说明 | 补充说明 |
|---|---|---|---|---|---|
| ResNet-32 | TDA-Base | `expansion_mode=plasticity_triggered_adapter`，`scope=last_stage`，`last_n=5`，`warmup=30`，`tau=0.15`，`ratio=0.25`，`alpha=1.0` | `Acc 54.28 / BWT -27.50 / Time NA` | 初版 triggered deep adapter | 性能很差 |
| ResNet-32 | TDA-Opt-0092 | `tau=0.0092`，`ratio=0.125`，`alpha=1.0` | `Acc 59.07 / BWT -21.29 / Time NA` | 第一轮参数收缩 | 比初版好但仍差 |
| ResNet-32 | TDA-Opt-00945 | `tau=0.00945`，`ratio=0.125`，`alpha=1.0` | `Acc 56.35 / BWT -24.66 / Time NA` | Tau 微调对比 | 不如 `0.0092` |
| ResNet-32 | TDA-NoTrigger | `disable_adapter_trigger=True`，其余沿用 `tau=0.0092`，`ratio=0.125` | `Acc 73.10 / BWT -2.34 / Time NA` | 关闭触发器诊断 | 说明主要问题在触发机制本身 |
| ResNet-32 | TDA-SingleAdapter | `force_single_adapter_task=2`，`block=stage4.2`，`max_total_adapters=1`，`alpha=0.3` | `Acc 73.96 / BWT -2.31 / Time NA` | 单 adapter 强约束诊断 | 是当前强正信号之一 |
| ResNet-32 | TDA-Fix-Alpha03 | `tau=0.0092`，`ratio=0.125`，`alpha=0.3` | `Acc 70.50 / BWT -7.93 / Time NA` | 降低 adapter alpha 的修复版 | 有改善但不够稳 |
| ResNet-32 | TDA-Fix-NormSqrt | `tau=0.0092`，`ratio=0.125`，`alpha=1.0`，`aggregation=norm_sqrt` | `Acc 69.10 / BWT -9.69 / Time NA` | 引入 `norm_sqrt` 聚合 | 仍不理想 |
| ResNet-32 | TDA-Budget1 | `tau=0.0092`，`ratio=0.125`，`alpha=0.3`，`aggregation=norm_sqrt`，`max_total_adapters=1` | `Acc 73.96 / BWT -2.31 / Time NA` | stage-budget round，预算=1 | 与 single-adapter 结论一致 |
| ResNet-32 | TDA-Budget2 | `tau=0.0092`，`ratio=0.125`，`alpha=0.3`，`aggregation=norm_sqrt`，`max_total_adapters=2` | `Acc 73.94 / BWT -2.94 / Time NA` | stage-budget round，预算=2 | Acc 近似，但 BWT 略差 |
| ResNet-32 | TDA-Unlimited | `tau=0.0092`，`ratio=0.125`，`alpha=0.3`，`aggregation=norm_sqrt`，不限 adapter 数 | `partial: task7 reached / latest ValAcc 76.10 / Time NA` | 无限预算版本 | 日志仍在中途，无 end-of-run summary |
| ResNet-32 | TDA-Smoke | `smoke_check` 与 `smoke_check_fix` | `no final summary / Time NA` | 只做 smoke 跑通 | 未导出正式结果 |

## 6. ResNet18 Depth + Spectrum 线（`Adam-Depth + Spectrum`）

| 使用的网络 | 方法简称 | 细分的试验参数 | 实验结果 | 试验方法说明 | 补充说明 |
|---|---|---|---|---|---|
| ResNet-18 | DS-None | `layerwise_thres_strategy=none`，`schedule=30/60/80`，`bs=32` | `Acc 73.76 / BWT -0.88 / Time NA` | 原始阈值策略对照 | 这一轮最好 |
| ResNet-18 | DS-Spectrum | `strategy=spectrum`，`alpha=0.5`，`beta=0.2`，`thres=8~12` | `Acc 73.47 / BWT -1.40 / Time NA` | 仅光谱阈值 | 比 baseline 略差 |
| ResNet-18 | DS-DepthSpectrum | `strategy=depth_spectrum`，`depth_lambda=0.2`，`spectrum_lambda=0.2`，`alpha=0.5`，`beta=0.2`，`thres=8~12` | `Acc 71.72 / BWT -2.40 / Time NA` | 深度 + 光谱联合阈值 | 明显不如 baseline |
| ResNet-18 | DS-DepthSpectrum-005 | `strategy=depth_spectrum`，`depth_lambda=0.05`，`spectrum_lambda=0.05` | `Acc 71.98 / BWT -2.28 / Time NA` | 降低耦合强度的调参版 | 比 `0.2/0.2` 略好，但仍差于 `none` |

## 7. ResNet18 SFCL / AdNS / Shared-Subspace 线（`Adam-NSCL-resnet18+AdNS `）

| 使用的网络 | 方法简称 | 细分的试验参数 | 实验结果 | 试验方法说明 | 补充说明 |
|---|---|---|---|---|---|
| ResNet-18 | AdNS-Port-Smoke-SFCL | `2 tasks`，smoke，`projection_mode=sfcl` | `Acc 24.62 / BWT 3.24 / Time NA` | 端口接入 smoke | 早期快速验证 |
| ResNet-18 | AdNS-Port-Smoke-ADNS | `2 tasks`，smoke，`projection_mode=sfcl_adns` | `Acc 25.11 / BWT -0.02 / Time NA` | AdNS 端口 smoke | 比 SFCL smoke 略高 Acc |
| ResNet-18 | AdNS-Port-Smoke-ADNS-KD | `2 tasks`，smoke，`projection_mode=sfcl_adns`，`KD on` | `Acc 26.56 / BWT 2.16 / Time NA` | AdNS + KD smoke | smoke 中 Acc 最好 |
| ResNet-18 | AdNS-Port-Quick-SFCL | `10 tasks`，`schedule=1`，quick validation | `Acc 43.17 / BWT -1.01 / Time NA` | SFCL quick10 | 作为 AdNS quick 对照 |
| ResNet-18 | AdNS-Port-Quick-ADNS | `10 tasks`，`schedule=1`，quick validation，`shared_lowrank+task_strength+KD` | `Acc 48.37 / BWT -9.64 / Time NA` | AdNS quick10 | Acc 高于 SFCL quick，但 BWT 很差 |
| ResNet-18 | AdNS-Repro-NSCL | `10 tasks`，`projection_mode=nscl`，`overlap_core`，`seed=0` | `Acc 72.95 / BWT -1.80 / Time NA` | 修正后 NSCL 正式主结果 | final repro 基线 |
| ResNet-18 | AdNS-Repro-SFCL | `10 tasks`，`projection_mode=sfcl`，`overlap_core`，`seed=0` | `Acc 74.25 / BWT -4.48 / Time NA` | 修正后 SFCL 正式主结果 | final repro 主结果 |
| ResNet-18 | AdNS-Repro-OverlapCore | `10 tasks`，`projection_mode=sfcl_adns`，`shared_subspace=overlap_core` | `Acc 49.51 / BWT -40.90 / Time NA` | 修正后 AdNS overlap_core 正式结果 | `v2` 与 `v3_gpu1` 一致 |
| ResNet-18 | AdNS-Repro-UnionLowrank | `10 tasks`，`projection_mode=sfcl_adns`，`shared_subspace=union_lowrank` | `Acc 52.44 / BWT -37.53 / Time NA` | union_lowrank 正式结果 | 比 overlap_core 略好，但仍很差 |
| ResNet-18 | AdNS-Ablation-E1 | `shared_lowrank=True`，`task_strength=False`，`KD=False` | `Acc 74.36 / BWT -3.51 / Time NA` | `union_shared_only` 消融 | 结果已在 `summary.json` 完整落盘，比早期报告中间态更完整 |
| ResNet-18 | AdNS-Ablation-E2 | `shared_lowrank=True`，`task_strength=True`，`KD=False` | `Acc 38.88 / BWT -53.89 / Time NA` | `task_strength-only` 消融 | 当前最强负面信号 |
| ResNet-18 | AdNS-Ablation-E3 | `shared_lowrank=True`，`task_strength=False`，`KD=True` | `running / no final summary / Time NA` | `union_shared_kd` 消融 | `run_gpu1.log` 空，正式结果未落盘 |
| ResNet-18 | AdNS-MA-Fixed-SFCL | `2 tasks`，`subset_smoke_samples=256`，`projection_mode=sfcl` | `Acc 19.53 / BWT 0.00 / Time NA` | method-alignment fixed smoke | 与 fixed ADNS smoke 对齐 |
| ResNet-18 | AdNS-MA-Fixed-ADNS | `2 tasks`，`subset_smoke_samples=256`，`projection_mode=sfcl_adns` | `Acc 19.53 / BWT 0.00 / Time NA` | method-alignment fixed smoke | `method_alignment_smoke_sfcl_adns_fixed_v3_gpu1` |
| ResNet-18 | AdNS-Resume-Check | `2 tasks`，`resume_checkpoint used`，`projection_mode=sfcl_adns` | `Acc 19.53 / BWT 0.00 / Time NA` | 修正后 resume 路径校验 | 工程正确性验证 |
| ResNet-18 | AdNS-MA-Full-SFCL-v2 | `10 tasks`，method-alignment smoke v2 | `Acc 43.82 / BWT -0.12 / Time NA` | method-alignment 全量 SFCL 校验 | 作为修正规则下的 SFCL 全量 smoke |
| ResNet-18 | AdNS-MA-Full-SFCL-v3 | `10 tasks`，method-alignment smoke v3 | `Acc 39.31 / BWT 0.17 / Time NA` | 另一版 SFCL 全量 smoke | 比 v2 略差 |
| ResNet-18 | AdNS-MA-Full-ADNS-v3 | `10 tasks`，method-alignment smoke v3，`projection_mode=sfcl_adns` | `Acc 41.41 / BWT -7.93 / Time NA` | 全量 ADNS smoke | BWT 明显较差 |
| ResNet-18 | AdNS-MA-Limited2-SFCL | `2 tasks`，`limited_tasks=2`，`projection_mode=sfcl` | `Acc 38.40 / BWT -2.00 / Time NA` | 限制到 2 tasks 的对齐 smoke | 用于快速端到端对照 |
| ResNet-18 | AdNS-MA-Subset-SFCL | `2 tasks`，`subset_smoke_samples=256`，两次重复：`subset` 与 `subset2` | `Acc 19.53 / BWT 0.00 / Time NA` | 子集 smoke helper | 两次结果一致 |
| ResNet-18 | AdNS-MA-Subset-ADNS | `2 tasks`，`subset_smoke_samples=256`，两次重复：`subset` 与 `subset2` | `Acc 19.53 / BWT 0.00 / Time NA` | 子集 smoke helper | 两次结果一致 |

## 8. ResNet18 决定性 / Resume 校验线（`Adam-NSCL-main-test`）

| 使用的网络 | 方法简称 | 细分的试验参数 | 实验结果 | 试验方法说明 | 补充说明 |
|---|---|---|---|---|---|
| ResNet-18 | Test-Smoke-PilotA-Det | `schedule=1`，`fix_seed=1`，`fix_cudnn=1` | `ValAcc(split2)=38.20 / BWT NA / Time NA` | 决定性 smoke | 只做 launcher/训练链路检查 |
| ResNet-18 | Test-Smoke-PilotA-NonDet | `schedule=1`，`fix_seed=0`，`fix_cudnn=0` | `ValAcc(split2)=33.70 / BWT NA / Time NA` | 非决定性 smoke | 对照决定性开关 |
| ResNet-18 | Test-Smoke-PilotB-s0-c0 | `schedule=1`，`seed off`，`cudnn off` | `ValAcc(split2)=33.80 / BWT NA / Time NA` | PilotB smoke | 检查 cudnn 影响 |
| ResNet-18 | Test-Smoke-PilotB-s0-c1 | `schedule=1`，`seed off`，`cudnn on` | `ValAcc(split2)=35.40 / BWT NA / Time NA` | PilotB smoke | 比 `cudnn off` 略高 |
| ResNet-18 | Test-Smoke-PilotB-s1-c0 | `schedule=1`，`seed on`，`cudnn off` | `ValAcc(split2)=37.70 / BWT NA / Time NA` | PilotB smoke | 与 `seed off` 比较 |
| ResNet-18 | Test-Smoke-PilotB-s1-c1 | `schedule=1`，`seed on`，`cudnn on` | `ValAcc(split2)=38.20 / BWT NA / Time NA` | PilotB smoke | 当前 smoke 里最高 |
| ResNet-18 | Test-PilotA-Det | `5 runs`，`fix_seed=1`，`fix_cudnn=1`，`schedule=30/60/80` | `logs present / no unified final summary / Time NA` | 决定性多 seed 正式校验 | `seed_0~4` 日志存在，但未导出统一 metrics 文件 |
| ResNet-18 | Test-PilotA-NonDet | `3 runs`，`fix_seed=0`，`fix_cudnn=0`，`schedule=30/60/80` | `logs present / some runs incomplete / Time NA` | 非决定性多次正式校验 | 用于对照随机性 |
| ResNet-18 | Test-PilotA-Resume-Det | `resume det seeds=1/2/4`，`fix_seed=1`，`fix_cudnn=1` | `final split10 ValAcc ≈ 77.4 / 78.5 / 77.9 / Time NA` | resume 路径校验 | 队列日志显示这 3 个 det resume run 均正常结束 |
| ResNet-18 | Test-PilotA-Resume-NonDet | `resume nondet seed=0` | `partial log only / Time NA` | resume 非决定性对照 | 日志存在，但无统一收尾摘要 |

## 9. 仅发现模板或未发现结果工件

- `Adam-resnet32/experiments`：
  - 找到 `cifar100_10`、`cifar100_10_a6/a15/a20/a30`、`cifar100_20`、`tiny_8_8` 的 `command.sh`
  - 当前未找到对应正式 `results/` 输出
- `Adam-resnet32-depth/experiments` 与 `Adam-resnet32-part2/experiments`：
  - 同样存在基准命令模板
  - 实际正式结果已分别落在各自 `results/` 中，上表已整理
- 本次扫描中未发现独立实验输出工件的目录：
  - `Adam-DS-depth`
  - `Adam-new`
  - `Adam-new-section2`
  - `Adam-NSCL-main-origin`

## 10. 当前全局主结论

- `Adam-NSCL-C-Flat` 线最清楚的结论仍然是：
  - direct-attach C-Flat 不适合做主线
  - `Layer-Selective` 有方法信号
  - full 10-task 下 `last_block` 比 `last_stage` 更可信
- `Adam-resnet32-depth` 线当前最强配置是：
  - `last_block + deep_a=50`
  - 3-seed 结果为 `72.30 / -1.56`
- `Adam-resnet32-part1` 的 spectral-budget 主线虽然从 `56.91/-16.87` 改善到了 `61.65/-3.12` 左右，但仍明显落后于 `part1` 内历史 baseline `73.18/-2.16`
- `Adam-resnet32-part2` 的关键结论是：
  - 问题主要来自 trigger / adapter 放开策略
  - 强约束单 adapter 或 budget=1 时可回到 `73.9 / -2.3` 量级
- `Adam-Depth + Spectrum` 线当前最优仍是 `none` baseline `73.76 / -0.88`
- `Adam-NSCL-resnet18+AdNS` 线当前最明显的负面来源是：
  - `task_strength`
  - overlap_core 与 union_lowrank 的 shared-subspace 变体都远弱于 `nscl/sfcl`

## 11. 主要原始总结文件

- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/CFLAT_INTEGRATION_SUMMARY.md`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/CFLAT_OFFICIAL_ALIGNMENT_SUMMARY.md`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/CFLAT_4_DIAG_EXPERIMENTS_SUMMARY.md`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/PLS_CFLAT_D2_SUMMARY.md`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/V1_FULL_SCOPE_V2_SUMMARY.md`
- `/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main/SELECTIVE_G0_SMALLLAMBDA_SUMMARY.md`
- `/home/moshiling/Adam-resnet32-depth/results/auto_depth_opt_cifar100_10/experiment_summary_index.md`
- `/home/moshiling/Adam-resnet32-depth/results/validate_best_lastblock_a50_multiseed/multiseed_summary.md`
- `/home/moshiling/Adam-resnet32-part1/results/ch3_spectral_budget_resnet32_cifar100_10_auto/result_summary.md`
- `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main/results/ablation_unionlowrank_round1/final_report.md`
- `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main/results/final_repro_round/final_report.md`
