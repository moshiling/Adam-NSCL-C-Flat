"""Main experiment script for Adam-NSCL + C-Flat on Split CIFAR-100.

Command-line interface
----------------------
All hyper-parameters can be set via ``--key value`` flags.  The most
important ones are:

Adam-NSCL
  --lr            Learning rate (default: 1e-3)
  --epochs        Epochs per task (default: 50)
  --n_tasks       Number of CL tasks (default: 10)
  --n_svd_components  SVD rank per parameter (default: 50)

C-Flat integration
  --use_cflat             Enable full-model C-Flat frontend (flag)
  --use_pls_cflat         Enable PLS-CFlat / selective C-Flat (flag)
  --cflat_rho             SAM perturbation radius ρ (default: 0.05)
  --cflat_lam             Aggregation weight λ; 0 = g0-only fast path
                          (default: 0.0)
  --cflat_target_scope    Scope: all | classifier | deep |
                          deep_plus_classifier |
                          last_block_plus_classifier (default)
  --deep_layer_rule       last_block (default) | last_stage | last_third
  --project_before_perturb  Apply SVD projection before perturbation (flag)
  --project_after_aggregate Apply SVD projection after aggregation (flag)

Experiment orchestration
  --repeat        Number of independent runs (default: 1)
  --seed          Base random seed; run k uses seed + k (default: 42)
  --output_dir    Root output directory (default: ./outputs/<run_name>)
  --run_name      Sub-directory name (default: derived from key params)
  --data_root     CIFAR-100 data directory (default: ./data)
  --device        cuda | cpu (default: auto-detect)
  --batch_size    Mini-batch size (default: 256)
  --verbose       Print per-epoch progress (flag)

Outputs (written to ``output_dir``)
  config_resolved.json   – resolved configuration for full reproducibility
  final_metrics.json     – per-run results + mean/std over repeats
  seed<k>/               – per-run sub-directory with detailed logs

Example
-------
# Baseline (no C-Flat)
python main.py --n_tasks 10 --epochs 50 --repeat 3

# PLS-CFlat, last_block+classifier, λ=0 (recommended config)
python main.py --use_pls_cflat --cflat_target_scope last_block_plus_classifier \\
               --cflat_lam 0 --repeat 3
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import numpy as np

# Allow running from the Adam-NSCL-main directory directly.
sys.path.insert(0, os.path.dirname(__file__))

from datasets.cifar import build_split_cifar100
from networks.resnet import resnet18_cifar
from svd_agent.agent import ContinualLearningAgent


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adam-NSCL + C-Flat on Split CIFAR-100",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dataset / general -----------------------------------------------
    p.add_argument("--data_root", default="./data", help="CIFAR-100 data root")
    p.add_argument("--n_tasks", type=int, default=10, help="Number of CL tasks")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'")

    # ---- Adam-NSCL -------------------------------------------------------
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50, help="Epochs per task")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--n_svd_components", type=int, default=50,
                   help="Max SVD rank stored per parameter")
    p.add_argument("--svd_n_batches", type=int, default=None,
                   help="Batches to use for SVD computation (None = all)")

    # ---- C-Flat ----------------------------------------------------------
    p.add_argument("--use_cflat", action="store_true",
                   help="Enable full-model C-Flat gradient frontend")
    p.add_argument("--use_pls_cflat", action="store_true",
                   help="Enable PLS-CFlat (parameter-layer selective C-Flat)")
    p.add_argument("--cflat_rho", type=float, default=0.05,
                   help="SAM perturbation radius ρ")
    p.add_argument("--cflat_lam", type=float, default=0.0,
                   help="Aggregation weight λ (0 = g0-only fast path)")
    p.add_argument("--cflat_target_scope", default="last_block_plus_classifier",
                   choices=[
                       "all", "classifier", "deep",
                       "deep_plus_classifier", "last_block_plus_classifier",
                   ])
    p.add_argument("--deep_layer_rule", default="last_block",
                   choices=["last_block", "last_stage", "last_third"])
    p.add_argument("--project_before_perturb", action="store_true",
                   help="Apply SVD projection before perturbation direction")
    p.add_argument("--project_after_aggregate", action="store_true",
                   help="Apply SVD projection to aggregated C-Flat gradient")

    # ---- Experiment orchestration ----------------------------------------
    p.add_argument("--repeat", type=int, default=1,
                   help="Number of independent runs (results are mean±std)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed; run k uses seed+k")
    p.add_argument("--output_dir", default=None,
                   help="Root output directory (auto-generated if not set)")
    p.add_argument("--run_name", default=None,
                   help="Sub-directory / experiment name (auto-generated if not set)")
    p.add_argument("--verbose", action="store_true", help="Per-epoch progress")

    return p


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _auto_run_name(args: argparse.Namespace) -> str:
    """Derive a meaningful run name from key parameters."""
    parts = []
    if args.use_pls_cflat:
        parts.append(f"pls_{args.cflat_target_scope}_lam{args.cflat_lam}")
    elif args.use_cflat:
        parts.append(f"cflat_lam{args.cflat_lam}")
    else:
        parts.append("baseline")
    parts.append(f"rho{args.cflat_rho}")
    parts.append(f"ep{args.epochs}_r{args.repeat}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(config: Dict[str, Any], seed: int, run_dir: str) -> Dict[str, Any]:
    """Execute one full CL experiment and return metrics.

    Args:
        config:   Resolved flat configuration dict.
        seed:     Random seed for this run.
        run_dir:  Directory for this run's per-task logs.

    Returns:
        Dict with ``per_task_acc``, ``final_avg_acc``, ``bwt``,
        ``total_time``, and ``seed``.
    """
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = config["device"]
    n_tasks = config["n_tasks"]
    classes_per_task = 100 // n_tasks

    # ---- Data ------------------------------------------------------------
    task_loaders = build_split_cifar100(
        data_root=config["data_root"],
        n_tasks=n_tasks,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=seed,
        download=True,
    )

    # ---- Model + agent ---------------------------------------------------
    model = resnet18_cifar(n_classes=classes_per_task)
    agent = ContinualLearningAgent(config)
    agent.setup(model)

    # ---- Continual learning loop -----------------------------------------
    # acc_matrix[i][j] = accuracy on task j after training task i
    acc_matrix: List[List[float]] = []
    total_time = 0.0

    for task_id, (train_loader, test_loader) in enumerate(task_loaders):
        print(f"\n[Run seed={seed}] === Task {task_id + 1}/{n_tasks} ===")

        # Train
        train_metrics = agent.train_task(task_id, train_loader)
        total_time += train_metrics["time"]

        # Post-task SVD update (skip for last task to save time)
        if task_id < n_tasks - 1:
            agent.after_task(task_id, train_loader)

        # Evaluate on all tasks seen so far
        row: List[float] = []
        for prev_task_id in range(task_id + 1):
            _, prev_test = task_loaders[prev_task_id]
            eval_res = agent.evaluate(prev_task_id, prev_test)
            row.append(eval_res["acc"])
            print(
                f"  Task {prev_task_id} acc: {eval_res['acc']:.4f}"
                if config.get("verbose") else "",
                end="",
            )
        if config.get("verbose"):
            print()
        acc_matrix.append(row)

    # ---- Metrics ---------------------------------------------------------
    # Final average accuracy (acc on each task after seeing all tasks)
    final_accs = [acc_matrix[-1][t] for t in range(n_tasks)]
    final_avg_acc = float(np.mean(final_accs))

    # Backward Transfer: BWT = mean_t (A_{T,t} - A_{t,t})
    # (negative = forgetting)
    bwt_terms = [
        acc_matrix[-1][t] - acc_matrix[t][t]
        for t in range(n_tasks - 1)
    ]
    bwt = float(np.mean(bwt_terms)) if bwt_terms else 0.0

    metrics = {
        "seed": seed,
        "final_avg_acc": final_avg_acc,
        "bwt": bwt,
        "per_task_acc": final_accs,
        "total_time": total_time,
        "acc_matrix": acc_matrix,
    }

    # Save per-run results
    with open(os.path.join(run_dir, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"\n[Run seed={seed}] Final Acc: {final_avg_acc:.4f}  "
        f"BWT: {bwt:.4f}  Time: {total_time:.1f}s"
    )
    return metrics


# ---------------------------------------------------------------------------
# Multi-run orchestration + metrics aggregation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ---- Resolve device --------------------------------------------------
    args.device = _resolve_device(args.device)

    # ---- Build output directory ------------------------------------------
    if args.run_name is None:
        args.run_name = _auto_run_name(args)
    if args.output_dir is None:
        args.output_dir = os.path.join("outputs", args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build resolved config dict (for reproducibility) ---------------
    config: Dict[str, Any] = vars(args)
    config_path = os.path.join(args.output_dir, "config_resolved.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # ---- Multiple independent runs ---------------------------------------
    all_metrics: List[Dict[str, Any]] = []

    for k in range(args.repeat):
        run_seed = args.seed + k
        run_dir = os.path.join(args.output_dir, f"seed{run_seed}")
        metrics = run_once(config, seed=run_seed, run_dir=run_dir)
        all_metrics.append(metrics)

    # ---- Aggregate results (mean ± std) -----------------------------------
    def _mean_std(values: List[float]):
        arr = np.array(values)
        return float(arr.mean()), float(arr.std())

    accs   = [m["final_avg_acc"] for m in all_metrics]
    bwts   = [m["bwt"]           for m in all_metrics]
    times  = [m["total_time"]    for m in all_metrics]

    acc_mean,  acc_std  = _mean_std(accs)
    bwt_mean,  bwt_std  = _mean_std(bwts)
    time_mean, time_std = _mean_std(times)

    final_metrics: Dict[str, Any] = {
        "config": config,
        "repeat": args.repeat,
        "per_run": all_metrics,
        "summary": {
            "final_avg_acc_mean": acc_mean,
            "final_avg_acc_std":  acc_std,
            "bwt_mean":           bwt_mean,
            "bwt_std":            bwt_std,
            "total_time_mean":    time_mean,
            "total_time_std":     time_std,
        },
    }

    metrics_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Final Acc : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  BWT       : {bwt_mean:.4f} ± {bwt_std:.4f}")
    print(f"  Time      : {time_mean:.1f} ± {time_std:.1f} s")
    print("=" * 60)
    print(f"Results saved to {metrics_path}")


if __name__ == "__main__":
    main()
