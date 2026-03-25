#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path("/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main")
OUTPUT_ROOT = Path("/home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round")
STAGE1_ROOT = OUTPUT_ROOT / "stage1_seed0"
STAGE2_ROOT = OUTPUT_ROOT / "stage2_best_3seed"
PYTHON_BIN = Path("/home/moshiling/miniconda3/envs/torch118/bin/python")

BASE_ARGS = [
    str(PROJECT_ROOT / "main.py"),
    "--schedule", "30", "60", "80",
    "--reg_coef", "100",
    "--model_lr", "1e-4",
    "--head_lr", "1e-3",
    "--svd_lr", "5e-5",
    "--bn_lr", "5e-4",
    "--svd_thres", "10",
    "--model_weight_decay", "5e-5",
    "--agent_type", "svd_based",
    "--agent_name", "svd_based",
    "--dataset", "CIFAR100",
    "--gpuid", "0",
    "--model_optimizer", "Adam",
    "--force_out_dim", "0",
    "--first_split_size", "10",
    "--other_split_size", "10",
    "--batch_size", "32",
    "--model_name", "resnet18",
    "--model_type", "resnet",
    "--workers", "0",
    "--print_freq", "10",
    "--use_pls_cflat",
    "--pls_cflat_mode", "layer_selective",
    "--cflat_rho", "0.02",
    "--pls_cflat_debug",
]

RUNS = {
    "last_block_plus_classifier_lambda0": {
        "scope": "deep_plus_classifier",
        "deep_rule": "last_block",
        "lambda": 0.0,
        "lambda_arg": "0",
        "family": "last_block_plus_classifier",
    },
    "last_block_plus_classifier_lambda0005": {
        "scope": "deep_plus_classifier",
        "deep_rule": "last_block",
        "lambda": 0.005,
        "lambda_arg": "0.005",
        "family": "last_block_plus_classifier",
    },
    "last_block_plus_classifier_lambda001": {
        "scope": "deep_plus_classifier",
        "deep_rule": "last_block",
        "lambda": 0.01,
        "lambda_arg": "0.01",
        "family": "last_block_plus_classifier",
    },
    "deep_last_block_lambda0": {
        "scope": "deep",
        "deep_rule": "last_block",
        "lambda": 0.0,
        "lambda_arg": "0",
        "family": "deep_last_block",
    },
    "deep_last_block_lambda0005": {
        "scope": "deep",
        "deep_rule": "last_block",
        "lambda": 0.005,
        "lambda_arg": "0.005",
        "family": "deep_last_block",
    },
    "deep_last_block_lambda001": {
        "scope": "deep",
        "deep_rule": "last_block",
        "lambda": 0.01,
        "lambda_arg": "0.01",
        "family": "deep_last_block",
    },
}

GROUP_CONFIGS = {
    "A": [
        "last_block_plus_classifier_lambda0",
        "last_block_plus_classifier_lambda0005",
        "last_block_plus_classifier_lambda001",
    ],
    "B": [
        "deep_last_block_lambda0",
        "deep_last_block_lambda0005",
        "deep_last_block_lambda001",
    ],
}

REFERENCE_PATHS = {
    "baseline_full_3seed": Path("/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_a_v1_3seed/baseline_repeat3/final_metrics.json"),
    "d2_direct_attach_3task": Path("/home/moshiling/Adam-NSCL-C-Flat/outputs/cflat_diag_4exp/diag_d2_r002_l002/final_metrics.json"),
    "last_block_plus_classifier_lambda002_old": Path("/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_b_scope_v2/last_block_plus_classifier_seed0/final_metrics.json"),
    "deep_last_block_lambda002_old": Path("/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_b_scope_v2/deep_last_block_seed0/final_metrics.json"),
}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_line(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{now()} {line}\n")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(path: Path):
    data = read_json(path)
    repeat_metrics = data.get("repeat_metrics", [])
    if repeat_metrics:
        wall_times = [float(item["wall_clock_seconds"]) for item in repeat_metrics]
    else:
        wall_times = []
    return {
        "acc_mean": float(data["acc_mean"]),
        "acc_std": float(data.get("acc_std", 0.0)),
        "bwt_mean": float(data["bwt_mean"]),
        "bwt_std": float(data.get("bwt_std", 0.0)),
        "time_mean": float(statistics.mean(wall_times)) if wall_times else None,
        "time_std": float(statistics.pstdev(wall_times)) if len(wall_times) > 1 else 0.0,
        "repeat_metrics": repeat_metrics,
    }


def build_command(gpu: int, run_name: str, repeat: int, seed: int):
    run_cfg = RUNS[run_name]
    run_dir = STAGE1_ROOT / run_name if repeat == 1 else STAGE2_ROOT / f"{run_name}_seed{seed}"
    args = [
        str(PYTHON_BIN), "-u",
        *BASE_ARGS,
        "--repeat", str(repeat),
        "--seed", str(seed),
        "--cflat_target_scope", run_cfg["scope"],
        "--deep_layer_rule", run_cfg["deep_rule"],
        "--cflat_lambda", run_cfg["lambda_arg"],
        "--output_dir", str(run_dir),
    ]
    cmd = "CUDA_VISIBLE_DEVICES={gpu} {args}".format(
        gpu=gpu,
        args=" ".join(shlex.quote(part) for part in args),
    )
    return run_dir, cmd


def write_command(run_dir: Path, cmd: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    command_path = run_dir / "command.sh"
    command_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        f"{cmd}\n",
        encoding="utf-8",
    )
    os.chmod(command_path, 0o755)
    return command_path


def run_logged(command_path: Path, run_log: Path):
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with open(run_log, "a", encoding="utf-8") as logf:
        logf.write(f"{now()} START_COMMAND {command_path}\n")
        logf.flush()
        proc = subprocess.run(
            ["/bin/bash", str(command_path)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        logf.write(f"{now()} EXIT_CODE {proc.returncode}\n")
        logf.flush()
        return proc.returncode


def stage1_queue(group: str, gpu: int):
    stage1_queue_log = STAGE1_ROOT / "queue_status.log"
    heartbeat = STAGE1_ROOT / f"heartbeat_group_{group.lower()}.log"
    append_line(stage1_queue_log, f"queue_started group={group} gpu={gpu}")
    append_line(heartbeat, f"running_group={group} gpu={gpu}")
    for run_name in GROUP_CONFIGS[group]:
        run_dir, cmd = build_command(gpu=gpu, run_name=run_name, repeat=1, seed=0)
        final_metrics = run_dir / "final_metrics.json"
        if final_metrics.exists():
            append_line(stage1_queue_log, f"SKIP {run_name} already_has_final_metrics")
            continue
        command_path = write_command(run_dir, cmd)
        append_line(stage1_queue_log, f"START {run_name} gpu={gpu}")
        append_line(heartbeat, f"running={run_name} gpu={gpu}")
        exit_code = run_logged(command_path, run_dir / "run.log")
        if exit_code == 0 and final_metrics.exists():
            append_line(stage1_queue_log, f"DONE {run_name}")
        else:
            append_line(stage1_queue_log, f"FAIL {run_name} exit_code={exit_code}")
    append_line(stage1_queue_log, f"queue_finished group={group} gpu={gpu}")
    append_line(heartbeat, f"idle_group={group} gpu={gpu}")


def aggregate_stage1():
    rows = []
    for run_name, cfg in RUNS.items():
        run_dir = STAGE1_ROOT / run_name
        final_path = run_dir / "final_metrics.json"
        row = {
            "name": run_name,
            "scope": cfg["scope"],
            "deep_rule": cfg["deep_rule"],
            "lambda": cfg["lambda"],
            "family": cfg["family"],
            "status": "completed" if final_path.exists() else "pending",
            "path": str(run_dir),
        }
        if final_path.exists():
            metrics = extract_metrics(final_path)
            row.update({
                "acc": metrics["acc_mean"],
                "bwt": metrics["bwt_mean"],
                "time": metrics["time_mean"],
            })
            if cfg["family"] == "last_block_plus_classifier":
                ref = extract_metrics(REFERENCE_PATHS["last_block_plus_classifier_lambda002_old"])
            else:
                ref = extract_metrics(REFERENCE_PATHS["deep_last_block_lambda002_old"])
            row["delta_vs_old_lambda002"] = {
                "acc": row["acc"] - ref["acc_mean"],
                "bwt": row["bwt"] - ref["bwt_mean"],
                "time": row["time"] - ref["time_mean"],
            }
        rows.append(row)

    completed = [row for row in rows if row["status"] == "completed"]
    ranking = sorted(completed, key=lambda r: (-r["bwt"], -r["acc"], r["time"]))
    best = ranking[0] if ranking else None
    second = ranking[1] if len(ranking) > 1 else None

    references = {}
    for name, path in REFERENCE_PATHS.items():
        metrics = extract_metrics(path)
        references[name] = metrics
        references[name]["path"] = str(path)
        references[name]["protocol_note"] = (
            "3-task historical reference; not directly full-protocol comparable"
            if name == "d2_direct_attach_3task"
            else "full-protocol reference"
        )

    payload = {
        "generated_at": now(),
        "stage": "stage1_seed0",
        "stage1_results": rows,
        "references": references,
        "ranking": ranking,
        "best_config": best,
        "second_config": second,
    }
    write_json(STAGE1_ROOT / "aggregate_results.json", payload)
    return payload


def run_stage2_best(gpu: int, poll_seconds: int):
    queue_log = STAGE2_ROOT / "queue_status.log"
    heartbeat = STAGE2_ROOT / "heartbeat.log"
    append_line(queue_log, f"stage2_waiter_started gpu={gpu}")
    while True:
        if all((STAGE1_ROOT / run_name / "final_metrics.json").exists() for run_name in RUNS):
            break
        append_line(heartbeat, f"waiting_for_stage1 gpu={gpu}")
        time.sleep(poll_seconds)

    aggregate = aggregate_stage1()
    best = aggregate.get("best_config")
    if not best:
        append_line(queue_log, "stage2_aborted no_best_config")
        return 1

    best_name = best["name"]
    write_json(STAGE2_ROOT / "selected_config.json", best)
    append_line(queue_log, f"stage2_selected {best_name} gpu={gpu}")

    for seed in [0, 1, 2]:
        run_dir, cmd = build_command(gpu=gpu, run_name=best_name, repeat=1, seed=seed)
        run_dir = STAGE2_ROOT / f"{best_name}_seed{seed}"
        cmd = cmd.replace(str(STAGE1_ROOT / best_name), str(run_dir))
        final_metrics = run_dir / "final_metrics.json"
        if final_metrics.exists():
            append_line(queue_log, f"SKIP {best_name}_seed{seed} already_has_final_metrics")
            continue
        command_path = write_command(run_dir, cmd)
        append_line(queue_log, f"START {best_name}_seed{seed} gpu={gpu}")
        append_line(heartbeat, f"running={best_name}_seed{seed} gpu={gpu}")
        exit_code = run_logged(command_path, run_dir / "run.log")
        if exit_code == 0 and final_metrics.exists():
            append_line(queue_log, f"DONE {best_name}_seed{seed}")
        else:
            append_line(queue_log, f"FAIL {best_name}_seed{seed} exit_code={exit_code}")
            return exit_code or 1

    aggregate_stage2()
    append_line(queue_log, f"stage2_finished {best_name} gpu={gpu}")
    append_line(heartbeat, f"idle gpu={gpu}")
    return 0


def aggregate_stage2():
    selected = read_json(STAGE2_ROOT / "selected_config.json")
    best_name = selected["name"]
    rows = []
    accs, bwts, times = [], [], []
    for seed in [0, 1, 2]:
        final_path = STAGE2_ROOT / f"{best_name}_seed{seed}" / "final_metrics.json"
        if not final_path.exists():
            continue
        metrics = extract_metrics(final_path)
        acc = metrics["acc_mean"]
        bwt = metrics["bwt_mean"]
        runtime = metrics["time_mean"]
        accs.append(acc)
        bwts.append(bwt)
        times.append(runtime)
        rows.append({
            "seed": seed,
            "path": str(final_path),
            "acc": acc,
            "bwt": bwt,
            "time": runtime,
        })
    payload = {
        "generated_at": now(),
        "selected_config": selected,
        "seed_results": rows,
        "acc_mean": statistics.mean(accs) if accs else None,
        "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "bwt_mean": statistics.mean(bwts) if bwts else None,
        "bwt_std": statistics.pstdev(bwts) if len(bwts) > 1 else 0.0,
        "time_mean": statistics.mean(times) if times else None,
        "time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }
    write_json(STAGE2_ROOT / "aggregate_results.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_group = sub.add_parser("stage1-group")
    p_group.add_argument("--group", choices=["A", "B"], required=True)
    p_group.add_argument("--gpu", type=int, required=True)

    sub.add_parser("aggregate-stage1")

    p_stage2 = sub.add_parser("stage2-auto")
    p_stage2.add_argument("--gpu", type=int, required=True)
    p_stage2.add_argument("--poll-seconds", type=int, default=180)

    sub.add_parser("aggregate-stage2")

    args = parser.parse_args(argv)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STAGE1_ROOT.mkdir(parents=True, exist_ok=True)
    STAGE2_ROOT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "stage1-group":
        stage1_queue(group=args.group, gpu=args.gpu)
        return 0
    if args.cmd == "aggregate-stage1":
        aggregate_stage1()
        return 0
    if args.cmd == "stage2-auto":
        return run_stage2_best(gpu=args.gpu, poll_seconds=args.poll_seconds)
    if args.cmd == "aggregate-stage2":
        aggregate_stage2()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
