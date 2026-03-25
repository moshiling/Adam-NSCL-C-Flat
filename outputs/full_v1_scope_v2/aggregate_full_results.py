#!/home/moshiling/miniconda3/envs/torch118/bin/python
import json
import math
import os
import sys
from typing import Any, Dict, List


def _safe_read_json(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: List[float]) -> float | None:
    if not values:
        return None
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def summarize_run(run_dir: str) -> Dict[str, Any]:
    metrics_path = os.path.join(run_dir, "final_metrics.json")
    metrics = _safe_read_json(metrics_path)
    result: Dict[str, Any] = {
        "run_dir": run_dir,
        "name": os.path.basename(run_dir),
        "status": "missing",
    }
    if metrics is None:
        return result

    repeat_metrics = metrics.get("repeat_metrics", [])
    accs = [float(item["final_avg_acc"]) for item in repeat_metrics]
    bwts = [float(item["final_bwt"]) for item in repeat_metrics]
    times = [float(item["wall_clock_seconds"]) for item in repeat_metrics]
    seeds = [item.get("repeat_seed", metrics.get("seed")) for item in repeat_metrics]
    result.update(
        {
            "status": "completed",
            "metrics_path": metrics_path,
            "acc_mean": _mean(accs),
            "acc_std": _std(accs),
            "bwt_mean": _mean(bwts),
            "bwt_std": _std(bwts),
            "time_mean": _mean(times),
            "time_std": _std(times),
            "repeat_count": len(repeat_metrics),
            "repeat_seeds": seeds,
            "command": metrics.get("command"),
        }
    )
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: aggregate_full_results.py <group_dir>", file=sys.stderr)
        return 1

    group_dir = sys.argv[1]
    entries = []
    for name in sorted(os.listdir(group_dir)):
        run_dir = os.path.join(group_dir, name)
        if os.path.isdir(run_dir):
            entries.append(summarize_run(run_dir))

    payload = {
        "group_dir": group_dir,
        "runs": entries,
    }
    out_path = os.path.join(group_dir, "aggregate_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
