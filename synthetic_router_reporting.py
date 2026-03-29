"""Artifact IO and report generation for optimizer comparison runs."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(json_ready(payload), fp, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def collect_trial_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    return [load_json(path) for path in sorted(run_dir.glob("tuning/setting_s*/**/trial_*/trial_summary.json"))]


def collect_best_configs(run_dir: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    best_configs: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("tuning/setting_s*/**/best_config.json")):
        payload = load_json(path)
        best_configs.setdefault(payload["setting"], {})[payload["optimizer_family"]] = payload
    return best_configs


def collect_final_results(run_dir: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    final_results: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("final/setting_s*/*.json")):
        payload = load_json(path)
        final_results.setdefault(payload["setting"], {})[payload["optimizer_family"]] = payload
    return final_results


def build_summary_rows(
    best_configs: Dict[int, Dict[str, Dict[str, Any]]],
    final_results: Dict[int, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows = []
    settings = sorted(set(best_configs.keys()) | set(final_results.keys()))
    for setting in settings:
        families = set(best_configs.get(setting, {}).keys()) | set(final_results.get(setting, {}).keys())
        for family in sorted(families):
            best_payload = best_configs.get(setting, {}).get(family, {})
            final_payload = final_results.get(setting, {}).get(family, {})
            best_summary = best_payload.get("summary", {})
            final_summary = final_payload.get("summary", {})
            best_config = best_payload.get("best_config", {})
            rows.append(
                {
                    "setting": setting,
                    "optimizer_family": family,
                    "best_expert_num": best_config.get("expert_num"),
                    "cv_mean_accuracy": best_summary.get("mean_val_accuracy"),
                    "cv_std_accuracy": best_summary.get("std_val_accuracy"),
                    "cv_mean_loss": best_summary.get("mean_val_loss"),
                    "cv_std_loss": best_summary.get("std_val_loss"),
                    "cv_mean_runtime_seconds": best_summary.get("mean_runtime_seconds"),
                    "final_mean_accuracy": final_summary.get("mean_test_accuracy"),
                    "final_std_accuracy": final_summary.get("std_test_accuracy"),
                    "final_mean_loss": final_summary.get("mean_test_loss"),
                    "final_std_loss": final_summary.get("std_test_loss"),
                    "final_mean_entropy": final_summary.get("mean_test_entropy"),
                    "final_std_entropy": final_summary.get("std_test_entropy"),
                    "final_mean_train_runtime_seconds": final_summary.get("mean_train_runtime_seconds"),
                    "final_std_train_runtime_seconds": final_summary.get("std_train_runtime_seconds"),
                }
            )
    return rows


def build_detailed_trial_rows(trial_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in trial_records:
        summary = record["summary"]
        config = record["config"]
        rows.append(
            {
                "setting": record["setting"],
                "optimizer_family": record["optimizer_family"],
                "trial_index": record["trial_index"],
                "expert_num": config.get("expert_num"),
                "expert_lr": config.get("expert_lr"),
                "expert_momentum": config.get("expert_momentum"),
                "expert_weight_decay": config.get("expert_weight_decay"),
                "router_lr": config.get("router_lr"),
                "router_momentum": config.get("router_momentum"),
                "router_weight_decay": config.get("router_weight_decay"),
                "ns_steps": config.get("ns_steps"),
                "mean_val_accuracy": summary.get("mean_val_accuracy"),
                "std_val_accuracy": summary.get("std_val_accuracy"),
                "mean_val_loss": summary.get("mean_val_loss"),
                "std_val_loss": summary.get("std_val_loss"),
                "mean_val_runtime_seconds": summary.get("mean_runtime_seconds"),
            }
        )
    return rows


def compare_families(final_results: Dict[int, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    wins = {"muon": 0, "original": 0, "tie": 0}
    improvements = []
    expert_counts = {"muon": [], "original": []}

    for _, family_payloads in sorted(final_results.items()):
        muon = family_payloads.get("muon")
        original = family_payloads.get("original")
        if muon:
            expert_counts["muon"].append(muon["best_config"].get("expert_num"))
        if original:
            expert_counts["original"].append(original["best_config"].get("expert_num"))
        if not muon or not original:
            continue

        muon_acc = muon["summary"]["mean_test_accuracy"]
        original_acc = original["summary"]["mean_test_accuracy"]
        improvements.append(muon_acc - original_acc)

        if abs(muon_acc - original_acc) < 1e-9:
            wins["tie"] += 1
        elif muon_acc > original_acc:
            wins["muon"] += 1
        else:
            wins["original"] += 1

    avg_improvement = float(np.mean(improvements)) if improvements else None
    if avg_improvement is None:
        recommendation = "Final comparison results are incomplete."
    elif wins["muon"] > wins["original"] and avg_improvement > 0:
        recommendation = "Muon is the stronger default for this synthetic setup."
    elif wins["original"] > wins["muon"] and avg_improvement < 0:
        recommendation = "The original optimizer family is the stronger default for this synthetic setup."
    else:
        recommendation = "The comparison is mixed; choose based on setting-specific results."

    return {
        "wins": wins,
        "average_muon_minus_original_accuracy": avg_improvement,
        "selected_expert_counts": expert_counts,
        "recommendation": recommendation,
    }


def generate_report(run_dir: Path) -> Dict[str, Any]:
    best_configs = collect_best_configs(run_dir)
    final_results = collect_final_results(run_dir)
    trial_records = collect_trial_summaries(run_dir)

    summary_rows = build_summary_rows(best_configs, final_results)
    detailed_rows = build_detailed_trial_rows(trial_records)
    overall = compare_families(final_results)

    save_json(run_dir / "summary.json", {"rows": summary_rows, "overall": overall})
    save_json(run_dir / "detailed_trials.json", {"rows": detailed_rows})

    write_csv(
        run_dir / "summary.csv",
        summary_rows,
        fieldnames=[
            "setting",
            "optimizer_family",
            "best_expert_num",
            "cv_mean_accuracy",
            "cv_std_accuracy",
            "cv_mean_loss",
            "cv_std_loss",
            "cv_mean_runtime_seconds",
            "final_mean_accuracy",
            "final_std_accuracy",
            "final_mean_loss",
            "final_std_loss",
            "final_mean_entropy",
            "final_std_entropy",
            "final_mean_train_runtime_seconds",
            "final_std_train_runtime_seconds",
        ],
    )
    write_csv(
        run_dir / "detailed_trials.csv",
        detailed_rows,
        fieldnames=[
            "setting",
            "optimizer_family",
            "trial_index",
            "expert_num",
            "expert_lr",
            "expert_momentum",
            "expert_weight_decay",
            "router_lr",
            "router_momentum",
            "router_weight_decay",
            "ns_steps",
            "mean_val_accuracy",
            "std_val_accuracy",
            "mean_val_loss",
            "std_val_loss",
            "mean_val_runtime_seconds",
        ],
    )

    lines = ["# Synthetic Optimizer Comparison Report", ""]
    lines.append(f"- Recommendation: {overall['recommendation']}")
    lines.append(f"- Win counts: {overall['wins']}")
    lines.append(f"- Average Muon minus original accuracy: {overall['average_muon_minus_original_accuracy']}")
    lines.append("")

    settings = sorted(set(best_configs.keys()) | set(final_results.keys()))
    for setting in settings:
        lines.append(f"## Setting s{setting}")
        lines.append("")
        lines.append("| Optimizer | Expert Num | CV Acc | CV Loss | Final Test Acc | Final Entropy | Mean Train Runtime (s) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        families = sorted(set(best_configs.get(setting, {}).keys()) | set(final_results.get(setting, {}).keys()))
        for family in families:
            best_payload = best_configs.get(setting, {}).get(family, {})
            final_payload = final_results.get(setting, {}).get(family, {})
            best_config = best_payload.get("best_config", {})
            best_summary = best_payload.get("summary", {})
            final_summary = final_payload.get("summary", {})

            def fmt(value: Any) -> str:
                if value is None:
                    return "n/a"
                if isinstance(value, float):
                    return f"{value:.4f}"
                return str(value)

            lines.append(
                "| "
                + " | ".join(
                    [
                        family,
                        fmt(best_config.get("expert_num")),
                        fmt(best_summary.get("mean_val_accuracy")),
                        fmt(best_summary.get("mean_val_loss")),
                        fmt(final_summary.get("mean_test_accuracy")),
                        fmt(final_summary.get("mean_test_entropy")),
                        fmt(final_summary.get("mean_train_runtime_seconds")),
                    ]
                )
                + " |"
            )
        lines.append("")

    report_path = run_dir / "report.md"
    ensure_parent(report_path)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary_rows": summary_rows,
        "detailed_rows": detailed_rows,
        "overall": overall,
        "report_path": str(report_path),
    }
