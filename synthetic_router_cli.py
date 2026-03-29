"""Thin CLI orchestration layer for the synthetic router pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from synthetic_router_data import build_cv_splits, load_setting
from synthetic_router_pipeline import aggregate_final_results, run_final_trials, run_single_experiment, tune_all
from synthetic_router_reporting import generate_report, save_json
from synthetic_router_search import load_search_space, load_yaml
from synthetic_router_settings import DEFAULT_CONFIG_PATH, DEFAULT_OPTIMIZERS


def train_config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    config = {
        "optimizer_family": args.optimizer_family,
        "expert_num": args.expert_num,
        "epochs": args.epochs,
        "load_balancing": args.load_balancing,
        "early_stopping": args.early_stopping,
        "expert_lr": args.expert_lr,
        "expert_momentum": args.expert_momentum,
        "expert_weight_decay": args.expert_weight_decay,
        "router_lr": args.router_lr,
        "router_momentum": args.router_momentum,
        "router_weight_decay": args.router_weight_decay,
    }
    if args.optimizer_family == "muon":
        config["ns_steps"] = args.ns_steps
    return config


def apply_yaml_defaults(parser: argparse.ArgumentParser, config_section: Dict[str, object]) -> None:
    parser.set_defaults(**config_section)


def load_command_config(config_path: Path, command: str) -> Dict[str, object]:
    payload = load_yaml(config_path)
    run = payload.get("run", {})
    command_section = payload.get(command, {})
    optimizer = payload.get("optimizer", {})
    merged = {}
    if isinstance(run, dict):
        merged.update(run)
    if isinstance(command_section, dict):
        merged.update(command_section)
    if command == "train" and isinstance(optimizer, dict):
        merged.update(optimizer)
    return merged


def default_run_dir(args: argparse.Namespace) -> Path:
    mode = "linear" if args.linear else "nonlinear"
    settings = "-".join(str(setting) for setting in args.settings)
    optimizers = "-".join(args.optimizers)
    slug = (
        f"{mode}__settings_{settings}__opts_{optimizers}"
        f"__folds_{args.cv_folds}__budget_{args.search_budget}__seed_{args.seed}"
    )
    return Path("artifacts") / "synthetic_optimizer_comparison" / slug


def command_manifest(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "command": args.command,
        "settings": getattr(args, "settings", [getattr(args, "setting", None)]),
        "optimizers": getattr(args, "optimizers", [getattr(args, "optimizer_family", None)]),
        "seed": args.seed if hasattr(args, "seed") else None,
        "epochs": getattr(args, "epochs", None),
        "cv_folds": getattr(args, "cv_folds", None),
        "search_budget": getattr(args, "search_budget", None),
        "config": str(getattr(args, "config", "")) if getattr(args, "config", None) else None,
        "linear": getattr(args, "linear", False),
        "load_balancing": getattr(args, "load_balancing", False),
        "early_stopping": getattr(args, "early_stopping", None),
        "final_trials": getattr(args, "final_trials", None),
    }


def print_dispatch_counts(dispatch_counts) -> None:
    if dispatch_counts is None:
        return
    for cluster_counts in dispatch_counts:
        print([int(round(value)) for value in cluster_counts])


def add_shared_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=601)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--load-balancing", action="store_true")
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linear", action="store_true", help="Use linear experts instead of nonlinear experts.")
    parser.add_argument("--quiet", action="store_true")


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)


def add_optimizer_hyperparameter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expert-lr", type=float, default=1e-3)
    parser.add_argument("--expert-momentum", type=float, default=0.95)
    parser.add_argument("--expert-weight-decay", type=float, default=0.0)
    parser.add_argument("--router-lr", type=float, default=0.02)
    parser.add_argument("--router-momentum", type=float, default=0.95)
    parser.add_argument("--router-weight-decay", type=float, default=0.0)
    parser.add_argument("--ns-steps", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic MoE optimizer comparison runner")
    add_config_arg(parser)
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run direct training/evaluation.")
    add_config_arg(train_parser)
    train_parser.add_argument("--setting", type=int, choices=[1, 2, 3, 4], default=1)
    train_parser.add_argument("--optimizer-family", choices=["muon", "original"], default="muon")
    train_parser.add_argument("--expert-num", type=int, default=8)
    train_parser.add_argument("--trials", type=int, default=1)
    train_parser.add_argument("--plot", action="store_true")
    train_parser.add_argument("--output-json", type=Path)
    add_shared_training_args(train_parser)
    add_optimizer_hyperparameter_args(train_parser)

    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters with CV.")
    add_config_arg(tune_parser)
    tune_parser.add_argument("--settings", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    tune_parser.add_argument("--optimizers", nargs="+", choices=["muon", "original"], default=DEFAULT_OPTIMIZERS)
    tune_parser.add_argument("--cv-folds", type=int, default=5)
    tune_parser.add_argument("--search-budget", type=int, default=24)
    tune_parser.add_argument("--resume", action="store_true")
    tune_parser.add_argument("--run-dir", type=Path)
    add_shared_training_args(tune_parser)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run tune + final evaluation + report generation.")
    add_config_arg(pipeline_parser)
    pipeline_parser.add_argument("--settings", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    pipeline_parser.add_argument("--optimizers", nargs="+", choices=["muon", "original"], default=DEFAULT_OPTIMIZERS)
    pipeline_parser.add_argument("--cv-folds", type=int, default=5)
    pipeline_parser.add_argument("--search-budget", type=int, default=24)
    pipeline_parser.add_argument("--final-trials", type=int, default=5)
    pipeline_parser.add_argument("--resume", action="store_true")
    pipeline_parser.add_argument("--run-dir", type=Path)
    add_shared_training_args(pipeline_parser)

    report_parser = subparsers.add_parser("report", help="Generate report files from cached artifacts.")
    add_config_arg(report_parser)
    report_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"train", "tune", "pipeline", "report"}
    if not argv or argv[0] not in commands:
        argv = ["train"] + argv

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = base_parser.parse_known_args(argv)
    command = next((token for token in argv if token in commands), "train")

    parser = build_parser()
    config_path = pre_args.config.resolve()
    if command in {"train", "tune", "pipeline"}:
        apply_yaml_defaults(parser, load_command_config(config_path, command))

    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    if args.command in {"tune", "pipeline"}:
        if args.run_dir is None:
            args.run_dir = default_run_dir(args)
        seen = set()
        args.optimizers = [family for family in args.optimizers if not (family in seen or seen.add(family))]
        args.search_space_config = load_search_space(load_yaml(args.config))
        args.cv_splits_by_setting = {setting: build_cv_splits(load_setting(setting), args.cv_folds, args.seed + setting) for setting in args.settings}
    return args


def run_train_command(args: argparse.Namespace, device: torch.device):
    setting_data = load_setting(args.setting)
    config = train_config_from_args(args)
    results = []
    for trial_idx in range(args.trials):
        if not args.quiet:
            mode = "linear" if args.linear else "nonlinear"
            print(
                f"\nTrial {trial_idx + 1}/{args.trials} | setting=s{args.setting} | "
                f"optimizer={args.optimizer_family} | mode={mode} | experts={args.expert_num}"
            )
        result = run_single_experiment(
            setting_data=setting_data,
            config=config,
            seed=args.seed + trial_idx,
            device=device,
            nonlinear=not args.linear,
            train_indices=None,
            eval_indices=None,
            evaluate_on_test=True,
            plot=args.plot,
            quiet=args.quiet,
        )
        results.append(result)
        if not args.quiet:
            print_dispatch_counts(result["train_metrics"]["dispatch_counts"])

    payload = {
        "setting": args.setting,
        "optimizer_family": args.optimizer_family,
        "config": config,
        "trials": results,
        "summary": aggregate_final_results(results),
    }
    if args.output_json is not None:
        save_json(args.output_json, payload)
    return payload


def run_tune_command(args: argparse.Namespace, device: torch.device):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.run_dir / "manifest.json", command_manifest(args))
    if not args.quiet:
        print(f"[Run] tune | run_dir={args.run_dir}", flush=True)
    return tune_all(args, device)


def run_pipeline_command(args: argparse.Namespace, device: torch.device):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.run_dir / "manifest.json", command_manifest(args))
    if not args.quiet:
        print(f"[Run] pipeline | run_dir={args.run_dir}", flush=True)
    best_configs = tune_all(args, device)
    for setting in args.settings:
        setting_data = load_setting(setting)
        for family in args.optimizers:
            run_final_trials(setting_data, best_configs[setting][family], args, device)
    report = generate_report(args.run_dir)
    if not args.quiet:
        print(f"[Run] pipeline | report={report['report_path']}", flush=True)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "train":
        run_train_command(args, device)
    elif args.command == "tune":
        run_tune_command(args, device)
    elif args.command == "pipeline":
        run_pipeline_command(args, device)
    elif args.command == "report":
        payload = generate_report(args.run_dir)
        print(f"Report written to {payload['report_path']}")
    else:
        raise ValueError(f"Unsupported command: {args.command}")
