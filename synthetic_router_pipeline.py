"""Training, tuning, and final-evaluation flows for synthetic MoE experiments."""

import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from synthetic_router_data import SettingData, load_setting, tensor_subset
from synthetic_router_model import build_model, cluster_dispatch_counts, entropy
from synthetic_router_optimizers import build_optimizers
from synthetic_router_reporting import load_json, save_json
from synthetic_router_search import sample_search_config
from synthetic_router_settings import CLUSTER_NUM


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_collect_plot_diagnostics(model, setting_data: SettingData) -> Dict[str, Any]:
    expert_feature = [[] for _ in range(model.expert_num)]
    expert_center = [[] for _ in range(model.expert_num)]
    router_feature = []
    router_center = []

    with torch.no_grad():
        for cluster in range(CLUSTER_NUM):
            feature_inner = torch.abs(
                torch.matmul(
                    model.router.conv1.weight.squeeze(1).cpu(),
                    setting_data.features[[cluster]].float().transpose(1, 0),
                )
            )
            center_inner = torch.abs(
                torch.matmul(
                    model.router.conv1.weight.squeeze(1).cpu(),
                    setting_data.centers[[cluster]].float().transpose(1, 0),
                )
            )
            router_feature.append(feature_inner.tolist())
            router_center.append(center_inner.tolist())

            for expert_idx in range(model.expert_num):
                feat = torch.max(
                    torch.abs(
                        torch.matmul(
                            model.models[expert_idx].conv1.weight.squeeze(1).cpu(),
                            setting_data.features[[cluster]].float().transpose(1, 0),
                        )
                    )
                )
                cent = torch.max(
                    torch.abs(
                        torch.matmul(
                            model.models[expert_idx].conv1.weight.squeeze(1).cpu(),
                            setting_data.centers[[cluster]].float().transpose(1, 0),
                        )
                    )
                )
                expert_feature[expert_idx].append(float(feat))
                expert_center[expert_idx].append(float(cent))

    return {
        "expert_feature": expert_feature,
        "expert_center": expert_center,
        "router_feature": router_feature,
        "router_center": router_center,
    }


def train_model(
    model,
    criterion: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    train_cluster_labels: torch.Tensor,
    optimizers: Sequence[Optimizer],
    epochs: int,
    load_balancing: bool = False,
    verbose: bool = True,
    early_stopping: bool = True,
    plot: bool = False,
    setting_data: Optional[SettingData] = None,
) -> Dict[str, Any]:
    min_loss = float("inf")
    entropy_record: List[float] = []
    final_loss = None
    final_dispatch = None
    epochs_ran = 0

    train_start = time.perf_counter()
    for epoch in range(epochs):
        for optimizer in optimizers:
            optimizer.zero_grad()

        outputs, select0, load_balancing_loss = model(train_data)
        dispatch = cluster_dispatch_counts(select0, train_cluster_labels, CLUSTER_NUM)
        entropy_record.append(float(entropy(dispatch).item()))

        loss = criterion(outputs, train_labels)
        if load_balancing:
            loss = loss + 0.0001 * load_balancing_loss

        final_loss = float(loss.item())
        final_dispatch = dispatch.detach().cpu()
        epochs_ran = epoch + 1

        should_stop = False
        if early_stopping:
            if loss.item() <= min_loss:
                min_loss = loss.item()
            elif loss > min_loss + 0.02 or loss <= 0.314:
                should_stop = True

        if should_stop:
            break

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch + 1} --- loss: {loss.item():.3f}")

    diagnostics = maybe_collect_plot_diagnostics(model, setting_data) if plot and setting_data is not None else None
    return {
        "epochs_ran": epochs_ran,
        "final_loss": final_loss,
        "final_entropy": entropy_record[-1] if entropy_record else None,
        "dispatch_counts": final_dispatch.tolist() if final_dispatch is not None else None,
        "entropy_record": entropy_record,
        "runtime_seconds": time.perf_counter() - train_start,
        "plot_diagnostics": diagnostics,
    }


def evaluate_model(
    model,
    criterion: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    cluster_labels: torch.Tensor,
    split_name: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    eval_start = time.perf_counter()
    with torch.no_grad():
        outputs, select0, _ = model(data)
        loss = criterion(outputs, labels)
        predicted = torch.max(outputs.data, 1).indices
        correct = (predicted == labels).sum().item()

    accuracy = 100.0 * correct / data.shape[0]
    dispatch = cluster_dispatch_counts(select0, cluster_labels, CLUSTER_NUM)
    if verbose:
        print(f"Accuracy on the {split_name} split ({data.shape[0]} examples): {accuracy:.4f} %")

    return {
        "split": split_name,
        "loss": float(loss.item()),
        "accuracy": float(accuracy),
        "entropy": float(entropy(dispatch).item()),
        "dispatch_counts": dispatch.detach().cpu().tolist(),
        "runtime_seconds": time.perf_counter() - eval_start,
    }


def run_single_experiment(
    setting_data: SettingData,
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
    nonlinear: bool,
    train_indices: Optional[Sequence[int]] = None,
    eval_indices: Optional[Sequence[int]] = None,
    evaluate_on_test: bool = False,
    plot: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    set_seed(seed)

    train_data = tensor_subset(setting_data.train_data, train_indices, device)
    train_labels = tensor_subset(setting_data.train_labels, train_indices, device)
    train_cluster_labels = tensor_subset(setting_data.train_cluster_labels, train_indices, device)

    if evaluate_on_test:
        eval_data = setting_data.test_data.to(device)
        eval_labels = setting_data.test_labels.to(device)
        eval_cluster_labels = setting_data.test_cluster_labels.to(device)
        split_name = "test"
    else:
        eval_data = tensor_subset(setting_data.train_data, eval_indices, device)
        eval_labels = tensor_subset(setting_data.train_labels, eval_indices, device)
        eval_cluster_labels = tensor_subset(setting_data.train_cluster_labels, eval_indices, device)
        split_name = "validation"

    model = build_model(config["expert_num"], nonlinear=nonlinear, max_samples=train_data.shape[0], device=device)
    criterion = nn.CrossEntropyLoss()
    optimizers = build_optimizers(model, config)

    train_metrics = train_model(
        model,
        criterion,
        train_data,
        train_labels,
        train_cluster_labels,
        optimizers,
        config["epochs"],
        load_balancing=config["load_balancing"],
        verbose=not quiet,
        early_stopping=config["early_stopping"],
        plot=plot,
        setting_data=setting_data if plot else None,
    )
    eval_metrics = evaluate_model(
        model,
        criterion,
        eval_data,
        eval_labels,
        eval_cluster_labels,
        split_name=split_name,
        verbose=not quiet,
    )

    return {
        "setting": setting_data.setting,
        "seed": seed,
        "mode": "linear" if not nonlinear else "nonlinear",
        "config": deepcopy(config),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "total_runtime_seconds": train_metrics["runtime_seconds"] + eval_metrics["runtime_seconds"],
    }


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def aggregate_trial_results(fold_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    val_accuracies = [result["eval_metrics"]["accuracy"] for result in fold_results]
    val_losses = [result["eval_metrics"]["loss"] for result in fold_results]
    val_entropies = [result["eval_metrics"]["entropy"] for result in fold_results]
    runtimes = [result["total_runtime_seconds"] for result in fold_results]
    acc_mean, acc_std = mean_std(val_accuracies)
    loss_mean, loss_std = mean_std(val_losses)
    ent_mean, ent_std = mean_std(val_entropies)
    runtime_mean, runtime_std = mean_std(runtimes)
    return {
        "mean_val_accuracy": acc_mean,
        "std_val_accuracy": acc_std,
        "mean_val_loss": loss_mean,
        "std_val_loss": loss_std,
        "mean_val_entropy": ent_mean,
        "std_val_entropy": ent_std,
        "mean_runtime_seconds": runtime_mean,
        "std_runtime_seconds": runtime_std,
    }


def aggregate_final_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    test_accuracies = [result["eval_metrics"]["accuracy"] for result in results]
    test_losses = [result["eval_metrics"]["loss"] for result in results]
    test_entropies = [result["eval_metrics"]["entropy"] for result in results]
    train_runtimes = [result["train_metrics"]["runtime_seconds"] for result in results]
    total_runtimes = [result["total_runtime_seconds"] for result in results]
    acc_mean, acc_std = mean_std(test_accuracies)
    loss_mean, loss_std = mean_std(test_losses)
    ent_mean, ent_std = mean_std(test_entropies)
    train_runtime_mean, train_runtime_std = mean_std(train_runtimes)
    total_runtime_mean, total_runtime_std = mean_std(total_runtimes)
    return {
        "mean_test_accuracy": acc_mean,
        "std_test_accuracy": acc_std,
        "mean_test_loss": loss_mean,
        "std_test_loss": loss_std,
        "mean_test_entropy": ent_mean,
        "std_test_entropy": ent_std,
        "mean_train_runtime_seconds": train_runtime_mean,
        "std_train_runtime_seconds": train_runtime_std,
        "mean_total_runtime_seconds": total_runtime_mean,
        "std_total_runtime_seconds": total_runtime_std,
    }


def trial_dir(run_dir, setting: int, family: str, trial_index: int):
    return run_dir / "tuning" / f"setting_s{setting}" / family / f"trial_{trial_index:03d}"


def select_best_trial(trial_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return min(
        trial_records,
        key=lambda record: (
            -record["summary"]["mean_val_accuracy"],
            record["summary"]["mean_val_loss"],
            record["summary"]["mean_runtime_seconds"],
        ),
    )


def tune_family_for_setting(setting_data: SettingData, family: str, args, device: torch.device) -> Dict[str, Any]:
    setting = setting_data.setting
    family_dir = args.run_dir / "tuning" / f"setting_s{setting}" / family
    family_dir.mkdir(parents=True, exist_ok=True)

    splits = args.cv_splits_by_setting[setting]
    rng = random.Random(args.seed + setting * 100 + (0 if family == "muon" else 1))
    trial_records = []

    for trial_index in range(args.search_budget):
        config = sample_search_config(
            family=family,
            search_space=args.search_space_config,
            rng=rng,
            epochs=args.epochs,
            load_balancing=args.load_balancing,
            early_stopping=not args.no_early_stopping,
        )
        trial_path = trial_dir(args.run_dir, setting, family, trial_index)
        fold_results = []

        for fold_index, (train_idx, val_idx) in enumerate(splits):
            fold_path = trial_path / f"fold_{fold_index:02d}.json"
            if args.resume and fold_path.exists():
                fold_result = load_json(fold_path)
            else:
                fold_result = run_single_experiment(
                    setting_data=setting_data,
                    config=config,
                    seed=args.seed + setting * 10_000 + trial_index * 100 + fold_index,
                    device=device,
                    nonlinear=not args.linear,
                    train_indices=train_idx,
                    eval_indices=val_idx,
                    evaluate_on_test=False,
                    plot=False,
                    quiet=args.quiet,
                )
                fold_result["fold_index"] = fold_index
                save_json(fold_path, fold_result)
            fold_results.append(fold_result)

        trial_record = {
            "setting": setting,
            "optimizer_family": family,
            "trial_index": trial_index,
            "config": config,
            "summary": aggregate_trial_results(fold_results),
            "fold_files": [
                str((trial_path / f"fold_{fold_index:02d}.json").relative_to(args.run_dir))
                for fold_index in range(len(splits))
            ],
        }
        save_json(trial_path / "trial_summary.json", trial_record)
        trial_records.append(trial_record)

    best_trial = select_best_trial(trial_records)
    best_payload = {
        "setting": setting,
        "optimizer_family": family,
        "best_trial_index": best_trial["trial_index"],
        "best_config": best_trial["config"],
        "summary": best_trial["summary"],
    }
    save_json(family_dir / "best_config.json", best_payload)
    return best_payload


def tune_all(args, device: torch.device) -> Dict[int, Dict[str, Dict[str, Any]]]:
    all_best: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for setting in args.settings:
        setting_data = load_setting(setting)
        setting_best = {}
        for family in args.optimizers:
            if not args.quiet:
                print(f"Tuning {family} on setting s{setting}")
            setting_best[family] = tune_family_for_setting(setting_data, family, args, device)
        all_best[setting] = setting_best
    return all_best


def run_final_trials(setting_data: SettingData, best_config: Dict[str, Any], args, device: torch.device) -> Dict[str, Any]:
    family = best_config["optimizer_family"]
    final_dir = args.run_dir / "final" / f"setting_s{setting_data.setting}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{family}.json"

    if args.resume and final_path.exists():
        return load_json(final_path)

    results = []
    for trial_index in range(args.final_trials):
        if not args.quiet:
            print(f"Final evaluation {family} on s{setting_data.setting}: trial {trial_index + 1}/{args.final_trials}")
        results.append(
            run_single_experiment(
                setting_data=setting_data,
                config=best_config["best_config"],
                seed=args.seed + 100_000 + setting_data.setting * 100 + trial_index,
                device=device,
                nonlinear=not args.linear,
                train_indices=None,
                eval_indices=None,
                evaluate_on_test=True,
                plot=False,
                quiet=args.quiet,
            )
        )

    payload = {
        "setting": setting_data.setting,
        "optimizer_family": family,
        "best_config": best_config["best_config"],
        "summary": aggregate_final_results(results),
        "trials": results,
    }
    save_json(final_path, payload)
    return payload
