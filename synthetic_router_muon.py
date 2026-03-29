"""
Standalone synthetic MoE experiment runner with optimizer comparison utilities.

This file now supports:
- direct training with either the original optimizer family or Muon
- hyperparameter tuning with random search and stratified cross-validation
- resumable end-to-end optimizer comparison pipelines
- Markdown + CSV/JSON report generation
"""

import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.optim import SGD, _functional
from torch.optim.optimizer import Optimizer, required


CLUSTER_NUM = 4
PATCH_NUM = 4
INPUT_DIM = 200
OUT_CHANNEL = 8
DEFAULT_EXPERT_NUM_OPTIONS = [2, 4, 8, 12]
DEFAULT_OPTIMIZERS = ["muon", "original"]
ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH_SPACE_PATH = ROOT / "synthetic_router_search_space.yaml"


@dataclass
class SettingData:
    setting: int
    train_data: torch.Tensor
    train_labels: torch.Tensor
    test_data: torch.Tensor
    test_labels: torch.Tensor
    centers: torch.Tensor
    features: torch.Tensor
    train_cluster_idx: List[List[int]]
    test_cluster_idx: List[List[int]]
    train_cluster_labels: torch.Tensor
    test_cluster_labels: torch.Tensor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def entropy(dispatch: torch.Tensor) -> torch.Tensor:
    """Compute entropy of the cluster-to-expert dispatch distribution."""
    n_expert = torch.sum(dispatch, axis=0)
    n_total = torch.sum(dispatch)

    prob = dispatch / n_expert
    ent = -torch.nansum(prob * torch.log(prob), axis=0)
    ent = torch.sum((n_expert / n_total) * ent)
    return ent


def top1(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    values, index = tensor.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


class ConvNet(nn.Module):
    def __init__(self, input_dim: int, out_channel: int, patch_num: int, small: bool = True, nonlinear: bool = True):
        super().__init__()
        kernel = int(input_dim / patch_num)
        self.conv1 = nn.Conv1d(1, out_channel * 2, kernel, kernel)
        if small:
            self.conv1.weight = torch.nn.Parameter(self.conv1.weight * 0.001)
            self.conv1.bias = torch.nn.Parameter(self.conv1.bias * 0.001)
        self.out_channel = out_channel
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.nonlinear:
            x = x**3
        x = torch.sum(x, 2)
        output = torch.stack(
            [
                torch.sum(x[:, : self.out_channel], 1),
                torch.sum(x[:, self.out_channel :], 1),
            ]
        ).transpose(1, 0)
        return output


class Router(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, patch_num: int, max_samples: int, noise: bool = True):
        super().__init__()
        kernel = int(input_dim / patch_num)
        self.conv1 = nn.Conv1d(1, out_dim, kernel, kernel, bias=False)
        self.out_dim = out_dim
        self.noise = noise
        self.register_buffer("break_tie_noise", torch.normal(0, 1e-5, size=(max_samples, out_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.conv1.weight = torch.nn.Parameter(self.conv1.weight * 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.sum(x, 2)
        if self.noise and self.training:
            return x + torch.rand(x.shape[0], self.out_dim, device=x.device)
        if self.training:
            return x + self.break_tie_noise[: x.shape[0]].to(x.device)
        return x


class MoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_channel: int,
        patch_num: int,
        expert_num: int,
        max_samples: int,
        strategy: str = "top1",
        nonlinear: bool = True,
    ):
        super().__init__()
        self.router = Router(input_dim, expert_num, patch_num, max_samples=max_samples)
        self.models = nn.ModuleList()
        for _ in range(expert_num):
            self.models.append(ConvNet(input_dim, out_channel, patch_num, nonlinear=nonlinear))
        self.strategy = strategy
        self.expert_num = expert_num

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        select = self.router(x)
        if self.strategy == "top1":
            gate, index = top1(select)
        else:
            raise NotImplementedError("Only top1 routing is supported.")

        mask = F.one_hot(index, self.expert_num).float()
        density = mask.mean(dim=-2)
        density_proxy = select.mean(dim=-2)
        load_balancing_loss = (density_proxy * density).mean() * float(self.expert_num**2)

        combine_tensor = gate[..., None, None] * mask[..., None]
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        select0 = dispatch_tensor.squeeze(-1)

        expert_inputs = torch.einsum("bnd,ben->ebd", x, dispatch_tensor).unsqueeze(2)
        outputs = []
        for expert_idx in range(self.expert_num):
            outputs.append(self.models[expert_idx](expert_inputs[expert_idx]))

        outputs = torch.stack(outputs)
        outputs = torch.einsum("ijk,jil->il", combine_tensor, outputs)
        outputs = F.softmax(outputs, dim=1)
        return outputs, select0, load_balancing_loss


class NormalizedGD(Optimizer):
    def __init__(
        self,
        params,
        expert_num: int,
        lr=required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if expert_num <= 0:
            raise ValueError("expert_num must be positive")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        self.expert_num = expert_num

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = list(group["params"])
            if len(params) % self.expert_num != 0:
                raise ValueError(
                    f"Expected params to be divisible by expert_num={self.expert_num}, got {len(params)} params."
                )

            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            per_expert_num = int(len(params) / self.expert_num)
            per_expert_norm = [0 for _ in range(self.expert_num)]
            for expert_idx in range(self.expert_num):
                for param_idx in range(expert_idx * per_expert_num, (expert_idx + 1) * per_expert_num):
                    param = params[param_idx]
                    if param.grad is not None:
                        per_expert_norm[expert_idx] += param.grad.norm()

            for idx, param in enumerate(params):
                if param.grad is None:
                    continue

                expert_idx = idx // per_expert_num
                if per_expert_norm[expert_idx] != 0:
                    param.grad /= per_expert_norm[expert_idx]

                params_with_grad.append(param)
                d_p_list.append(param.grad)

                state = self.state[param]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

            _functional.sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=False,
            )

            for param, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[param]
                state["momentum_buffer"] = momentum_buffer

        return loss


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int) -> torch.Tensor:
    """Moonlight-style Newton-Schulz orthogonalization for a 2D update matrix."""
    assert len(grad.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    orth = grad.bfloat16() if grad.is_cuda else grad.float()
    if grad.size(0) > grad.size(1):
        orth = orth.T
    orth = orth / (orth.norm() + 1e-7)
    for _ in range(steps):
        mat_a = orth @ orth.T
        mat_b = b * mat_a + c * mat_a @ mat_a
        orth = a * orth + mat_b @ orth
    if grad.size(0) > grad.size(1):
        orth = orth.T
    return orth.to(grad.dtype)


class Muon(Optimizer):
    """
    Adapted from Moonlight's toy_train.py for single-device use in this repo.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.1,
        muon_params=None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params=None,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        params = list(muon_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)
        for param in muon_params:
            if param.ndim < 2:
                raise ValueError(f"Muon expects at least 2D params, got shape {tuple(param.shape)}")
            self.state[param]["use_muon"] = True
        for param in adamw_params:
            self.state[param]["use_muon"] = False

    @staticmethod
    def _matrix_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
        return shape[:1] + (math.prod(shape[1:]),) if len(shape) > 2 else shape

    def adjust_lr_for_muon(self, lr: float, param_shape: Tuple[int, ...]) -> float:
        rows, cols = self._matrix_shape(tuple(param_shape))
        adjusted_ratio = 0.2 * math.sqrt(max(rows, cols))
        return lr * adjusted_ratio

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            muon_params = [param for param in group["params"] if self.state[param]["use_muon"]]
            for param in muon_params:
                grad = param.grad
                if grad is None:
                    continue

                if grad.ndim > 2:
                    grad = grad.view(grad.size(0), -1)
                if grad.ndim != 2:
                    raise ValueError(f"Muon expects a matrix update, got ndim={grad.ndim}")

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(grad)
                if group["nesterov"]:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

                update = zeropower_via_newtonschulz5(grad, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, tuple(param.shape))

                param.data.mul_(1 - lr * wd)
                param.data.add_(update.view_as(param.data), alpha=-adjusted_lr)

            adamw_params = [param for param in group["params"] if not self.state[param]["use_muon"]]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for param in adamw_params:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(grad)
                    state["moment2"] = torch.zeros_like(grad)

                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                buf1.lerp_(grad, 1 - beta1)
                buf2.lerp_(grad.square(), 1 - beta2)
                grad = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / (bias_correction2**0.5)

                param.data.mul_(1 - lr * wd)
                param.data.add_(grad, alpha=-lr / scale)

        return loss


def build_cluster_labels(num_samples: int, cluster_idx: Sequence[Sequence[int]]) -> torch.Tensor:
    labels = torch.empty(num_samples, dtype=torch.long)
    for cluster_id, indices in enumerate(cluster_idx):
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        labels[idx_tensor] = cluster_id
    return labels


def load_setting(setting: int) -> SettingData:
    setting_dir = ROOT / f"synthetic_data_s{setting}"

    train_data = torch.load(setting_dir / "train_data.pt", map_location="cpu")
    train_labels = torch.load(setting_dir / "train_labels.pt", map_location="cpu").long()
    test_data = torch.load(setting_dir / "test_data.pt", map_location="cpu")
    test_labels = torch.load(setting_dir / "test_labels.pt", map_location="cpu").long()
    centers = torch.load(setting_dir / "centers.pt", map_location="cpu")
    features = torch.load(setting_dir / "features.pt", map_location="cpu")

    with open(setting_dir / "train_cluster", "rb") as fp:
        train_cluster_idx = pickle.load(fp)
    with open(setting_dir / "test_cluster", "rb") as fp:
        test_cluster_idx = pickle.load(fp)

    train_cluster_labels = build_cluster_labels(train_data.shape[0], train_cluster_idx)
    test_cluster_labels = build_cluster_labels(test_data.shape[0], test_cluster_idx)

    return SettingData(
        setting=setting,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        centers=centers,
        features=features,
        train_cluster_idx=train_cluster_idx,
        test_cluster_idx=test_cluster_idx,
        train_cluster_labels=train_cluster_labels,
        test_cluster_labels=test_cluster_labels,
    )


def build_model(expert_num: int, nonlinear: bool, max_samples: int, device: torch.device) -> MoE:
    model = MoE(
        INPUT_DIM,
        OUT_CHANNEL,
        PATCH_NUM,
        expert_num=expert_num,
        max_samples=max_samples,
        strategy="top1",
        nonlinear=nonlinear,
    )
    return model.to(device)


def split_muon_params(parameters) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    muon_params = []
    adamw_params = []
    for param in parameters:
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_optimizers(model: MoE, config: Dict[str, Any]) -> List[Optimizer]:
    family = config["optimizer_family"]
    if family == "original":
        expert_optimizer = NormalizedGD(
            model.models.parameters(),
            expert_num=model.expert_num,
            lr=config["expert_lr"],
            momentum=config["expert_momentum"],
            weight_decay=config["expert_weight_decay"],
        )
        router_optimizer = SGD(
            model.router.parameters(),
            lr=config["router_lr"],
            momentum=config["router_momentum"],
            weight_decay=config["router_weight_decay"],
        )
        return [expert_optimizer, router_optimizer]

    if family == "muon":
        expert_muon_params, expert_adamw_params = split_muon_params(model.models.parameters())
        router_muon_params, router_adamw_params = split_muon_params(model.router.parameters())

        expert_optimizer = Muon(
            lr=config["expert_lr"],
            wd=config["expert_weight_decay"],
            muon_params=expert_muon_params,
            adamw_params=expert_adamw_params,
            momentum=config["expert_momentum"],
            ns_steps=config["ns_steps"],
        )
        router_optimizer = Muon(
            lr=config["router_lr"],
            wd=config["router_weight_decay"],
            muon_params=router_muon_params,
            adamw_params=router_adamw_params,
            momentum=config["router_momentum"],
            ns_steps=config["ns_steps"],
        )
        return [expert_optimizer, router_optimizer]

    raise ValueError(f"Unsupported optimizer family: {family}")


def cluster_dispatch_counts(select0: torch.Tensor, cluster_labels: torch.Tensor, cluster_num: int) -> torch.Tensor:
    counts = []
    for cluster_id in range(cluster_num):
        mask = cluster_labels == cluster_id
        if mask.any():
            counts.append(select0[mask].sum(dim=0))
        else:
            counts.append(torch.zeros(select0.shape[1], device=select0.device, dtype=select0.dtype))
    return torch.stack(counts)


def maybe_collect_plot_diagnostics(model: MoE, setting_data: SettingData) -> Dict[str, Any]:
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


def tensor_subset(tensor: torch.Tensor, indices: Optional[Sequence[int]], device: torch.device) -> torch.Tensor:
    if indices is None:
        return tensor.to(device)
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return tensor.index_select(0, index_tensor).to(device)


def train_model(
    model: MoE,
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
        ent = entropy(dispatch)
        entropy_record.append(float(ent.item()))

        if load_balancing:
            loss = criterion(outputs, train_labels) + 0.0001 * load_balancing_loss
        else:
            loss = criterion(outputs, train_labels)

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

    train_runtime = time.perf_counter() - train_start
    diagnostics = None
    if plot and setting_data is not None:
        diagnostics = maybe_collect_plot_diagnostics(model, setting_data)

    return {
        "epochs_ran": epochs_ran,
        "final_loss": final_loss,
        "final_entropy": entropy_record[-1] if entropy_record else None,
        "dispatch_counts": final_dispatch.tolist() if final_dispatch is not None else None,
        "entropy_record": entropy_record,
        "runtime_seconds": train_runtime,
        "plot_diagnostics": diagnostics,
    }


def evaluate_model(
    model: MoE,
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
    eval_runtime = time.perf_counter() - eval_start

    if verbose:
        print(f"Accuracy on the {split_name} split ({data.shape[0]} examples): {accuracy:.4f} %")

    return {
        "split": split_name,
        "loss": float(loss.item()),
        "accuracy": float(accuracy),
        "entropy": float(entropy(dispatch).item()),
        "dispatch_counts": dispatch.detach().cpu().tolist(),
        "runtime_seconds": eval_runtime,
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

    model = build_model(
        expert_num=config["expert_num"],
        nonlinear=nonlinear,
        max_samples=train_data.shape[0],
        device=device,
    )
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


def build_stratify_labels(setting_data: SettingData) -> np.ndarray:
    return (setting_data.train_labels.numpy() * CLUSTER_NUM + setting_data.train_cluster_labels.numpy()).astype(np.int64)


def build_cv_splits(setting_data: SettingData, cv_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    stratify_labels = build_stratify_labels(setting_data)
    _, counts = np.unique(stratify_labels, return_counts=True)
    min_count = int(counts.min())
    if min_count < cv_folds:
        raise ValueError(
            f"Cannot build {cv_folds} stratified folds for setting s{setting_data.setting}; "
            f"smallest bucket has only {min_count} samples."
        )
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    dummy_x = np.zeros_like(stratify_labels)
    return list(splitter.split(dummy_x, stratify_labels))


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at top level in {path}")
    return payload


def validate_distribution_spec(name: str, spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"Search space '{name}' must be a mapping.")
    spec_type = spec.get("type")
    if spec_type not in {"choice", "uniform", "log_uniform"}:
        raise ValueError(f"Search space '{name}' has unsupported type '{spec_type}'.")
    if spec_type == "choice":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Search space '{name}' choice spec must define non-empty 'values'.")
    else:
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            raise ValueError(f"Search space '{name}' {spec_type} spec must define 'low' and 'high'.")
        if float(low) >= float(high):
            raise ValueError(f"Search space '{name}' requires low < high.")
    return spec


def load_search_space(path: Path) -> Dict[str, Any]:
    payload = load_yaml(path)
    shared = payload.get("shared")
    families = payload.get("families")
    if not isinstance(shared, dict):
        raise ValueError("Search space YAML must contain a 'shared' mapping.")
    if not isinstance(families, dict):
        raise ValueError("Search space YAML must contain a 'families' mapping.")

    expert_num_options = shared.get("expert_num_options")
    if not isinstance(expert_num_options, list) or not expert_num_options:
        raise ValueError("'shared.expert_num_options' must be a non-empty list.")

    required_shared = [
        "expert_lr",
        "expert_momentum",
        "expert_weight_decay",
        "router_momentum",
        "router_weight_decay",
    ]
    for name in required_shared:
        validate_distribution_spec(f"shared.{name}", shared.get(name))

    for family in DEFAULT_OPTIMIZERS:
        family_spec = families.get(family)
        if not isinstance(family_spec, dict):
            raise ValueError(f"Search space YAML must define families.{family}.")
        validate_distribution_spec(f"families.{family}.router_lr", family_spec.get("router_lr"))
        if family == "muon":
            validate_distribution_spec(f"families.{family}.ns_steps", family_spec.get("ns_steps"))

    return {
        "path": path,
        "shared": shared,
        "families": families,
    }


def sample_from_spec(name: str, spec: Dict[str, Any], rng: random.Random) -> Any:
    spec = validate_distribution_spec(name, spec)
    spec_type = spec["type"]
    if spec_type == "choice":
        return rng.choice(spec["values"])
    if spec_type == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if spec_type == "log_uniform":
        return log_uniform(rng, float(spec["low"]), float(spec["high"]))
    raise ValueError(f"Unsupported spec type '{spec_type}' for '{name}'.")


def sample_search_config(
    family: str,
    search_space: Dict[str, Any],
    rng: random.Random,
    epochs: int,
    load_balancing: bool,
    early_stopping: bool,
) -> Dict[str, Any]:
    shared = search_space["shared"]
    family_space = search_space["families"][family]
    config = {
        "optimizer_family": family,
        "expert_num": int(rng.choice(list(shared["expert_num_options"]))),
        "epochs": epochs,
        "load_balancing": load_balancing,
        "early_stopping": early_stopping,
        "expert_lr": round(float(sample_from_spec("shared.expert_lr", shared["expert_lr"], rng)), 8),
        "expert_momentum": round(float(sample_from_spec("shared.expert_momentum", shared["expert_momentum"], rng)), 6),
        "expert_weight_decay": round(
            float(sample_from_spec("shared.expert_weight_decay", shared["expert_weight_decay"], rng)), 8
        ),
        "router_momentum": round(float(sample_from_spec("shared.router_momentum", shared["router_momentum"], rng)), 6),
        "router_weight_decay": round(
            float(sample_from_spec("shared.router_weight_decay", shared["router_weight_decay"], rng)), 8
        ),
    }
    config["router_lr"] = round(float(sample_from_spec(f"families.{family}.router_lr", family_space["router_lr"], rng)), 8)
    if family == "muon":
        config["ns_steps"] = int(sample_from_spec("families.muon.ns_steps", family_space["ns_steps"], rng))
    return config


def train_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    config = {
        "optimizer_family": args.optimizer_family,
        "expert_num": args.expert_num,
        "epochs": args.epochs,
        "load_balancing": args.load_balancing,
        "early_stopping": not args.no_early_stopping,
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


def default_run_dir(args: argparse.Namespace) -> Path:
    mode = "linear" if args.linear else "nonlinear"
    settings = "-".join(str(setting) for setting in args.settings)
    optimizers = "-".join(args.optimizers)
    slug = (
        f"{mode}__settings_{settings}__opts_{optimizers}"
        f"__folds_{args.cv_folds}__budget_{args.search_budget}__seed_{args.seed}"
    )
    return ROOT / "artifacts" / "synthetic_optimizer_comparison" / slug


def command_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "command": args.command,
        "settings": getattr(args, "settings", [getattr(args, "setting", None)]),
        "optimizers": getattr(args, "optimizers", [getattr(args, "optimizer_family", None)]),
        "seed": args.seed if hasattr(args, "seed") else None,
        "epochs": getattr(args, "epochs", None),
        "cv_folds": getattr(args, "cv_folds", None),
        "search_budget": getattr(args, "search_budget", None),
        "expert_num_options": getattr(args, "expert_num_options", None),
        "search_space": str(getattr(args, "search_space", "")) if getattr(args, "search_space", None) else None,
        "linear": getattr(args, "linear", False),
        "load_balancing": getattr(args, "load_balancing", False),
        "no_early_stopping": getattr(args, "no_early_stopping", False),
        "final_trials": getattr(args, "final_trials", None),
    }


def trial_dir(run_dir: Path, setting: int, family: str, trial_index: int) -> Path:
    return run_dir / "tuning" / f"setting_s{setting}" / family / f"trial_{trial_index:03d}"


def select_best_trial(trial_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def score(record: Dict[str, Any]) -> Tuple[float, float, float]:
        summary = record["summary"]
        return (
            -summary["mean_val_accuracy"],
            summary["mean_val_loss"],
            summary["mean_runtime_seconds"],
        )

    return min(trial_records, key=score)


def tune_family_for_setting(
    setting_data: SettingData,
    family: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    setting = setting_data.setting
    family_dir = args.run_dir / "tuning" / f"setting_s{setting}" / family
    family_dir.mkdir(parents=True, exist_ok=True)

    splits = build_cv_splits(setting_data, args.cv_folds, seed=args.seed + setting)
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
        trial_summary_path = trial_path / "trial_summary.json"
        fold_results = []

        for fold_index, (train_idx, val_idx) in enumerate(splits):
            fold_path = trial_path / f"fold_{fold_index:02d}.json"
            if args.resume and fold_path.exists():
                fold_result = load_json(fold_path)
            else:
                fold_seed = args.seed + setting * 10_000 + trial_index * 100 + fold_index
                fold_result = run_single_experiment(
                    setting_data=setting_data,
                    config=config,
                    seed=fold_seed,
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

        summary = aggregate_trial_results(fold_results)
        trial_record = {
            "setting": setting,
            "optimizer_family": family,
            "trial_index": trial_index,
            "config": config,
            "summary": summary,
            "fold_files": [
                str((trial_path / f"fold_{fold_index:02d}.json").relative_to(args.run_dir))
                for fold_index in range(len(splits))
            ],
        }
        save_json(trial_summary_path, trial_record)
        trial_records.append(trial_record)

    best_trial = select_best_trial(trial_records)
    best_config_payload = {
        "setting": setting,
        "optimizer_family": family,
        "best_trial_index": best_trial["trial_index"],
        "best_config": best_trial["config"],
        "summary": best_trial["summary"],
    }
    save_json(family_dir / "best_config.json", best_config_payload)
    return best_config_payload


def tune_all(args: argparse.Namespace, device: torch.device) -> Dict[int, Dict[str, Dict[str, Any]]]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.run_dir / "manifest.json", command_manifest(args))

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


def run_final_trials(
    setting_data: SettingData,
    best_config: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    family = best_config["optimizer_family"]
    setting = setting_data.setting
    final_dir = args.run_dir / "final" / f"setting_s{setting}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{family}.json"

    if args.resume and final_path.exists():
        return load_json(final_path)

    results = []
    for trial_index in range(args.final_trials):
        seed = args.seed + 100_000 + setting * 100 + trial_index
        if not args.quiet:
            print(f"Final evaluation {family} on s{setting}: trial {trial_index + 1}/{args.final_trials}")
        result = run_single_experiment(
            setting_data=setting_data,
            config=best_config["best_config"],
            seed=seed,
            device=device,
            nonlinear=not args.linear,
            train_indices=None,
            eval_indices=None,
            evaluate_on_test=True,
            plot=False,
            quiet=args.quiet,
        )
        results.append(result)

    payload = {
        "setting": setting,
        "optimizer_family": family,
        "best_config": best_config["best_config"],
        "summary": aggregate_final_results(results),
        "trials": results,
    }
    save_json(final_path, payload)
    return payload


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def collect_trial_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    trial_records = []
    for trial_summary_path in sorted(run_dir.glob("tuning/setting_s*/**/trial_*/trial_summary.json")):
        trial_records.append(load_json(trial_summary_path))
    return trial_records


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


def print_dispatch_counts(dispatch_counts: Optional[Sequence[Sequence[float]]]) -> None:
    if dispatch_counts is None:
        return
    for cluster_counts in dispatch_counts:
        rounded = [int(round(value)) for value in cluster_counts]
        print(rounded)


def add_shared_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=601)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--load-balancing", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--linear", action="store_true", help="Use linear experts instead of nonlinear experts.")
    parser.add_argument("--quiet", action="store_true")


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
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run direct training/evaluation.")
    train_parser.add_argument("--setting", type=int, choices=[1, 2, 3, 4], default=1)
    train_parser.add_argument("--optimizer-family", choices=["muon", "original"], default="muon")
    train_parser.add_argument("--expert-num", type=int, default=8)
    train_parser.add_argument("--trials", type=int, default=1)
    train_parser.add_argument("--plot", action="store_true")
    train_parser.add_argument("--output-json", type=Path)
    add_shared_training_args(train_parser)
    add_optimizer_hyperparameter_args(train_parser)

    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters with CV.")
    tune_parser.add_argument("--settings", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    tune_parser.add_argument("--optimizers", nargs="+", choices=["muon", "original"], default=DEFAULT_OPTIMIZERS)
    tune_parser.add_argument("--cv-folds", type=int, default=5)
    tune_parser.add_argument("--search-budget", type=int, default=24)
    tune_parser.add_argument("--search-space", type=Path, default=DEFAULT_SEARCH_SPACE_PATH)
    tune_parser.add_argument("--resume", action="store_true")
    tune_parser.add_argument("--run-dir", type=Path)
    add_shared_training_args(tune_parser)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run tune + final evaluation + report generation.")
    pipeline_parser.add_argument("--settings", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    pipeline_parser.add_argument("--optimizers", nargs="+", choices=["muon", "original"], default=DEFAULT_OPTIMIZERS)
    pipeline_parser.add_argument("--cv-folds", type=int, default=5)
    pipeline_parser.add_argument("--search-budget", type=int, default=24)
    pipeline_parser.add_argument("--search-space", type=Path, default=DEFAULT_SEARCH_SPACE_PATH)
    pipeline_parser.add_argument("--final-trials", type=int, default=5)
    pipeline_parser.add_argument("--resume", action="store_true")
    pipeline_parser.add_argument("--run-dir", type=Path)
    add_shared_training_args(pipeline_parser)

    report_parser = subparsers.add_parser("report", help="Generate report files from cached artifacts.")
    report_parser.add_argument("--run-dir", type=Path, required=True)

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    commands = {"train", "tune", "pipeline", "report"}
    argv = list(argv)
    if not argv or argv[0] not in commands:
        argv = ["train"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"tune", "pipeline"} and args.run_dir is None:
        args.run_dir = default_run_dir(args)

    if hasattr(args, "optimizers"):
        seen = set()
        ordered = []
        for family in args.optimizers:
            if family not in seen:
                seen.add(family)
                ordered.append(family)
        args.optimizers = ordered

    if args.command in {"tune", "pipeline"}:
        args.search_space = args.search_space.resolve()
        args.search_space_config = load_search_space(args.search_space)

    return args


def run_train_command(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    setting_data = load_setting(args.setting)
    config = train_config_from_args(args)
    results = []

    for trial_idx in range(args.trials):
        seed = args.seed + trial_idx
        if not args.quiet:
            mode = "linear" if args.linear else "nonlinear"
            print(
                f"\nTrial {trial_idx + 1}/{args.trials} | setting=s{args.setting} | "
                f"optimizer={args.optimizer_family} | mode={mode} | experts={args.expert_num}"
            )
        result = run_single_experiment(
            setting_data=setting_data,
            config=config,
            seed=seed,
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

    summary = aggregate_final_results(results)
    payload = {
        "setting": args.setting,
        "optimizer_family": args.optimizer_family,
        "config": config,
        "trials": results,
        "summary": summary,
    }

    if args.output_json is not None:
        save_json(args.output_json, payload)

    if args.trials > 1 and not args.quiet:
        print()
        print(f"Average test accuracy: {summary['mean_test_accuracy']:.2f}")
        print(f"Standard deviation: {summary['std_test_accuracy']:.2f}")
        print()
        print(f"Average dispatch entropy: {summary['mean_test_entropy']:.3f}")
        print(f"Standard deviation: {summary['std_test_entropy']:.3f}")

    return payload


def run_pipeline_command(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    best_configs = tune_all(args, device)
    final_payloads: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for setting in args.settings:
        setting_data = load_setting(setting)
        final_payloads[setting] = {}
        for family in args.optimizers:
            best_payload = best_configs[setting][family]
            final_payloads[setting][family] = run_final_trials(setting_data, best_payload, args, device)

    report_payload = generate_report(args.run_dir)
    return {
        "best_configs": best_configs,
        "final_results": final_payloads,
        "report": report_payload,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "train":
        run_train_command(args, device)
        return

    if args.command == "tune":
        tune_all(args, device)
        return

    if args.command == "pipeline":
        run_pipeline_command(args, device)
        return

    if args.command == "report":
        payload = generate_report(args.run_dir)
        print(f"Report written to {payload['report_path']}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
