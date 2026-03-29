"""Model definitions and routing metrics for the synthetic MoE experiments."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_router_settings import CLUSTER_NUM, INPUT_DIM, OUT_CHANNEL, PATCH_NUM


def entropy(dispatch: torch.Tensor) -> torch.Tensor:
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
        return torch.stack(
            [torch.sum(x[:, : self.out_channel], 1), torch.sum(x[:, self.out_channel :], 1)]
        ).transpose(1, 0)


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
        self.models = nn.ModuleList(
            [ConvNet(input_dim, out_channel, patch_num, nonlinear=nonlinear) for _ in range(expert_num)]
        )
        self.strategy = strategy
        self.expert_num = expert_num

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        select = self.router(x)
        if self.strategy != "top1":
            raise NotImplementedError("Only top1 routing is supported.")

        gate, index = top1(select)
        mask = F.one_hot(index, self.expert_num).float()
        density = mask.mean(dim=-2)
        density_proxy = select.mean(dim=-2)
        load_balancing_loss = (density_proxy * density).mean() * float(self.expert_num**2)

        combine_tensor = gate[..., None, None] * mask[..., None]
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        select0 = dispatch_tensor.squeeze(-1)

        expert_inputs = torch.einsum("bnd,ben->ebd", x, dispatch_tensor).unsqueeze(2)
        outputs = torch.stack([expert(expert_inputs[idx]) for idx, expert in enumerate(self.models)])
        outputs = torch.einsum("ijk,jil->il", combine_tensor, outputs)
        outputs = F.softmax(outputs, dim=1)
        return outputs, select0, load_balancing_loss


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


def cluster_dispatch_counts(select0: torch.Tensor, cluster_labels: torch.Tensor, cluster_num: int = CLUSTER_NUM) -> torch.Tensor:
    counts = []
    for cluster_id in range(cluster_num):
        mask = cluster_labels == cluster_id
        if mask.any():
            counts.append(select0[mask].sum(dim=0))
        else:
            counts.append(torch.zeros(select0.shape[1], device=select0.device, dtype=select0.dtype))
    return torch.stack(counts)
