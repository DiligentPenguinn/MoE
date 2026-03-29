"""
Standalone synthetic MoE experiment with Muon on the router layer.

This script is adapted from the synthetic demo notebooks in this repository and
the Muon implementation pattern in MoonshotAI Moonlight's `examples/toy_train.py`:
https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
"""

import argparse
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import _functional
from torch.optim.optimizer import Optimizer, required


DATA_NUM = 16000
CLUSTER_NUM = 4
EXPERT_NUM = 8
PATCH_NUM = 4
PATCH_LEN = 50

ROOT = Path(__file__).resolve().parent
training_data = None
training_labels = None
test_data = None
test_labels = None
centers = None
features = None
train_cluster_idx = None
test_cluster_idx = None


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic MoE with Muon router")
    parser.add_argument("--setting", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--epochs", type=int, default=601)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--expert-lr", type=float, default=1e-3)
    parser.add_argument("--expert-momentum", type=float, default=0.95)
    parser.add_argument("--expert-weight-decay", type=float, default=0.0)
    parser.add_argument("--router-lr", type=float, default=0.02)
    parser.add_argument("--router-weight-decay", type=float, default=0.0)
    parser.add_argument("--router-momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--load-balancing", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--linear", action="store_true", help="Use linear experts instead of nonlinear experts")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def entropy(dispatch):
    """Computes entropy of label distribution."""
    n_expert = torch.sum(dispatch, axis=0)
    n_total = torch.sum(dispatch)

    prob = dispatch / n_expert
    ent = -torch.nansum(prob * torch.log(prob), axis=0)
    ent = torch.sum((n_expert / n_total) * ent)

    return ent


class ConvNet(nn.Module):
    def __init__(self, input_dim, out_channel, patch_num, small=True, nonlinear=True):
        super().__init__()
        self.conv1 = nn.Conv1d(1, out_channel * 2, int(input_dim / patch_num), int(input_dim / patch_num))
        if small:
            self.conv1.weight = torch.nn.Parameter(self.conv1.weight * 0.001)
            self.conv1.bias = torch.nn.Parameter(self.conv1.bias * 0.001)
        self.out_channel = out_channel
        self.nonlinear = nonlinear

    def forward(self, x):
        x = self.conv1(x)
        if self.nonlinear:
            x = x**3
        x = torch.sum(x, 2)
        output = torch.stack([torch.sum(x[:, : self.out_channel], 1), torch.sum(x[:, self.out_channel :], 1)]).transpose(1, 0)
        return output


def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


class Router(nn.Module):
    def __init__(self, input_dim, out_dim, patch_num, noise=True):
        super().__init__()
        self.conv1 = nn.Conv1d(1, out_dim, int(input_dim / patch_num), int(input_dim / patch_num), bias=False)
        self.out_dim = out_dim
        self.noise = noise
        self.register_buffer("break_tie_noise", torch.normal(0, 1e-5, size=(DATA_NUM, EXPERT_NUM)))
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight = torch.nn.Parameter(self.conv1.weight * 0)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sum(x, 2)
        if self.noise and self.training:
            return x + torch.rand(x.shape[0], self.out_dim, device=x.device)
        if self.training:
            return x + self.break_tie_noise[: x.shape[0]].to(x.device)
        return x


class MoE(nn.Module):
    def __init__(self, input_dim, out_channel, patch_num, expert_num, strategy="top1", nonlinear=True):
        super().__init__()
        self.router = Router(input_dim, expert_num, patch_num)
        self.models = nn.ModuleList()
        for _ in range(expert_num):
            self.models.append(ConvNet(input_dim, out_channel, patch_num, nonlinear=nonlinear))
        self.strategy = strategy
        self.expert_num = expert_num

    def forward(self, x):
        select = self.router(x)
        if self.strategy == "top1":
            gate, index = top1(select)
        else:
            raise NotImplementedError("Only top1 routing is supported in this script.")

        mask = F.one_hot(index, self.expert_num).float()

        density = mask.mean(dim=-2)
        density_proxy = select.mean(dim=-2)
        loss = (density_proxy * density).mean() * float(self.expert_num**2)

        mask_flat = mask.sum(dim=-1)
        combine_tensor = gate[..., None, None] * mask_flat[..., None, None] * F.one_hot(index, self.expert_num)[..., None]
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        select0 = dispatch_tensor.squeeze(-1)

        expert_inputs = torch.einsum("bnd,ben->ebd", x, dispatch_tensor).unsqueeze(2)

        output = []
        for i in range(self.expert_num):
            output.append(self.models[i](expert_inputs[i]))

        output = torch.stack(output)
        output = torch.einsum("ijk,jil->il", combine_tensor, output)
        output = F.softmax(output, dim=1)

        return output, select0, loss


class NormalizedGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

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
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            per_expert_num = int(len(group["params"]) / EXPERT_NUM)
            per_expert_norm = [0 for _ in range(EXPERT_NUM)]
            for i in range(EXPERT_NUM):
                for j in range(i * per_expert_num, (i + 1) * per_expert_num):
                    p = group["params"][j]
                    if p.grad is not None:
                        per_expert_norm[i] += p.grad.norm()

            for idx, p in enumerate(group["params"]):
                if p.grad is not None:
                    if per_expert_norm[idx // per_expert_num] != 0:
                        p.grad /= per_expert_norm[idx // per_expert_num]

                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
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

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def zeropower_via_newtonschulz5(G, steps):
    """Moonlight-style Newton-Schulz orthogonalization for a 2D update matrix."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Adapted from Moonlight's toy_train.py for single-device use in this repo.

    The router parameter here is a Conv1d weight, so non-2D gradients are flattened
    to a matrix before the Muon update and then reshaped back.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
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
        for p in muon_params:
            if p.ndim < 2:
                raise ValueError(f"Muon expects at least 2D params, got shape {tuple(p.shape)}")
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @staticmethod
    def _matrix_shape(shape):
        return shape[:1] + (math.prod(shape[1:]),) if len(shape) > 2 else shape

    def adjust_lr_for_muon(self, lr, param_shape):
        a, b = self._matrix_shape(tuple(param_shape))
        adjusted_ratio = 0.2 * math.sqrt(max(a, b))
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
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            for p in params:
                g = p.grad
                if g is None:
                    continue

                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                if g.ndim != 2:
                    raise ValueError(f"Muon expects a matrix update, got ndim={g.ndim}")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                p.data.mul_(1 - lr * wd)
                p.data.add_(u.view_as(p.data), alpha=-adjusted_lr)

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)

                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / (bias_correction2**0.5)

                p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def test_expert_inner(model, cluster=0):
    expert_fea = []
    expert_cen = []
    with torch.no_grad():
        for i in range(model.expert_num):
            feature_inner = torch.max(
                torch.abs(torch.matmul(model.models[i].conv1.weight.squeeze(1), features[[cluster]].float().transpose(1, 0)))
            )
            center_inner = torch.max(
                torch.abs(torch.matmul(model.models[i].conv1.weight.squeeze(1), centers[[cluster]].float().transpose(1, 0)))
            )
            expert_fea.append(feature_inner)
            expert_cen.append(center_inner)
    return expert_fea, expert_cen


def test_router_inner(model):
    router_feature = []
    router_center = []
    with torch.no_grad():
        for cluster in range(CLUSTER_NUM):
            feature_inner = torch.abs(torch.matmul(model.router.conv1.weight.squeeze(1), features[[cluster]].float().transpose(1, 0)))
            center_inner = torch.abs(torch.matmul(model.router.conv1.weight.squeeze(1), centers[[cluster]].float().transpose(1, 0)))
            router_feature.append(feature_inner.cpu().tolist())
            router_center.append(center_inner.cpu().tolist())
    return router_feature, router_center


def test_each_expert(model, data, labels, datatype="training"):
    del data, labels, datatype
    expert_feature = [[] for _ in range(EXPERT_NUM)]
    expert_center = [[] for _ in range(EXPERT_NUM)]
    router_feature, router_center = test_router_inner(model)

    for i in range(CLUSTER_NUM):
        feat, cent = test_expert_inner(model, cluster=i)
        for each in range(EXPERT_NUM):
            expert_feature[each].append(feat[each].cpu())
            expert_center[each].append(cent[each].cpu())

    return expert_feature, expert_center, router_feature, router_center


def train(model, criterion, data, labels, optimizers, epochs, plot=False, load_balancing=False, verbose=True, early_stopping=True):
    expert_acc_train = [[[] for _ in range(CLUSTER_NUM)] for _ in range(EXPERT_NUM)]
    expert_inner_train = [[[] for _ in range(CLUSTER_NUM)] for _ in range(EXPERT_NUM)]

    router_acc_train = [[] for _ in range(CLUSTER_NUM)]
    router_inner_train = [[] for _ in range(CLUSTER_NUM)]

    entropy_record = []
    min_loss = float("inf")

    for epoch in range(epochs):
        for optimizer in optimizers:
            optimizer.zero_grad()
        outputs, select0, load_balancing_loss = model(data)

        e = entropy(
            torch.stack(
                [
                    select0[train_cluster_idx[0]].squeeze(-1).sum(dim=0),
                    select0[train_cluster_idx[1]].squeeze(-1).sum(dim=0),
                    select0[train_cluster_idx[2]].squeeze(-1).sum(dim=0),
                    select0[train_cluster_idx[3]].squeeze(-1).sum(dim=0),
                ]
            )
        )
        entropy_record.append(e)

        if load_balancing:
            loss = criterion(outputs, labels) + 0.0001 * load_balancing_loss
        else:
            loss = criterion(outputs, labels)

        if early_stopping:
            if loss.item() <= min_loss:
                min_loss = loss.item()
            elif loss > min_loss + 0.02 or loss <= 0.314:
                break

        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        if epoch % 100 == 0:
            if verbose:
                print(f"Epoch {epoch + 1} --- loss: {loss.item():.3f}")
            if plot:
                acc_list, inner_list, router_flist, router_clist = test_each_expert(model, training_data, training_labels, datatype="training")
                for each_cluster in range(CLUSTER_NUM):
                    router_acc_train[each_cluster].append(router_flist[each_cluster])
                    router_inner_train[each_cluster].append(router_clist[each_cluster])

                    for each_expert in range(EXPERT_NUM):
                        expert_acc_train[each_expert][each_cluster].append(acc_list[each_expert][each_cluster])
                        expert_inner_train[each_expert][each_cluster].append(inner_list[each_expert][each_cluster])

    print("Finished Training")
    return expert_acc_train, expert_inner_train, router_acc_train, router_inner_train, select0, entropy_record


def test(model, criterion, data, labels, verbose=True):
    correct = 0

    with torch.no_grad():
        outputs, _, _ = model(data)
        predicted = torch.max(outputs.data, 1).indices
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / data.shape[0]
    if verbose:
        print(f"Accuracy of the network on the {data.shape[0]} test images: {accuracy:.4f} %")

    return accuracy


def load_setting(setting, device):
    global training_data, training_labels, test_data, test_labels
    global centers, features, train_cluster_idx, test_cluster_idx

    setting_dir = ROOT / f"synthetic_data_s{setting}"
    training_data = torch.load(setting_dir / "train_data.pt", map_location=device)
    training_labels = torch.load(setting_dir / "train_labels.pt", map_location=device)
    test_data = torch.load(setting_dir / "test_data.pt", map_location=device)
    test_labels = torch.load(setting_dir / "test_labels.pt", map_location=device)
    centers = torch.load(setting_dir / "centers.pt", map_location=device)
    features = torch.load(setting_dir / "features.pt", map_location=device)

    with open(setting_dir / "train_cluster", "rb") as fp:
        train_cluster_idx = pickle.load(fp)

    with open(setting_dir / "test_cluster", "rb") as fp:
        test_cluster_idx = pickle.load(fp)


def build_optimizers(model, args):
    expert_muon_params = []
    expert_adamw_params = []
    for p in model.models.parameters():
        if p.ndim >= 2:
            expert_muon_params.append(p)
        else:
            expert_adamw_params.append(p)

    expert_optimizer = Muon(
        lr=args.expert_lr,
        wd=args.expert_weight_decay,
        muon_params=expert_muon_params,
        adamw_params=expert_adamw_params,
        momentum=args.expert_momentum,
        ns_steps=args.ns_steps,
    )
    router_optimizer = Muon(
        lr=args.router_lr,
        wd=args.router_weight_decay,
        muon_params=model.router.parameters(),
        momentum=args.router_momentum,
        ns_steps=args.ns_steps,
    )
    return [expert_optimizer, router_optimizer]


def run_trial(args, trial_idx, nonlinear=True):
    set_seed(args.seed + trial_idx)
    model = MoE(200, 8, PATCH_NUM, EXPERT_NUM, strategy="top1", nonlinear=nonlinear).to(training_data.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = build_optimizers(model, args)

    _, _, _, _, select, entropy_record = train(
        model,
        criterion,
        training_data,
        training_labels,
        optimizers,
        args.epochs,
        plot=args.plot,
        load_balancing=args.load_balancing,
        verbose=not args.quiet,
        early_stopping=not args.no_early_stopping,
    )
    acc = test(model, criterion, test_data, test_labels, verbose=not args.quiet)
    return model, acc, select, entropy_record


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    load_setting(args.setting, device)

    acc_list = []
    ent_list = []
    mode = "linear" if args.linear else "nonlinear"

    for trial_idx in range(args.trials):
        print(f"\nTrial {trial_idx + 1}/{args.trials} | setting=s{args.setting} | experts={mode}")
        _, acc, select, entropy_record = run_trial(args, trial_idx, nonlinear=not args.linear)
        acc_list.append(acc)
        ent_list.append(entropy_record[-1].item())

        print(select[train_cluster_idx[0]].squeeze(-1).sum(dim=0).to(torch.long))
        print(select[train_cluster_idx[1]].squeeze(-1).sum(dim=0).to(torch.long))
        print(select[train_cluster_idx[2]].squeeze(-1).sum(dim=0).to(torch.long))
        print(select[train_cluster_idx[3]].squeeze(-1).sum(dim=0).to(torch.long))

    if args.trials > 1:
        print()
        print(f"Average test accuracy: {round(np.mean(acc_list), 2)}.")
        print(f"Standard deviation: {round(np.std(acc_list), 2)}")
        print()
        print(f"Average dispatch entropy {round(np.mean(ent_list), 3)}.")
        print(f"Standard deviation: {round(np.std(ent_list), 3)}")


if __name__ == "__main__":
    main()
