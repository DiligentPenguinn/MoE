"""Custom optimizers and optimizer-family construction for the MoE model."""

import math
from typing import Any, Dict, List, Tuple

import torch
from torch.optim import SGD, _functional
from torch.optim.optimizer import Optimizer, required


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
                start = expert_idx * per_expert_num
                end = (expert_idx + 1) * per_expert_num
                for param in params[start:end]:
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
                momentum_buffer_list.append(state.get("momentum_buffer"))

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
                self.state[param]["momentum_buffer"] = momentum_buffer

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
        return lr * (0.2 * math.sqrt(max(rows, cols)))

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
                grad = grad.add(buf, alpha=momentum) if group["nesterov"] else buf

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


def split_muon_params(parameters) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    muon_params = []
    adamw_params = []
    for param in parameters:
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_optimizers(model, config: Dict[str, Any]) -> List[Optimizer]:
    family = config["optimizer_family"]
    if family == "original":
        return [
            NormalizedGD(
                model.models.parameters(),
                expert_num=model.expert_num,
                lr=config["expert_lr"],
                momentum=config["expert_momentum"],
                weight_decay=config["expert_weight_decay"],
            ),
            SGD(
                model.router.parameters(),
                lr=config["router_lr"],
                momentum=config["router_momentum"],
                weight_decay=config["router_weight_decay"],
            ),
        ]

    if family == "muon":
        expert_muon_params, expert_adamw_params = split_muon_params(model.models.parameters())
        router_muon_params, router_adamw_params = split_muon_params(model.router.parameters())
        return [
            Muon(
                lr=config["expert_lr"],
                wd=config["expert_weight_decay"],
                muon_params=expert_muon_params,
                adamw_params=expert_adamw_params,
                momentum=config["expert_momentum"],
                ns_steps=config["ns_steps"],
            ),
            Muon(
                lr=config["router_lr"],
                wd=config["router_weight_decay"],
                muon_params=router_muon_params,
                adamw_params=router_adamw_params,
                momentum=config["router_momentum"],
                ns_steps=config["ns_steps"],
            ),
        ]

    raise ValueError(f"Unsupported optimizer family: {family}")
