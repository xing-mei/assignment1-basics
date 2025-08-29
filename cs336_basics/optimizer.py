from typing import Optional
from collections.abc import Callable, Iterator
import torch
from torch import nn
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params: Iterator[nn.Parameter], 
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        assert lr >= 0.
        assert weight_decay >= 0.
        assert betas[0] >= 0. and betas[0] < 1.0
        assert betas[1] >= 0. and betas[1] < 1.0
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }        
        super().__init__(params, defaults=defaults)
    
    @torch.no_grad()
    def step(self, 
            closure: Optional[Callable] = None
        ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)

                grad = p.grad.data
                m = betas[0] * m + (1. - betas[0]) * grad
                v = betas[1] * v + (1. - betas[1]) * grad * grad
                lr_t = lr * math.sqrt(1. - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
