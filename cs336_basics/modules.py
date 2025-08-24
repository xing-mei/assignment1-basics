import torch
from torch import nn
from einops import einsum
import math

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert in_features > 0 and out_features > 0
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2. / (in_features + out_features))
        nn.init.trunc_normal_(weight, 0, std, -3. * std, 3. * std)
        self.W = nn.Parameter(weight)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
