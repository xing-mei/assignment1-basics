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
        self.w = nn.Parameter(weight)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        embedding_table = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        self.embedding_table = nn.Parameter(embedding_table)
    
    def forward(
        self,
        token_ids: torch.Tensor      
    ) -> torch.Tensor:
        return self.embedding_table[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        gain = torch.empty(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gain)
        self.eps = eps
        self.d_model = d_model
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.square().mean(dim=-1, keepdim=True) + self.eps
        rms = rms.sqrt()
        output = x * self.gain / rms
        return output.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int,
        d_ff: int,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        w1x = self.w1(x)
        silu_w1x = w1x / (1 + torch.exp(-w1x))
        return self.w2(silu_w1x * self.w3(x))
