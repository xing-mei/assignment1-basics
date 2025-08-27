import torch
from torch import nn
from einops import einsum, rearrange
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        w1x = self.w1(x)
        silu_w1x = w1x / (1 + torch.exp(-w1x))
        return self.w2(silu_w1x * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        assert d_k % 2 == 0
        super().__init__()
        # (d_k / 2)
        omega = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        # (max_seq_len)
        pos = torch.arange(max_seq_len, device=device)
        angles = torch.outer(pos, omega).float()
        rotations = torch.polar(torch.ones_like(angles, device = device), angles)
        self.register_buffer("rotations", rotations, persistent=False)
        self.d_k = d_k
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        assert x.shape[-1] == self.d_k
        assert x.shape[-2] == token_positions.shape[-1]
        in_dtype = x.dtype
        x_complex = torch.view_as_complex(rearrange(x.float(), "... (d1 d2) -> ... d1 d2", d2 = 2))
        rotations = self.rotations[token_positions]
        x_rotated = rearrange(torch.view_as_real(rotations * x_complex), "... d1 d2-> ... (d1 d2)", d2 = 2)
        return x_rotated.to(in_dtype)

def softmax(
    x: torch.Tensor,
    dim: int,        
) -> torch.Tensor:
    in_dtype = x.dtype
    max_x, _ = torch.max(x.float(), dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_x)
    return (exp_x / torch.sum(exp_x, dim=dim, keepdim=True)).to(in_dtype)

def scaled_dot_product_attention(
    Q: torch.Tensor, # " ... queries d_k"
    K: torch.Tensor, # " ... keys d_k"
    V: torch.Tensor, # " ... keys d_v"
    mask: torch.Tensor | None = None, # " ... queries keys"
) -> torch.Tensor:
    assert Q.shape[-1] == K.shape[-1]
    d_k = Q.shape[-1]
    q_kt = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    q_kt.masked_fill_(~mask, -torch.inf)
    return einsum(softmax(q_kt, dim = -1), V, "... queries keys, ... keys d_v -> ... queries d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        assert d_model % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.qkv_proj = Linear(d_model, 3 * d_model, device, dtype)
        self.o_proj = Linear(d_model, d_model, device, dtype)
    
    def forward(
        self,
        x: torch.Tensor, # " ... sequence_length d_model"
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        seq_len = x.shape[-2]
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)
        q = rearrange(q, "... seq_length (num_heads d_head) -> ... num_heads seq_length d_head", d_head = self.d_head)
        k = rearrange(k, "... seq_length (num_heads d_head) -> ... num_heads seq_length d_head", d_head = self.d_head)
        if rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)
        v = rearrange(v, "... seq_length (num_heads d_head) -> ... num_heads seq_length d_head", d_head = self.d_head)

        if causal:
            mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        else:
            mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "... num_heads seq_length d_head -> ... seq_length (num_heads d_head)")
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_head = int(d_model / num_heads)
        self.ln1 = RMSNorm(d_model = d_model, device = device, dtype = dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device, dtype)
        self.ln2 = RMSNorm(d_model = d_model, device = device, dtype = dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.rope)
        return x + self.ffn(self.ln2(x))

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = []
        for l in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype))
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(
        self,
        x: torch.Tensor, # "batch_size sequence_length"
    ) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
