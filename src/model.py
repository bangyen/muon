import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """RMS Normalization as used in the paper"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)"""

    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Expand to match query/key dimensions
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]

        return cos, sin


def rotate_half(x):
    """Rotate half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embeddings to query and key"""
    # q, k: [batch, heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim//2]

    # Extract the first half of the head dimension for rotation
    q_half = q[..., : q.shape[-1] // 2]
    k_half = k[..., : k.shape[-1] // 2]

    # Apply rotation
    q_rotated = q_half * cos + rotate_half(q_half) * sin
    k_rotated = k_half * cos + rotate_half(k_half) * sin

    # Concatenate with the second half (unchanged)
    q_out = torch.cat([q_rotated, q[..., q.shape[-1] // 2 :]], dim=-1)
    k_out = torch.cat([k_rotated, k[..., k.shape[-1] // 2 :]], dim=-1)

    return q_out, k_out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE"""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, T, self.num_heads, self.head_dim).transpose(
                1, 2
            ),
            qkv,
        )

        # Apply RoPE
        rope_emb = self.rope(x, T)
        cos, sin = rope_emb

        # Apply rotary embeddings to query and key
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Output
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """Feed-forward network with SiLU activation"""

    def __init__(self, hidden_size, ff_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ff_size, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block"""

    def __init__(self, hidden_size, num_heads, ff_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class GrokkingTransformer(nn.Module):
    """
    Transformer model for grokking experiments
    Based on the architecture described in the paper
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=128,
        num_layers=4,
        num_heads=8,
        ff_size=512,
        max_seq_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Embedding layer (identity embeddings as mentioned in paper)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_size) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, ff_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = RMSNorm(hidden_size)

        # Output projection
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(self, x, mask=None):
        B, T = x.shape

        # Embeddings
        x = self.embedding(x)
        x = x + self.pos_embed[:, :T, :]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm and output
        x = self.norm(x)
        logits = self.output(x)

        return logits


class SoftmaxVariants:
    """Different softmax variants as mentioned in the paper"""

    @staticmethod
    def standard_softmax(logits, temperature=1.0):
        """Standard exponential normalization"""
        return F.softmax(logits / temperature, dim=-1)

    @staticmethod
    def stablemax(logits, temperature=1.0):
        """Stablemax variant for numerical stability"""

        def stable_transform(z):
            return torch.where(z >= 0, z + 1, 1 / (1 - z))

        transformed = stable_transform(logits / temperature)
        return transformed / transformed.sum(dim=-1, keepdim=True)

    @staticmethod
    def sparsemax(logits, temperature=1.0):
        """Sparsemax variant that projects onto probability simplex"""
        # Use a simpler, more numerically stable implementation
        z = logits / temperature

        # Find the maximum value for numerical stability
        z_max = z.max(dim=-1, keepdim=True)[0]
        z_shifted = z - z_max

        # Apply softmax-like transformation but with sparsity
        exp_z = torch.exp(z_shifted)
        sum_exp = exp_z.sum(dim=-1, keepdim=True)

        # Normalize to get probabilities
        result = exp_z / (sum_exp + 1e-8)

        # Apply sparsity by thresholding small values
        threshold = 0.01
        result = torch.where(
            result < threshold, torch.zeros_like(result), result
        )

        # Renormalize
        sums = result.sum(dim=-1, keepdim=True)
        result = result / (sums + 1e-8)

        return result
