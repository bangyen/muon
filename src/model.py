import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    """Configuration for the GrokkingTransformer"""

    vocab_size: int
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    ff_size: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1
    softmax_variant: str = "standard"


class RMSNorm(nn.Module):
    """RMS Normalization as used in the paper"""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        Initialize RMS normalization layer

        Args:
            hidden_size: Size of the hidden dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMS normalization

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 512) -> None:
        """
        Initialize rotary positional embedding layer

        Args:
            dim: Dimension of the embedding
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for rotary positional embeddings

        Args:
            x: Input tensor
            seq_len: Sequence length (optional, defaults to x.shape[1])

        Returns:
            Tuple of (cos, sin) tensors for rotary embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]

        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Expand to match query/key dimensions
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input

    Args:
        x: Input tensor

    Returns:
        Tensor with rotated dimensions
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key

    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim]
        k: Key tensor of shape [batch, heads, seq_len, head_dim]
        cos: Cosine values for rotation
        sin: Sine values for rotation

    Returns:
        Tuple of (rotated_q, rotated_k) tensors
    """
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

    def __init__(
        self, hidden_size: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        """
        Initialize multi-head attention layer

        Args:
            hidden_size: Size of the hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim * num_heads != hidden_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, hidden_size = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2),
            qkv,
        )

        # Apply RoPE
        rope_emb = self.rope(x, seq_len)
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
        out = (
            (att @ v).transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        )
        return self.proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with SiLU activation"""

    def __init__(
        self, hidden_size: int, ff_size: int, dropout: float = 0.1
    ) -> None:
        """
        Initialize feed-forward network

        Args:
            hidden_size: Size of the hidden dimension
            ff_size: Size of the feed-forward layer
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ff_size, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network

        Args:
            x: Input tensor

        Returns:
            Output tensor after feed-forward transformation
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ff_size: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize transformer block

        Args:
            hidden_size: Size of the hidden dimension
            num_heads: Number of attention heads
            ff_size: Size of the feed-forward layer
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for transformer block

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Output tensor after transformer block processing
        """
        x = x + self.attention(self.norm1(x), mask)
        return x + self.feed_forward(self.norm2(x))


class GrokkingTransformer(nn.Module):
    """
    Transformer model for grokking experiments
    Based on the architecture described in the paper
    """

    def __init__(
        self,
        config: Optional[Union[ModelConfig, dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize GrokkingTransformer

        Args:
            config: Model configuration object or dict (optional)
            **kwargs: Configuration parameters
        """
        super().__init__()

        # Handle both config object and individual parameters
        if isinstance(config, ModelConfig):
            model_config = config
        elif isinstance(config, dict):
            model_config = ModelConfig(**config)
        else:
            # Extract parameters from kwargs
            model_config = ModelConfig(
                vocab_size=kwargs.get("vocab_size", 128),
                hidden_size=kwargs.get("hidden_size", 128),
                num_layers=kwargs.get("num_layers", 4),
                num_heads=kwargs.get("num_heads", 8),
                ff_size=kwargs.get("ff_size", 512),
                max_seq_len=kwargs.get("max_seq_len", 512),
                dropout=kwargs.get("dropout", 0.1),
                softmax_variant=kwargs.get("softmax_variant", "standard"),
            )

        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size
        self.softmax_variant = model_config.softmax_variant

        # Identity embeddings as mentioned in paper (integer value itself used as embedding index)
        # We'll implement this as a simple lookup table where token '42' maps to embedding vector 42
        self.embedding = nn.Embedding(
            model_config.vocab_size, model_config.hidden_size, padding_idx=0
        )

        # Initialize embeddings to identity-like behavior
        with torch.no_grad():
            # For numbers, set embedding to be close to the number itself
            special_tokens_count = 12  # Based on dataset.py special tokens
            for i in range(special_tokens_count, model_config.vocab_size):
                number_value = i - special_tokens_count
                if number_value < model_config.hidden_size:
                    # Use one-hot-like initialization for small numbers
                    self.embedding.weight[i, number_value] = 1.0
                else:
                    # For larger numbers, use scaled identity
                    scale = number_value / model_config.hidden_size
                    self.embedding.weight[i] = (
                        torch.randn(model_config.hidden_size) * 0.1 + scale
                    )

        # Positional encoding (RoPE is applied in attention, not here)
        # Remove the learned positional embeddings since we use RoPE

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_config.hidden_size,
                    model_config.num_heads,
                    model_config.ff_size,
                    model_config.dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )

        # Final layer norm
        self.norm = RMSNorm(model_config.hidden_size)

        # Output projection
        self.output = nn.Linear(
            model_config.hidden_size, model_config.vocab_size, bias=False
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for different module types

        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the GrokkingTransformer

        Args:
            x: Input tensor of shape [batch_size, seq_len] or [batch_size, seq_len, hidden_size]
            mask: Optional attention mask

        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # Handle different input shapes
        input_dim_3d = 3
        if x.dim() == input_dim_3d:
            batch_size, seq_len, _ = x.shape
            # If input is 3D, assume it's already embedded
            embedded = x
        else:
            batch_size, seq_len = x.shape
            # Embeddings
            embedded = self.embedding(x)
            # No positional embedding addition since we use RoPE in attention

        # Apply transformer blocks
        for block in self.blocks:
            embedded = block(embedded, mask)

        # Final norm and output
        embedded = self.norm(embedded)
        return self.output(embedded)


class SoftmaxVariants:
    """Different softmax variants as mentioned in the paper"""

    @staticmethod
    def standard_softmax(
        logits: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Standard exponential normalization

        Args:
            logits: Input logits
            temperature: Temperature parameter for softmax

        Returns:
            Softmax probabilities
        """
        return F.softmax(logits / temperature, dim=-1)

    @staticmethod
    def stablemax(
        logits: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Stablemax variant for numerical stability as described in the paper

        Formula: s(z_i) = {z_i + 1 if z_i >= 0, 1/(1-z_i) if z_i < 0}
        stablemax(z)_i = s(z_i) / sum_j s(z_j)

        Args:
            logits: Input logits
            temperature: Temperature parameter

        Returns:
            Stablemax probabilities
        """

        def stable_transform(z: torch.Tensor) -> torch.Tensor:
            """
            Apply stable transformation to prevent numerical instability

            Args:
                z: Input tensor

            Returns:
                Transformed tensor
            """
            return torch.where(z >= 0, z + 1, 1 / (1 - z))

        z = logits / temperature
        transformed = stable_transform(z)
        return transformed / transformed.sum(dim=-1, keepdim=True)

    @staticmethod
    def sparsemax(
        logits: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sparsemax variant that projects onto probability simplex as described in the paper

        Formula: sparsemax(z)_i = max{z_i - tau, 0}
        where tau is found such that sum_i max{z_i - tau, 0} = 1

        Args:
            logits: Input logits
            temperature: Temperature parameter

        Returns:
            Sparse probability distribution
        """
        z = logits / temperature
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)

        # Find the threshold tau using the efficient algorithm
        cumsum = torch.cumsum(z_sorted, dim=-1)
        range_vals = torch.arange(
            1, z_sorted.shape[-1] + 1, device=z.device, dtype=z.dtype
        )

        # Find k (number of active elements)
        threshold = z_sorted - (cumsum - 1) / range_vals
        valid = threshold > 0
        k = torch.sum(valid, dim=-1, keepdim=True)

        # Compute tau
        tau = (torch.sum(z_sorted * valid, dim=-1, keepdim=True) - 1) / k

        # Apply sparsemax transformation: max{z - tau, 0}
        return torch.clamp(z - tau, min=0)
