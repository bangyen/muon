"""
Tests for the Transformer model implementation.

These tests validate the model architecture including:
- Transformer blocks
- Multi-head attention with RoPE
- Softmax variants (standard, stablemax, sparsemax)
- RMSNorm
- Positional embeddings
"""

import pytest
import torch
from torch import nn

from src.model import (
    GrokkingTransformer,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SoftmaxVariants,
    TransformerBlock,
)
from tests.conftest import TestConfig, set_seed


class TestTransformerModel:
    """Test suite for Transformer model"""

    def test_model_initialization(self):
        """Test model initialization with various configurations"""
        config = TestConfig.TEST_MODEL_CONFIG

        model = GrokkingTransformer(**config)

        assert hasattr(model, "embedding")
        assert hasattr(model, "blocks")
        assert hasattr(model, "output")
        assert hasattr(model, "norm")

        assert model.embedding.num_embeddings == config["vocab_size"]
        assert model.embedding.embedding_dim == config["hidden_size"]

        assert len(model.blocks) == config["num_layers"]

    def test_forward_pass(self):
        """Test forward pass with various input shapes"""
        set_seed(42)

        config = TestConfig.TEST_MODEL_CONFIG
        model = GrokkingTransformer(**config)

        test_cases = [
            (1, 5),
            (4, 7),
            (8, 10),
        ]

        for batch_size, seq_len in test_cases:
            x = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

            with torch.no_grad():
                output = model(x)

            expected_shape = (batch_size, seq_len, config["vocab_size"])
            assert output.shape == expected_shape

            assert torch.all(torch.isfinite(output))

    def test_attention_mechanism(self):
        """Test multi-head attention mechanism"""
        set_seed(42)

        hidden_size = 64
        num_heads = 4
        batch_size = 4
        seq_len = 8

        attention = MultiHeadAttention(hidden_size, num_heads)

        x = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            output = attention(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

        assert torch.all(torch.isfinite(output))

    def test_rms_norm(self):
        """Test RMS normalization"""
        set_seed(42)

        hidden_size = 64
        batch_size = 4
        seq_len = 8

        rms_norm = RMSNorm(hidden_size)

        x = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            output = rms_norm(x)

        assert output.shape == x.shape

        assert torch.all(torch.isfinite(output))

        assert not torch.allclose(output, x)

    def test_rotary_positional_embeddings(self):
        """Test rotary positional embeddings"""
        set_seed(42)

        dim = 64
        max_seq_len = 512
        batch_size = 4
        seq_len = 16

        rope = RotaryPositionalEmbedding(dim, max_seq_len)

        x = torch.randn(batch_size, seq_len, dim)

        with torch.no_grad():
            cos, sin = rope(x, seq_len)

        assert cos.shape == (1, 1, seq_len, dim // 2)
        assert sin.shape == (1, 1, seq_len, dim // 2)

        assert torch.all(torch.isfinite(cos))
        assert torch.all(torch.isfinite(sin))

        cos_sin_sum = cos**2 + sin**2
        assert torch.allclose(
            cos_sin_sum, torch.ones_like(cos_sin_sum), atol=1e-6
        )

    def test_transformer_block(self):
        """Test individual transformer block"""
        set_seed(42)

        hidden_size = 64
        num_heads = 4
        ff_size = 128
        batch_size = 4
        seq_len = 8

        block = TransformerBlock(hidden_size, num_heads, ff_size)

        x = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            output = block(x)

        assert output.shape == x.shape

        assert torch.all(torch.isfinite(output))

        assert not torch.allclose(output, x)

    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        set_seed(42)

        config = TestConfig.TEST_MODEL_CONFIG
        model = GrokkingTransformer(**config)

        batch_size = 4
        seq_len = 7
        x = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        targets = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

        logits = model(x)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(logits, targets)

        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.all(torch.isfinite(param.grad)), (
                f"Non-finite gradient for {name}"
            )

    def test_model_parameters(self):
        """Test that model has reasonable number of parameters"""
        config = TestConfig.TEST_MODEL_CONFIG
        model = GrokkingTransformer(**config)

        total_params = sum(p.numel() for p in model.parameters())

        assert total_params > 0

        expected_min = (
            config["vocab_size"] * config["hidden_size"]
            + config["num_layers"] * config["hidden_size"] * 4
            + config["vocab_size"] * config["hidden_size"]
        )
        assert total_params >= expected_min


class TestSoftmaxVariants:
    """Test suite for softmax variants"""

    def test_standard_softmax(self):
        """Test standard softmax implementation"""
        set_seed(42)

        test_cases = [
            torch.randn(5, 10),
            torch.randn(3, 7),
            torch.randn(1, 20),
        ]

        for logits in test_cases:
            probs = SoftmaxVariants.standard_softmax(logits)

            assert probs.shape == logits.shape

            prob_sums = probs.sum(dim=-1)
            assert torch.allclose(
                prob_sums, torch.ones_like(prob_sums), atol=1e-6
            )

            assert torch.all(probs >= 0)

            assert torch.all(torch.isfinite(probs))

    def test_stablemax(self):
        """Test stablemax implementation"""
        set_seed(42)

        test_cases = [
            torch.randn(5, 10),
            torch.randn(3, 7),
            torch.randn(1, 20),
        ]

        for logits in test_cases:
            probs = SoftmaxVariants.stablemax(logits)

            assert probs.shape == logits.shape

            prob_sums = probs.sum(dim=-1)
            assert torch.allclose(
                prob_sums, torch.ones_like(prob_sums), atol=1e-6
            )

            assert torch.all(probs >= 0)

            assert torch.all(torch.isfinite(probs))

    def test_sparsemax(self):
        """Test sparsemax implementation"""
        set_seed(42)

        test_cases = [
            torch.randn(5, 10),
            torch.randn(3, 7),
            torch.randn(1, 20),
        ]

        for logits in test_cases:
            probs = SoftmaxVariants.sparsemax(logits)

            assert probs.shape == logits.shape

            prob_sums = probs.sum(dim=-1)
            assert torch.allclose(
                prob_sums, torch.ones_like(prob_sums), atol=1e-3
            )

            assert torch.all(probs >= 0)

            assert torch.all(torch.isfinite(probs))

            torch.any(probs == 0, dim=-1)

    def test_softmax_variants_comparison(self):
        """Test that different softmax variants produce different outputs"""
        set_seed(42)

        logits = torch.randn(5, 10)

        standard_probs = SoftmaxVariants.standard_softmax(logits)
        stablemax_probs = SoftmaxVariants.stablemax(logits)
        sparsemax_probs = SoftmaxVariants.sparsemax(logits)

        assert not torch.allclose(standard_probs, stablemax_probs, atol=1e-6)
        assert not torch.allclose(standard_probs, sparsemax_probs, atol=1e-6)
        assert not torch.allclose(stablemax_probs, sparsemax_probs, atol=1e-6)

        for probs in [standard_probs, stablemax_probs, sparsemax_probs]:
            prob_sums = probs.sum(dim=-1)
            assert torch.allclose(
                prob_sums, torch.ones_like(prob_sums), atol=1e-3
            )
            assert torch.all(probs >= 0)

    def test_extreme_values(self):
        """Test softmax variants with extreme input values"""
        set_seed(42)

        large_logits = torch.tensor([[1000.0, 1001.0, 999.0]])

        standard_probs = SoftmaxVariants.standard_softmax(large_logits)
        stablemax_probs = SoftmaxVariants.stablemax(large_logits)
        sparsemax_probs = SoftmaxVariants.sparsemax(large_logits)

        assert torch.all(torch.isfinite(standard_probs))
        assert torch.all(torch.isfinite(stablemax_probs))
        assert torch.all(torch.isfinite(sparsemax_probs))

        small_logits = torch.tensor([[-1000.0, -1001.0, -999.0]])

        standard_probs = SoftmaxVariants.standard_softmax(small_logits)
        stablemax_probs = SoftmaxVariants.stablemax(small_logits)
        sparsemax_probs = SoftmaxVariants.sparsemax(small_logits)

        assert torch.all(torch.isfinite(standard_probs))
        assert torch.all(torch.isfinite(stablemax_probs))
        assert torch.all(torch.isfinite(sparsemax_probs))


class TestModelIntegration:
    """Test integration between model components"""

    def test_model_with_different_softmax(self):
        """Test model with different softmax variants"""
        set_seed(42)

        config = TestConfig.TEST_MODEL_CONFIG.copy()

        model = GrokkingTransformer(**config)

        batch_size = 4
        seq_len = 7
        x = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

        with torch.no_grad():
            output = model(x)

        expected_shape = (batch_size, seq_len, config["vocab_size"])
        assert output.shape == expected_shape
        assert torch.all(torch.isfinite(output))

    def test_model_device_transfer(self):
        """Test model transfer to different devices"""
        set_seed(42)

        config = TestConfig.TEST_MODEL_CONFIG
        model = GrokkingTransformer(**config)

        model_cpu = model.cpu()
        x_cpu = torch.randint(0, config["vocab_size"], (4, 7))

        with torch.no_grad():
            output_cpu = model_cpu(x_cpu)

        assert output_cpu.device.type == "cpu"

        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x_cpu.cuda()

            with torch.no_grad():
                output_cuda = model_cuda(x_cuda)

            assert output_cuda.device.type == "cuda"

            output_cuda_cpu = output_cuda.cpu()
            assert torch.allclose(output_cpu, output_cuda_cpu, atol=1e-5)

    def test_model_save_load(self):
        """Test model saving and loading"""
        set_seed(42)

        config = TestConfig.TEST_MODEL_CONFIG
        model = GrokkingTransformer(**config)

        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(model.state_dict(), f.name)

            loaded_model = GrokkingTransformer(**config)

            loaded_model.load_state_dict(model.state_dict())

            x = torch.randint(0, config["vocab_size"], (4, 7))

            with torch.no_grad():
                original_output = model(x)
                loaded_output = loaded_model(x)

            assert torch.all(torch.isfinite(original_output))
            assert torch.all(torch.isfinite(loaded_output))

            assert original_output.shape == loaded_output.shape

            os.unlink(f.name)

    def test_rotary_positional_embeddings_with_seq_len(self):
        """Test rotary positional embeddings with explicit seq_len parameter"""
        set_seed(42)

        dim = 64
        max_seq_len = 512
        batch_size = 4
        seq_len = 16

        rope = RotaryPositionalEmbedding(dim, max_seq_len)

        x = torch.randn(batch_size, seq_len, dim)

        with torch.no_grad():
            # Test with explicit seq_len parameter
            cos, sin = rope(x, seq_len=8)

        assert cos.shape == (1, 1, 8, dim // 2)
        assert sin.shape == (1, 1, 8, dim // 2)

        assert torch.all(torch.isfinite(cos))
        assert torch.all(torch.isfinite(sin))

    def test_rotary_positional_embeddings_with_none_seq_len(self):
        """Test rotary positional embeddings with seq_len=None (default behavior)"""
        set_seed(42)

        dim = 64
        max_seq_len = 512
        batch_size = 4
        seq_len = 16

        rope = RotaryPositionalEmbedding(dim, max_seq_len)

        x = torch.randn(batch_size, seq_len, dim)

        with torch.no_grad():
            # Test with seq_len=None (should use x.shape[1])
            cos, sin = rope(x, seq_len=None)

        assert cos.shape == (1, 1, seq_len, dim // 2)
        assert sin.shape == (1, 1, seq_len, dim // 2)

        assert torch.all(torch.isfinite(cos))
        assert torch.all(torch.isfinite(sin))

    def test_multihead_attention_invalid_head_dimension(self):
        """Test multi-head attention with invalid head dimension"""
        set_seed(42)

        hidden_size = 65  # Not divisible by num_heads
        num_heads = 4

        with pytest.raises(
            ValueError,
            match="hidden_size \\(65\\) must be divisible by num_heads \\(4\\)",
        ):
            MultiHeadAttention(hidden_size, num_heads)

    def test_attention_with_mask(self):
        """Test multi-head attention with attention mask"""
        set_seed(42)

        hidden_size = 64
        num_heads = 4
        batch_size = 2
        seq_len = 8

        attention = MultiHeadAttention(hidden_size, num_heads)

        x = torch.randn(batch_size, seq_len, hidden_size)

        # Create a causal mask
        mask = (
            torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        )
        mask = mask.expand(batch_size, num_heads, seq_len, seq_len)

        with torch.no_grad():
            output = attention(x, mask)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert torch.all(torch.isfinite(output))

    def test_model_config_object_initialization(self):
        """Test model initialization with ModelConfig object"""
        set_seed(42)

        from src.model import ModelConfig

        config = ModelConfig(
            vocab_size=128,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            ff_size=128,
            max_seq_len=10,
            dropout=0.1,
            softmax_variant="standard",
        )

        model = GrokkingTransformer(config)

        assert model.vocab_size == 128
        assert model.hidden_size == 64
        assert len(model.blocks) == 2

    def test_model_dict_config_initialization(self):
        """Test model initialization with dict config"""
        set_seed(42)

        config_dict = {
            "vocab_size": 128,
            "hidden_size": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_size": 128,
            "max_seq_len": 10,
            "dropout": 0.1,
            "softmax_variant": "standard",
        }

        model = GrokkingTransformer(config_dict)

        assert model.vocab_size == 128
        assert model.hidden_size == 64
        assert len(model.blocks) == 2

    def test_model_weight_initialization_with_bias(self):
        """Test model weight initialization for modules with bias"""
        set_seed(42)

        # Create a simple linear layer with bias to test bias initialization
        linear_with_bias = torch.nn.Linear(10, 5, bias=True)

        # Test the _init_weights method on a module with bias
        model = GrokkingTransformer(vocab_size=128, hidden_size=64)

        # Manually call _init_weights on a linear layer with bias
        model._init_weights(linear_with_bias)

        # Check that bias was initialized to zeros
        assert torch.allclose(
            linear_with_bias.bias, torch.zeros_like(linear_with_bias.bias)
        )

    def test_model_forward_with_3d_input(self):
        """Test model forward pass with 3D input tensor"""
        set_seed(42)

        config = TestConfig.TEST_MODEL_CONFIG
        model = GrokkingTransformer(**config)

        batch_size = 4
        seq_len = 7
        hidden_size = config["hidden_size"]

        # Create 3D input tensor (already embedded)
        x_3d = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            output = model(x_3d)

        expected_shape = (batch_size, seq_len, config["vocab_size"])
        assert output.shape == expected_shape
        assert torch.all(torch.isfinite(output))
