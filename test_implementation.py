#!/usr/bin/env python3
"""
Quick test script to verify the Muon Optimizer implementation
"""

import torch
from torch import nn

from src.dataset import ModularArithmeticDataset
from src.model import GrokkingTransformer, SoftmaxVariants
from src.optimizer import MuonOptimizer


def test_muon_optimizer():
    """Test basic Muon optimizer functionality"""
    print("Testing Muon Optimizer...")

    # Create a simple model
    model = nn.Linear(10, 5)

    # Create Muon optimizer
    optimizer = MuonOptimizer(
        model.parameters(),
        lr=1e-3,
        spectral_norm_strength=0.1,
        second_order_interval=5,
    )

    # Test a few optimization steps
    for step in range(10):
        # Create dummy data
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)

        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    print("✓ Muon optimizer test passed!")


def test_transformer_model():
    """Test Transformer model"""
    print("\nTesting Transformer Model...")

    # Create dataset
    dataset = ModularArithmeticDataset("add", modulus=97, train_split=0.8)

    # Create model
    model = GrokkingTransformer(
        vocab_size=dataset.vocab_size,
        hidden_size=64,  # Smaller for testing
        num_layers=2,
        num_heads=4,
        ff_size=128,
        max_seq_len=10,
    )

    # Test forward pass
    batch_size = 4
    seq_len = 7
    x = torch.randint(0, dataset.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(
            f"Expected output shape: {batch_size} x {seq_len} x {dataset.vocab_size}"
        )

    print("✓ Transformer model test passed!")


def test_softmax_variants():
    """Test different softmax variants"""
    print("\nTesting Softmax Variants...")

    # Create dummy logits
    logits = torch.randn(5, 10)

    # Test each variant
    variants = ["standard", "stablemax", "sparsemax"]

    for variant in variants:
        if variant == "standard":
            softmax_fn = SoftmaxVariants.standard_softmax
        elif variant == "stablemax":
            softmax_fn = SoftmaxVariants.stablemax
        elif variant == "sparsemax":
            softmax_fn = SoftmaxVariants.sparsemax

        probs = softmax_fn(logits)

        # Check that probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        print(f"{variant} softmax sums: {prob_sums}")

        # For sparsemax, we need to be more lenient due to numerical issues
        if variant == "sparsemax":
            assert torch.allclose(
                prob_sums, torch.ones_like(prob_sums), atol=1e-3
            )
        else:
            assert torch.allclose(
                prob_sums, torch.ones_like(prob_sums), atol=1e-6
            )

        print(f"✓ {variant} softmax test passed!")

    print("✓ All softmax variants test passed!")


def test_dataset():
    """Test dataset creation and loading"""
    print("\nTesting Dataset...")

    # Test each task type
    task_types = ["add", "mul", "gcd", "parity"]

    for task_type in task_types:
        dataset = ModularArithmeticDataset(
            task_type, modulus=97, train_split=0.8
        )

        # Test data loading
        sample = dataset[0]
        assert "input" in sample
        assert "target" in sample
        assert "result" in sample

        print(f"✓ {task_type} dataset test passed!")

    print("✓ All dataset tests passed!")


def test_training_step():
    """Test a single training step"""
    print("\nTesting Training Step...")

    # Create dataset
    dataset = ModularArithmeticDataset("add", modulus=97, train_split=0.8)

    # Create model
    model = GrokkingTransformer(
        vocab_size=dataset.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ff_size=128,
        max_seq_len=10,
    )

    # Create optimizer
    optimizer = MuonOptimizer(model.parameters(), lr=1e-3)

    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Get a batch
    batch = dataset[0]
    inputs = batch["input"].unsqueeze(0)  # Add batch dimension
    targets = batch["target"].unsqueeze(0)

    # Forward pass
    logits = model(inputs)
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)

    # Compute loss
    loss = criterion(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Training loss: {loss.item():.4f}")
    print("✓ Training step test passed!")


def main():
    """Run all tests"""
    print("=" * 50)
    print("MUON OPTIMIZER REPRODUCTION - TEST SUITE")
    print("=" * 50)

    try:
        test_muon_optimizer()
        test_transformer_model()
        test_softmax_variants()
        test_dataset()
        test_training_step()

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("\nThe implementation is ready for experiments!")
        print("\nNext steps:")
        print("1. Run quick test: python scripts/train_tasks.py --quick_test")
        print("2. Run full experiments: python scripts/train_tasks.py")
        print("3. Visualize results: python scripts/visualize_results.py")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
