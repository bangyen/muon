"""
Test configuration and utilities for the Muon optimizer test suite.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch


class TestConfig:
    """Configuration for running tests"""

    # Model parameters for testing
    TEST_MODEL_CONFIG = {
        "vocab_size": 100,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_size": 128,
        "max_seq_len": 10,
        "dropout": 0.1,
    }

    # Optimizer parameters for testing
    TEST_OPTIMIZER_CONFIG = {
        "lr": 1e-3,
        "betas": (0.9, 0.98),
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "spectral_norm_strength": 0.1,
        "second_order_interval": 5,
        "use_orthogonal_updates": True,
    }

    # Dataset parameters for testing
    TEST_DATASET_CONFIG = {
        "modulus": 97,
        "train_split": 0.8,
        "max_seq_len": 5,
        "seed": 42,
    }

    # Training parameters for testing
    TEST_TRAINING_CONFIG = {
        "batch_size": 32,
        "num_epochs": 10,
        "eval_interval": 5,
        "save_interval": 10,
    }


@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    return TestConfig()


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def device():
    """Fixture providing the device to run tests on"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Fixture providing a fixed random seed for reproducibility"""
    return 42


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_test_model(config: dict[str, Any], device: torch.device):
    """Create a test model with given configuration"""
    from src.model import GrokkingTransformer

    model = GrokkingTransformer(**config)
    model = model.to(device)
    return model


def create_test_optimizer(model_params, config: dict[str, Any]):
    """Create a test optimizer with given configuration"""
    from src.optimizer import MuonOptimizer

    optimizer = MuonOptimizer(model_params, **config)
    return optimizer


def create_test_dataset(task_type: str, config: dict[str, Any]):
    """Create a test dataset with given configuration"""
    from src.dataset import ModularArithmeticDataset

    dataset = ModularArithmeticDataset(task_type, **config)
    return dataset


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from logits and targets"""
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def detect_grokking(
    train_acc: list, val_acc: list, threshold: float = 0.95
) -> int:
    """
    Detect grokking epoch based on the paper's definition:
    First epoch where validation accuracy reaches threshold after training accuracy stabilizes
    """
    if len(val_acc) < 2:
        return -1

    # Find first epoch where validation accuracy reaches threshold
    for i, acc in enumerate(val_acc):
        if acc >= threshold:
            # Check if training accuracy has stabilized (close to 100%)
            # Make sure we have enough training data and it's stabilized
            if i < len(train_acc) and train_acc[i] >= 0.95:
                return i

    return -1  # No grokking detected
