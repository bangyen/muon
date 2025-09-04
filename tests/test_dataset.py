"""
Tests for the dataset implementation.

These tests validate the dataset functionality including:
- Modular arithmetic tasks (add, mul, div, exp, gcd)
- Parity task
- Data generation and splitting
- Vocabulary creation
- Data loading
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import ModularArithmeticDataset
from tests.conftest import TestConfig, set_seed


class TestModularArithmeticDataset:
    """Test suite for ModularArithmeticDataset"""

    def test_initialization(self):
        """Test dataset initialization with various configurations"""
        config = TestConfig.TEST_DATASET_CONFIG

        # Test each task type
        task_types = ["add", "mul", "div", "exp", "gcd", "parity"]

        for task_type in task_types:
            dataset = ModularArithmeticDataset(task_type, **config)

            # Check basic attributes
            assert dataset.task_type == task_type
            assert dataset.modulus == config["modulus"]
            assert dataset.train_split == config["train_split"]
            assert dataset.max_seq_len == config["max_seq_len"]
            assert dataset.seed == config["seed"]

            # Check that data exists
            assert len(dataset.data) > 0

            # Check that train/val split exists
            assert len(dataset.train_data) > 0
            assert len(dataset.val_data) > 0

    def test_data_generation(self):
        """Test data generation for different task types"""
        set_seed(42)

        modulus = 97
        test_cases = [
            ("add", 9409),  # 97 * 97 = 9409
            ("mul", 9409),
            ("div", 9409),
            ("exp", 9409),
            ("gcd", 9409),
            ("parity", 1024),  # 2^10 = 1024
        ]

        for task_type, expected_size in test_cases:
            dataset = ModularArithmeticDataset(
                task_type, modulus=modulus, train_split=0.5
            )

            # Check data size
            assert len(dataset.data) == expected_size

            # Check data format
            for item in dataset.data[:10]:  # Check first 10 items
                assert len(item) == 3  # (a, b, result)
                assert all(isinstance(x, int) for x in item)
                assert all(0 <= x < modulus for x in item[:2])  # inputs
                assert 0 <= item[2] < modulus  # output

    def test_train_val_split(self):
        """Test train/validation split functionality"""
        set_seed(42)

        # Test different split ratios
        split_ratios = [0.1, 0.5, 0.8, 0.9]

        for split_ratio in split_ratios:
            dataset = ModularArithmeticDataset("add", train_split=split_ratio)

            total_size = len(dataset.data)
            train_size = len(dataset.train_data)
            val_size = len(dataset.val_data)

            # Check that split adds up correctly
            assert train_size + val_size == total_size

            # Check that split ratio is approximately correct
            expected_train_size = int(total_size * split_ratio)
            assert abs(train_size - expected_train_size) <= 1

    def test_vocabulary_creation(self):
        """Test vocabulary creation and tokenization"""
        set_seed(42)

        dataset = ModularArithmeticDataset("add", modulus=97)

        # Check vocabulary size
        assert dataset.vocab_size > 0

        # Check that vocabulary contains necessary tokens
        assert 0 in dataset.token_to_id  # padding token
        assert "=" in dataset.token_to_id  # equals token
        assert "mod" in dataset.token_to_id  # modulo token

        # Check that numbers are in vocabulary
        for i in range(min(10, dataset.modulus)):
            assert str(i) in dataset.token_to_id

    def test_data_loading(self):
        """Test data loading and sample format"""
        set_seed(42)

        dataset = ModularArithmeticDataset("add", modulus=97)

        # Test getting a sample
        sample = dataset[0]

        # Check sample format
        assert "input" in sample
        assert "target" in sample
        assert "result" in sample

        # Check data types
        assert isinstance(sample["input"], torch.Tensor)
        assert isinstance(sample["target"], torch.Tensor)
        assert isinstance(sample["result"], int)

        # Check tensor shapes
        assert sample["input"].dim() == 1  # 1D sequence
        assert sample["target"].dim() == 1  # 1D sequence

        # Check that input and target have same length
        assert sample["input"].shape == sample["target"].shape

    def test_data_loader(self):
        """Test DataLoader integration"""
        set_seed(42)

        dataset = ModularArithmeticDataset("add", modulus=97)

        # Create DataLoader
        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Get a batch
        batch = next(iter(dataloader))

        # Check batch format
        assert "input" in batch
        assert "target" in batch
        assert "result" in batch

        # Check batch shapes
        assert batch["input"].shape[0] == batch_size
        assert batch["target"].shape[0] == batch_size
        assert len(batch["result"]) == batch_size

        # Check that all samples in batch have same sequence length
        assert batch["input"].shape[1] == batch["target"].shape[1]

    def test_modular_arithmetic_correctness(self):
        """Test that modular arithmetic operations are correct"""
        set_seed(42)

        modulus = 97
        test_cases = [
            ("add", lambda a, b: (a + b) % modulus),
            ("mul", lambda a, b: (a * b) % modulus),
            ("exp", lambda a, b: pow(a, b, modulus)),
        ]

        for task_type, expected_func in test_cases:
            dataset = ModularArithmeticDataset(task_type, modulus=modulus)

            # Test first 100 samples
            for i, (a, b, result) in enumerate(dataset.data[:100]):
                expected = expected_func(a, b)
                assert (
                    result == expected
                ), f"Task {task_type}: {a} op {b} = {result}, expected {expected}"

    def test_parity_task(self):
        """Test parity task specifically"""
        set_seed(42)

        dataset = ModularArithmeticDataset("parity", modulus=2)

        # Check data size
        assert len(dataset.data) == 1024  # 2^10

        # Test that parity is calculated correctly
        for i, (binary_str, _, parity) in enumerate(dataset.data[:100]):
            # Convert binary string to integer
            num = int(binary_str, 2)
            expected_parity = bin(num).count("1") % 2
            assert (
                parity == expected_parity
            ), f"Parity mismatch for {binary_str}"

    def test_gcd_task(self):
        """Test GCD task specifically"""
        set_seed(42)

        dataset = ModularArithmeticDataset("gcd", modulus=97)

        # Test that GCD is calculated correctly
        for i, (a, b, result) in enumerate(dataset.data[:100]):
            expected_gcd = dataset._gcd(a, b)
            assert (
                result == expected_gcd
            ), f"GCD mismatch: gcd({a}, {b}) = {result}, expected {expected_gcd}"

    def test_division_task(self):
        """Test division task specifically"""
        set_seed(42)

        dataset = ModularArithmeticDataset("div", modulus=97)

        # Test that division is calculated correctly
        for i, (a, b, result) in enumerate(dataset.data[:100]):
            if b != 0:
                # Check that result * b ≡ a (mod modulus)
                check_result = (result * b) % dataset.modulus
                assert (
                    check_result == a
                ), f"Division check failed: {result} * {b} ≡ {check_result} (mod {dataset.modulus}), expected {a}"

    def test_sequence_length_handling(self):
        """Test handling of different sequence lengths"""
        set_seed(42)

        # Test with different max_seq_len values
        for max_seq_len in [3, 5, 7]:
            dataset = ModularArithmeticDataset(
                "add", modulus=97, max_seq_len=max_seq_len
            )

            # Check that sequences don't exceed max length
            sample = dataset[0]
            assert sample["input"].shape[0] <= max_seq_len
            assert sample["target"].shape[0] <= max_seq_len

    def test_reproducibility(self):
        """Test that dataset generation is reproducible with same seed"""
        set_seed(42)

        # Create two datasets with same seed
        dataset1 = ModularArithmeticDataset("add", seed=123)
        dataset2 = ModularArithmeticDataset("add", seed=123)

        # Check that data is identical
        assert len(dataset1.data) == len(dataset2.data)
        assert dataset1.data == dataset2.data

        # Check that train/val splits are identical
        assert dataset1.train_data == dataset2.train_data
        assert dataset1.val_data == dataset2.val_data

        # Check that vocabulary is identical
        assert dataset1.token_to_id == dataset2.token_to_id
        assert dataset1.id_to_token == dataset2.id_to_token

    def test_different_seeds(self):
        """Test that different seeds produce different data"""
        set_seed(42)

        # Create datasets with different seeds
        dataset1 = ModularArithmeticDataset("add", seed=123)
        dataset2 = ModularArithmeticDataset("add", seed=456)

        # Check that data is different
        assert dataset1.data != dataset2.data

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        set_seed(42)

        # Test invalid task type
        with pytest.raises(ValueError, match="Invalid task type"):
            ModularArithmeticDataset("invalid_task")

        # Test invalid modulus
        with pytest.raises(ValueError, match="Modulus must be positive"):
            ModularArithmeticDataset("add", modulus=0)

        # Test invalid train split
        with pytest.raises(
            ValueError, match="Train split must be between 0 and 1"
        ):
            ModularArithmeticDataset("add", train_split=1.5)

        with pytest.raises(
            ValueError, match="Train split must be between 0 and 1"
        ):
            ModularArithmeticDataset("add", train_split=-0.1)

    def test_memory_efficiency(self):
        """Test that dataset doesn't use excessive memory"""
        set_seed(42)

        # Create a large dataset
        dataset = ModularArithmeticDataset("add", modulus=97)

        # Check that dataset size is reasonable
        # This is a rough estimate - actual memory usage depends on implementation
        estimated_size = (
            len(dataset.data) * 3 * 8
        )  # 3 ints per sample, 8 bytes per int
        assert estimated_size < 1e6  # Less than 1MB for basic data


class TestDatasetIntegration:
    """Test integration between dataset and other components"""

    def test_dataset_with_model(self):
        """Test dataset integration with model"""
        set_seed(42)

        from src.model import GrokkingTransformer

        # Create dataset and model
        dataset = ModularArithmeticDataset("add", modulus=97)
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            ff_size=128,
            max_seq_len=10,
        )

        # Test forward pass with dataset sample
        sample = dataset[0]
        inputs = sample["input"].unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(inputs)

        # Check output shape
        expected_shape = (1, inputs.shape[1], dataset.vocab_size)
        assert output.shape == expected_shape

    def test_dataset_with_optimizer(self):
        """Test dataset integration with optimizer"""
        set_seed(42)

        from src.model import GrokkingTransformer
        from src.optimizer import MuonOptimizer

        # Create dataset, model, and optimizer
        dataset = ModularArithmeticDataset("add", modulus=97)
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            ff_size=128,
            max_seq_len=10,
        )
        optimizer = MuonOptimizer(model.parameters())

        # Test training step
        sample = dataset[0]
        inputs = sample["input"].unsqueeze(0)
        targets = sample["target"].unsqueeze(0)

        # Forward pass
        logits = model(inputs)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        # Compute loss
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that training step completed successfully
        assert loss.item() > 0
        assert torch.isfinite(loss.item())
