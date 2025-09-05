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

        task_types = ["add", "mul", "div", "exp", "gcd", "parity"]

        for task_type in task_types:
            dataset = ModularArithmeticDataset(task_type, **config)

            assert dataset.task_type == task_type
            assert dataset.modulus == config["modulus"]
            assert dataset.train_split == config["train_split"]
            assert dataset.max_seq_len == config["max_seq_len"]
            assert dataset.seed == config["seed"]

            assert len(dataset.data) > 0

            assert len(dataset.train_data) > 0
            assert len(dataset.val_data) > 0

    def test_data_generation(self):
        """Test data generation for different task types"""
        set_seed(42)

        modulus = 97
        test_cases = [
            ("add", 9409),
            ("mul", 9409),
            ("div", 9312),
            ("exp", 9409),
            ("gcd", 9409),
            ("parity", 1024),
        ]

        for task_type, expected_size in test_cases:
            if task_type == "parity":
                dataset = ModularArithmeticDataset(
                    task_type, modulus=2, train_split=0.5
                )
            else:
                dataset = ModularArithmeticDataset(
                    task_type, modulus=modulus, train_split=0.5
                )

            assert len(dataset.data) == expected_size

            for item in dataset.data[:10]:
                assert len(item) == 3
                assert all(isinstance(x, int) for x in item)
                if task_type == "parity":
                    assert 0 <= item[0] < 1024
                    assert 0 <= item[1] < 2
                    assert 0 <= item[2] < 2
                else:
                    assert all(0 <= x < modulus for x in item[:2])
                assert 0 <= item[2] < (2 if task_type == "parity" else modulus)

    def test_train_val_split(self):
        """Test train/validation split functionality"""
        set_seed(42)

        split_ratios = [0.1, 0.5, 0.8, 0.9]

        for split_ratio in split_ratios:
            dataset = ModularArithmeticDataset("add", train_split=split_ratio)

            total_size = len(dataset.data)
            train_size = len(dataset.train_data)
            val_size = len(dataset.val_data)

            assert train_size + val_size == total_size

            expected_train_size = int(total_size * split_ratio)
            assert abs(train_size - expected_train_size) <= 1

    def test_vocabulary_creation(self):
        """Test vocabulary creation and tokenization"""
        set_seed(42)

        dataset = ModularArithmeticDataset("add", modulus=97)

        assert dataset.vocab_size > 0

        assert "<pad>" in dataset.token_to_id
        assert "=" in dataset.token_to_id
        assert "mod" in dataset.token_to_id

        for i in range(min(10, dataset.modulus)):
            assert str(i) in dataset.token_to_id

    def test_data_loading(self):
        """Test data loading and sample format"""
        set_seed(42)

        dataset = ModularArithmeticDataset("add", modulus=97)

        sample = dataset[0]

        assert "input" in sample
        assert "target" in sample
        assert "result" in sample

        assert isinstance(sample["input"], torch.Tensor)
        assert isinstance(sample["target"], torch.Tensor)
        assert isinstance(sample["result"], int)

        assert sample["input"].dim() == 1
        assert sample["target"].dim() == 1

        assert sample["input"].shape == sample["target"].shape

    def test_data_loader(self):
        """Test DataLoader integration"""
        set_seed(42)

        dataset = ModularArithmeticDataset("add", modulus=97)

        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        batch = next(iter(dataloader))

        assert "input" in batch
        assert "target" in batch
        assert "result" in batch

        assert batch["input"].shape[0] == batch_size
        assert batch["target"].shape[0] == batch_size
        assert len(batch["result"]) == batch_size

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

            for _i, (a, b, result) in enumerate(dataset.data[:100]):
                expected = expected_func(a, b)
                assert (
                    result == expected
                ), f"Task {task_type}: {a} op {b} = {result}, expected {expected}"

    def test_parity_task(self):
        """Test parity task specifically"""
        set_seed(42)

        dataset = ModularArithmeticDataset("parity", modulus=2)

        assert len(dataset.data) == 1024

        for _i, (num, _, parity) in enumerate(dataset.data[:100]):
            binary_str = format(num, "010b")
            expected_parity = sum(int(bit) for bit in binary_str) % 2
            assert (
                parity == expected_parity
            ), f"Parity mismatch for {num} (binary: {binary_str})"

    def test_gcd_task(self):
        """Test GCD task specifically"""
        set_seed(42)

        dataset = ModularArithmeticDataset("gcd", modulus=97)

        for _i, (a, b, result) in enumerate(dataset.data[:100]):
            expected_gcd = dataset._gcd(a, b)
            assert (
                result == expected_gcd
            ), f"GCD mismatch: gcd({a}, {b}) = {result}, expected {expected_gcd}"

    def test_division_task(self):
        """Test division task specifically"""
        set_seed(42)

        dataset = ModularArithmeticDataset("div", modulus=97)

        for _i, (a, b, result) in enumerate(dataset.data[:100]):
            if b != 0:
                check_result = (result * b) % dataset.modulus
                assert (
                    check_result == a
                ), f"Division check failed: {result} * {b} ≡ {check_result} (mod {dataset.modulus}), expected {a}"

    def test_sequence_length_handling(self):
        """Test handling of different sequence lengths"""
        set_seed(42)

        for max_seq_len in [3, 5, 7]:
            dataset = ModularArithmeticDataset(
                "add", modulus=97, max_seq_len=max_seq_len
            )

            sample = dataset[0]
            assert sample["input"].shape[0] <= max_seq_len
            assert sample["target"].shape[0] <= max_seq_len

    def test_reproducibility(self):
        """Test that dataset generation is reproducible with same seed"""
        set_seed(42)

        dataset1 = ModularArithmeticDataset("add", seed=123)
        dataset2 = ModularArithmeticDataset("add", seed=123)

        assert len(dataset1.data) == len(dataset2.data)
        assert dataset1.data == dataset2.data

        assert dataset1.train_data == dataset2.train_data
        assert dataset1.val_data == dataset2.val_data

        assert dataset1.token_to_id == dataset2.token_to_id
        assert dataset1.idx_to_token == dataset2.idx_to_token

    def test_different_seeds(self):
        """Test that different seeds produce different data"""
        set_seed(42)

        dataset1 = ModularArithmeticDataset("add", seed=123)
        dataset2 = ModularArithmeticDataset("add", seed=456)

        assert dataset1.data != dataset2.data

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        set_seed(42)

        with pytest.raises(ValueError, match="Invalid task type"):
            ModularArithmeticDataset("invalid_task")

        with pytest.raises(ValueError, match="Modulus must be positive"):
            ModularArithmeticDataset("add", modulus=0)

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

        dataset = ModularArithmeticDataset("add", modulus=97)

        estimated_size = len(dataset.data) * 3 * 8
        assert estimated_size < 1e6

    def test_dataset_config_object(self):
        """Test dataset initialization with DatasetConfig object"""
        set_seed(42)

        from src.dataset import DatasetConfig

        config = DatasetConfig(
            task_type="add",
            modulus=97,
            train_split=0.5,
            max_seq_len=5,
            seed=42,
        )

        dataset = ModularArithmeticDataset(config)

        assert dataset.task_type == "add"
        assert dataset.modulus == 97
        assert dataset.train_split == 0.5
        assert dataset.max_seq_len == 5
        assert dataset.seed == 42

    def test_division_task_value_error(self):
        """Test division task with values that cause ValueError in modular inverse"""
        set_seed(42)

        # Create a dataset with a modulus that has non-coprime values
        dataset = ModularArithmeticDataset(
            "div", modulus=4
        )  # 4 has non-coprime values

        # The division task should skip values where modular inverse doesn't exist
        # This tests the ValueError handling in the division generation
        assert len(dataset.data) > 0

    def test_tokenize_parity_task(self):
        """Test tokenization for parity task specifically"""
        set_seed(42)

        dataset = ModularArithmeticDataset("parity", modulus=2)

        # Test the parity tokenization path
        tokens = dataset._tokenize(5, 0, 1)  # 5, 0, parity=1

        assert len(tokens) == 3
        assert tokens[0] == dataset.vocab["<bos>"]
        assert tokens[1] == 5 + len(dataset.special_tokens)  # token_index
        assert tokens[2] == dataset.vocab["<eos>"]

    def test_sequence_padding_and_truncation(self):
        """Test sequence padding and truncation in __getitem__"""
        set_seed(42)

        # Test padding case
        dataset_short = ModularArithmeticDataset(
            "add", modulus=97, max_seq_len=10
        )
        sample_short = dataset_short[0]

        assert sample_short["input"].shape[0] == 10
        assert sample_short["target"].shape[0] == 10

        # Test truncation case
        dataset_long = ModularArithmeticDataset(
            "add", modulus=97, max_seq_len=3
        )
        sample_long = dataset_long[0]

        assert sample_long["input"].shape[0] == 3
        assert sample_long["target"].shape[0] == 3

    def test_get_val_data_method(self):
        """Test the get_val_data method"""
        set_seed(42)

        dataset = ModularArithmeticDataset("add", modulus=97)

        val_data = dataset.get_val_data()

        assert len(val_data) == len(dataset.val_data)

        for sample in val_data[:5]:  # Test first 5 samples
            assert "input" in sample
            assert "target" in sample
            assert "result" in sample
            assert isinstance(sample["input"], torch.Tensor)
            assert isinstance(sample["target"], torch.Tensor)
            assert isinstance(sample["result"], int)

    def test_get_val_data_method_truncation(self):
        """Test the get_val_data method with sequence truncation"""
        set_seed(42)

        # Use a very short max_seq_len to trigger truncation
        dataset = ModularArithmeticDataset("add", modulus=97, max_seq_len=2)

        val_data = dataset.get_val_data()

        assert len(val_data) == len(dataset.val_data)

        for sample in val_data[:5]:  # Test first 5 samples
            assert "input" in sample
            assert "target" in sample
            assert "result" in sample
            assert isinstance(sample["input"], torch.Tensor)
            assert isinstance(sample["target"], torch.Tensor)
            assert isinstance(sample["result"], int)
            # Check that sequences are truncated to max_seq_len
            assert sample["input"].shape[0] == 2
            assert sample["target"].shape[0] == 2

    def test_get_val_data_method_padding(self):
        """Test the get_val_data method with sequence padding"""
        set_seed(42)

        # Use a longer max_seq_len to trigger padding
        dataset = ModularArithmeticDataset("add", modulus=97, max_seq_len=20)

        val_data = dataset.get_val_data()

        assert len(val_data) == len(dataset.val_data)

        for sample in val_data[:5]:  # Test first 5 samples
            assert "input" in sample
            assert "target" in sample
            assert "result" in sample
            assert isinstance(sample["input"], torch.Tensor)
            assert isinstance(sample["target"], torch.Tensor)
            assert isinstance(sample["result"], int)
            # Check that sequences are padded to max_seq_len
            assert sample["input"].shape[0] == 20
            assert sample["target"].shape[0] == 20

    def test_create_dataloader_function(self):
        """Test the create_dataloader function"""
        set_seed(42)

        from src.dataset import create_dataloader

        dataset = ModularArithmeticDataset("add", modulus=97)
        dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(dataloader))
        assert batch["input"].shape[0] == 4

    def test_get_task_configs_function(self):
        """Test the get_task_configs function"""
        from src.dataset import get_task_configs

        configs = get_task_configs()

        expected_tasks = ["gcd", "add", "div", "exp", "mul", "parity"]
        assert all(task in configs for task in expected_tasks)

        for task, config in configs.items():
            assert "modulus" in config
            assert "train_split" in config
            assert isinstance(config["modulus"], int)
            assert isinstance(config["train_split"], float)


class TestDatasetIntegration:
    """Test integration between dataset and other components"""

    def test_dataset_with_model(self):
        """Test dataset integration with model"""
        set_seed(42)

        from src.model import GrokkingTransformer

        dataset = ModularArithmeticDataset("add", modulus=97)
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            ff_size=128,
            max_seq_len=10,
        )

        sample = dataset[0]
        inputs = sample["input"].unsqueeze(0)

        with torch.no_grad():
            output = model(inputs)

        expected_shape = (1, inputs.shape[1], dataset.vocab_size)
        assert output.shape == expected_shape

    def test_dataset_with_optimizer(self):
        """Test dataset integration with optimizer"""
        set_seed(42)

        from muon import SingleDeviceMuonWithAuxAdam

        from src.model import GrokkingTransformer

        dataset = ModularArithmeticDataset("add", modulus=97)
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            ff_size=128,
            max_seq_len=10,
        )

        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]

        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=1e-3,
                weight_decay=1e-2,
            ),
            dict(
                params=other_params,
                use_muon=False,
                lr=1e-4,
                betas=(0.9, 0.95),
                weight_decay=1e-2,
            ),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        sample = dataset[0]
        inputs = sample["input"].unsqueeze(0)
        targets = sample["target"].unsqueeze(0)

        logits = model(inputs)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert torch.isfinite(loss)
