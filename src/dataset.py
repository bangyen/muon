import random
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetConfig:
    """Configuration for the ModularArithmeticDataset"""

    task_type: str
    modulus: int = 97
    train_split: float = 0.5
    max_seq_len: int = 5
    seed: int = 42


class ModularArithmeticDataset(Dataset[dict[str, Union[torch.Tensor, int]]]):
    """
    Dataset for modular arithmetic tasks as described in the paper
    Supports: addition, multiplication, division, exponentiation, GCD, and parity
    """

    def __init__(
        self, task_type: Union[str, DatasetConfig], **kwargs: Any
    ) -> None:
        """
        Initialize modular arithmetic dataset

        Args:
            task_type: Task type or DatasetConfig object
            **kwargs: Additional configuration parameters
        """
        if isinstance(task_type, DatasetConfig):
            config = task_type
        else:
            config = DatasetConfig(
                task_type=task_type,
                modulus=kwargs.get("modulus", 97),
                train_split=kwargs.get("train_split", 0.5),
                max_seq_len=kwargs.get("max_seq_len", 5),
                seed=kwargs.get("seed", 42),
            )

        if config.task_type not in [
            "add",
            "mul",
            "div",
            "exp",
            "gcd",
            "parity",
        ]:
            raise ValueError(f"Invalid task type: {config.task_type}")

        if config.modulus <= 0:
            raise ValueError(f"Modulus must be positive: {config.modulus}")

        min_train_split = 0.0
        max_train_split = 1.0
        if not min_train_split < config.train_split < max_train_split:
            raise ValueError(
                f"Train split must be between 0 and 1: {config.train_split}"
            )

        self.task_type = config.task_type
        self.modulus = config.modulus
        self.train_split = config.train_split
        self.max_seq_len = config.max_seq_len
        self.seed = config.seed

        random.seed(config.seed)
        np.random.seed(config.seed)

        if config.task_type == "parity":
            self.data = self._generate_parity_data()
        else:
            self.data = self._generate_modular_data()

        self._split_data()
        self._create_vocabulary()

    def _generate_modular_data(self) -> list[tuple[int, int, int]]:
        """Generate data for modular arithmetic tasks"""
        data = []

        if self.task_type == "add":
            for a in range(self.modulus):
                for b in range(self.modulus):
                    result = (a + b) % self.modulus
                    data.append((a, b, result))

        elif self.task_type == "mul":
            for a in range(self.modulus):
                for b in range(self.modulus):
                    result = (a * b) % self.modulus
                    data.append((a, b, result))

        elif self.task_type == "div":
            for a in range(self.modulus):
                for b in range(1, self.modulus):
                    try:
                        inv_b = pow(b, -1, self.modulus)
                        result = (a * inv_b) % self.modulus
                        data.append((a, b, result))
                    except ValueError:
                        continue

        elif self.task_type == "exp":
            for a in range(self.modulus):
                for b in range(self.modulus):
                    result = pow(a, b, self.modulus)
                    data.append((a, b, result))

        elif self.task_type == "gcd":
            for a in range(self.modulus):
                for b in range(self.modulus):
                    result = self._gcd(a, b) % self.modulus
                    data.append((a, b, result))

        return data

    def _generate_parity_data(self) -> list[tuple[int, int, int]]:
        """Generate data for parity task (10-bit binary strings)"""
        data = []
        for i in range(1024):
            binary_str = format(i, "010b")
            parity = sum(int(bit) for bit in binary_str) % 2
            data.append((i, 0, parity))
        return data

    def _gcd(self, a: int, b: int) -> int:
        """Compute greatest common divisor"""
        while b:
            a, b = b, a % b
        return a

    def _split_data(self) -> None:
        """Split data into train and validation sets"""
        random.shuffle(self.data)

        split_idx = int(len(self.data) * self.train_split)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

    def _create_vocabulary(self) -> None:
        """Create vocabulary mapping for tokenization"""
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<eos>": 2,
            "<bos>": 3,
            "=": 4,
            "+": 5,
            "-": 6,
            "*": 7,
            "/": 8,
            "^": 9,
            "gcd": 10,
            "mod": 11,
            "parity": 12,
        }

        self.number_tokens = {
            str(i): i + len(self.special_tokens) for i in range(self.modulus)
        }

        self.vocab = {**self.special_tokens, **self.number_tokens}
        self.vocab_size = len(self.vocab)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.token_to_id = self.vocab

    def _tokenize(self, a: int, b: int, result: int) -> list[int]:
        """Convert arithmetic expression to token sequence"""
        if self.task_type == "parity":
            token_index = a + len(self.special_tokens)
            return [
                self.vocab["<bos>"],
                token_index,
                self.vocab["<eos>"],
            ]
        op_token = {
            "add": self.vocab["+"],
            "sub": self.vocab["-"],
            "mul": self.vocab["*"],
            "div": self.vocab["/"],
            "exp": self.vocab["^"],
            "gcd": self.vocab["gcd"],
        }[self.task_type]

        return [
            self.vocab["<bos>"],
            a + len(self.special_tokens),
            op_token,
            b + len(self.special_tokens),
            self.vocab["="],
            result + len(self.special_tokens),
            self.vocab["<eos>"],
        ]

    def __len__(self) -> int:
        """Return the number of training samples"""
        return len(self.train_data)

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, int]]:
        """
        Get a single training sample

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing input tokens, target tokens, and result
        """
        a, b, result = self.train_data[idx]

        input_tokens = self._tokenize(a, b, result)
        target_tokens = input_tokens[1:] + [self.vocab["<pad>"]]

        if len(input_tokens) < self.max_seq_len:
            input_tokens += [self.vocab["<pad>"]] * (
                self.max_seq_len - len(input_tokens)
            )
            target_tokens += [self.vocab["<pad>"]] * (
                self.max_seq_len - len(target_tokens)
            )
        else:
            input_tokens = input_tokens[: self.max_seq_len]
            target_tokens = target_tokens[: self.max_seq_len]

        return {
            "input": torch.tensor(input_tokens, dtype=torch.long),
            "target": torch.tensor(target_tokens, dtype=torch.long),
            "result": result,
        }

    def get_val_data(self) -> list[dict[str, Union[torch.Tensor, int]]]:
        """Get validation data"""
        val_samples: list[dict[str, Union[torch.Tensor, int]]] = []
        for a, b, result in self.val_data:
            input_tokens = self._tokenize(a, b, result)
            target_tokens = input_tokens[1:] + [self.vocab["<pad>"]]

            if len(input_tokens) < self.max_seq_len:
                input_tokens += [self.vocab["<pad>"]] * (
                    self.max_seq_len - len(input_tokens)
                )
                target_tokens += [self.vocab["<pad>"]] * (
                    self.max_seq_len - len(target_tokens)
                )
            else:
                input_tokens = input_tokens[: self.max_seq_len]
                target_tokens = target_tokens[: self.max_seq_len]

            val_samples.append(
                {
                    "input": torch.tensor(input_tokens, dtype=torch.long),
                    "target": torch.tensor(target_tokens, dtype=torch.long),
                    "result": result,
                }
            )

        return val_samples


def create_dataloader(
    dataset: ModularArithmeticDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader[dict[str, Union[torch.Tensor, int]]]:
    """Create DataLoader for the dataset"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_task_configs() -> dict[str, dict[str, Union[int, float]]]:
    """Get task configurations as described in the paper (Figure 2)"""
    return {
        "gcd": {
            "modulus": 97,
            "train_split": 0.5,
        },
        "add": {
            "modulus": 97,
            "train_split": 0.8,
        },
        "div": {
            "modulus": 97,
            "train_split": 0.8,
        },
        "exp": {
            "modulus": 97,
            "train_split": 0.7,
        },
        "mul": {
            "modulus": 97,
            "train_split": 0.5,
        },
        "parity": {
            "modulus": 1024,
            "train_split": 0.5,
        },
    }
