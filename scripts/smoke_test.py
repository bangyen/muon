"""
CI Smoke Test Script for Muon Optimizer Grokking Reproduction

This script runs a minimal grokking experiment to verify that the system works
correctly in CI environments. It's designed to complete within 2 minutes and
produces the same output format as the full experiments for artifact upload.

The smoke test:
- Uses the smallest possible model configuration
- Runs only one task (modular exponentiation)
- Uses only the standard softmax variant
- Limits training to 5 epochs maximum
- Compares Muon vs AdamW optimizers
- Saves results in the same JSON/CSV format as full experiments
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch import nn, optim
from torch.utils.data import DataLoader

from src.dataset import DatasetConfig, ModularArithmeticDataset
from src.model import GrokkingTransformer, ModelConfig


class TimeoutError(Exception):
    """Custom timeout exception"""

    pass


def timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for timeout"""
    raise TimeoutError("Smoke test timed out")


@dataclass
class SmokeTestConfig:
    """Configuration for smoke test"""

    # Minimal model configuration for fast execution
    model_config: dict
    # Optimizer configurations
    muon_config: dict
    adamw_config: dict
    # Test parameters
    device: str = "cpu"
    seed: int = 42
    max_epochs: int = 5
    grokking_threshold: float = 0.95
    timeout_seconds: int = 120


class SmokeTestTrainer:
    """
    Minimal trainer for smoke testing

    Simplified version of GrokkingTrainer optimized for speed and CI environments.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_data: list[dict],
        device: str = "cpu",
    ):
        """Initialize smoke test trainer"""
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_data = val_data
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Track metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.grokking_epoch = None

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch (simplified)"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in self.train_loader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            logits = self.model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                mask = targets != 0
                correct_predictions += (
                    (predictions[mask] == targets[mask]).sum().item()
                )
                total_predictions += mask.sum().item()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = (
            correct_predictions / total_predictions
            if total_predictions > 0
            else 0.0
        )
        return avg_loss, accuracy

    def evaluate(self) -> float:
        """Evaluate on validation set (simplified)"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for sample in self.val_data:
                inputs = sample["input"].unsqueeze(0).to(self.device)
                targets = sample["target"].to(self.device)

                logits = self.model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)

                predictions = torch.argmax(logits, dim=-1)
                mask = targets != 0
                correct_predictions += (
                    (predictions[mask] == targets[mask]).sum().item()
                )
                total_predictions += mask.sum().item()

        return (
            correct_predictions / total_predictions
            if total_predictions > 0
            else 0.0
        )

    def train(
        self, max_epochs: int = 5, grokking_threshold: float = 0.95
    ) -> dict:
        """Train with early stopping for grokking"""
        best_val_acc = 0.0

        for epoch in range(max_epochs):
            train_loss, train_acc = self.train_epoch()
            val_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Check for grokking
            if (
                val_acc >= grokking_threshold
                and train_acc >= 0.99
                and self.grokking_epoch is None
            ):
                self.grokking_epoch = epoch
                break

            # Update best validation accuracy
            best_val_acc = max(best_val_acc, val_acc)

        return {
            "grokking_epoch": self.grokking_epoch,
            "final_train_loss": self.train_losses[-1]
            if self.train_losses
            else 0.0,
            "final_train_acc": self.train_accuracies[-1]
            if self.train_accuracies
            else 0.0,
            "final_val_acc": self.val_accuracies[-1]
            if self.val_accuracies
            else 0.0,
            "best_val_acc": best_val_acc,
            "total_epochs": len(self.train_losses),
        }


def run_smoke_experiment(config: SmokeTestConfig, optimizer_type: str) -> dict:
    """
    Run a single smoke test experiment

    Args:
        config: Smoke test configuration
        optimizer_type: Either 'muon' or 'adamw'

    Returns:
        Experiment results dictionary
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create dataset (modular exponentiation task)
    dataset = ModularArithmeticDataset(
        DatasetConfig(
            task_type="exp",
            modulus=97,
            train_split=0.7,
            max_seq_len=config.model_config["max_seq_len"],
            seed=config.seed,
        )
    )

    train_loader = DataLoader(
        dataset, batch_size=config.model_config["batch_size"], shuffle=True
    )
    val_data = dataset.get_val_data()

    # Create minimal model
    model = GrokkingTransformer(
        ModelConfig(
            vocab_size=dataset.vocab_size,
            hidden_size=config.model_config["hidden_size"],
            num_layers=config.model_config["num_layers"],
            num_heads=config.model_config["num_heads"],
            ff_size=config.model_config["ff_size"],
            max_seq_len=config.model_config["max_seq_len"],
            dropout=config.model_config["dropout"],
        )
    )

    # Create optimizer
    if optimizer_type == "muon":
        # Separate parameters for Muon optimizer
        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if (
                param.ndim >= 2
                and (
                    "attention" in name
                    or "feed_forward" in name
                    or "blocks" in name
                )
                and "weight" in name
                and "embedding" not in name
                and "norm" not in name
            ):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=config.muon_config["lr"],
                momentum=0.95,
                weight_decay=config.muon_config["weight_decay"],
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=config.muon_config["lr"] * 0.1,
                betas=config.muon_config["betas"],
                weight_decay=config.muon_config["weight_decay"],
            ),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        optimizer_config = config.muon_config
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.adamw_config["lr"],
            betas=config.adamw_config["betas"],
            weight_decay=config.adamw_config["weight_decay"],
        )
        optimizer_config = config.adamw_config

    # Train model
    trainer = SmokeTestTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_data=val_data,
        device=config.device,
    )

    results = trainer.train(
        max_epochs=config.max_epochs,
        grokking_threshold=config.grokking_threshold,
    )

    # Add metadata
    results.update(
        {
            "task_type": "exp",
            "optimizer_type": optimizer_type,
            "softmax_variant": "standard",
            "model_config": config.model_config,
            "optimizer_config": optimizer_config,
            "seed": config.seed,
            "smoke_test": True,
        }
    )

    return results


def save_smoke_results(
    results: list[dict], output_dir: str = "results"
) -> None:
    """Save smoke test results in the same format as full experiments"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir, f"experiment_results_{timestamp}.json"
    )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create summary CSV
    summary_data = []
    for result in results:
        summary_data.append(
            {
                "task_type": result["task_type"],
                "optimizer_type": result["optimizer_type"],
                "softmax_variant": result["softmax_variant"],
                "grokking_epoch": result["grokking_epoch"],
                "final_val_acc": result["final_val_acc"],
                "best_val_acc": result["best_val_acc"],
                "total_epochs": result["total_epochs"],
                "seed": result["seed"],
                "smoke_test": result.get("smoke_test", False),
            }
        )

    df = pd.DataFrame(summary_data)
    summary_file = os.path.join(
        output_dir, f"experiment_summary_{timestamp}.csv"
    )
    df.to_csv(summary_file, index=False)

    print(f"Results saved to {results_file}")
    print(f"Summary saved to {summary_file}")


def main():
    """Main entry point for smoke test"""
    parser = argparse.ArgumentParser(
        description="Run Muon Optimizer smoke test"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on"
    )

    args = parser.parse_args()

    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    try:
        print("=== Muon Optimizer Smoke Test ===")
        print(f"Timeout: {args.timeout} seconds")
        print(f"Device: {args.device}")

        # Minimal configuration for fast execution
        config = SmokeTestConfig(
            model_config={
                "hidden_size": 32,
                "num_layers": 2,
                "num_heads": 4,
                "ff_size": 128,
                "max_seq_len": 7,
                "batch_size": 16,
                "dropout": 0.1,
            },
            muon_config={
                "lr": 0.01,
                "betas": (0.9, 0.98),
                "weight_decay": 1e-2,
            },
            adamw_config={
                "lr": 0.001,
                "betas": (0.9, 0.98),
                "weight_decay": 1e-2,
            },
            device=args.device,
            timeout_seconds=args.timeout,
        )

        start_time = time.time()

        # Run experiments
        print("\nRunning Muon optimizer test...")
        muon_results = run_smoke_experiment(config, "muon")

        print("Running AdamW optimizer test...")
        adamw_results = run_smoke_experiment(config, "adamw")

        all_results = [muon_results, adamw_results]

        # Save results
        save_smoke_results(all_results, args.output_dir)

        # Display summary
        print("\n=== Smoke Test Results ===")
        for result in all_results:
            print(
                f"{result['optimizer_type'].upper()}: "
                f"Grokking Epoch: {result['grokking_epoch']}, "
                f"Final Val Acc: {result['final_val_acc']:.3f}, "
                f"Total Epochs: {result['total_epochs']}"
            )

        elapsed_time = time.time() - start_time
        print(f"\nSmoke test completed in {elapsed_time:.1f} seconds")

        # Check if we achieved the expected behavior
        muon_grokking = muon_results["grokking_epoch"] is not None
        adamw_grokking = adamw_results["grokking_epoch"] is not None

        if muon_grokking and adamw_grokking:
            speedup = (
                adamw_results["grokking_epoch"]
                / muon_results["grokking_epoch"]
            )
            print(f"Speedup: {speedup:.1f}x (Muon faster than AdamW)")
        elif muon_grokking and not adamw_grokking:
            print("Muon achieved grokking, AdamW did not (expected behavior)")
        elif not muon_grokking and not adamw_grokking:
            print(
                "Neither optimizer achieved grokking (acceptable for smoke test)"
            )
        else:
            print("AdamW achieved grokking but Muon did not (unexpected)")

        print("✅ Smoke test passed!")

    except TimeoutError:
        print(f"❌ Smoke test timed out after {args.timeout} seconds")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Smoke test failed with error: {e}")
        sys.exit(1)
    finally:
        # Cancel timeout
        signal.alarm(0)


if __name__ == "__main__":
    main()
