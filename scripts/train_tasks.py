import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    DatasetConfig,
    ModularArithmeticDataset,
    get_task_configs,
)
from src.model import GrokkingTransformer, ModelConfig, SoftmaxVariants


@dataclass
class TrainerConfig:
    """Configuration for the GrokkingTrainer"""

    model: nn.Module
    optimizer: optim.Optimizer
    train_loader: DataLoader
    val_data: list[dict]
    device: str = "cpu"
    softmax_variant: str = "standard"


class GrokkingTrainer:
    """
    Trainer class for grokking experiments
    Implements the training loop and evaluation as described in the paper
    """

    def __init__(self, config: TrainerConfig):
        """
        Initialize trainer

        Args:
            config: Trainer configuration containing all required components
        """
        self.model = config.model.to(config.device)
        self.optimizer = config.optimizer
        self.train_loader = config.train_loader
        self.val_data = config.val_data
        self.device = config.device
        self.softmax_variant = config.softmax_variant

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0
        )  # Ignore padding token

        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.grokking_epoch = None

        # Softmax function
        self.softmax_fn = getattr(
            SoftmaxVariants, f"{config.softmax_variant}_softmax"
        )

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in self.train_loader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            logits = self.model(inputs)

            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            # Compute loss
            loss = self.criterion(logits, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                mask = targets != 0  # Ignore padding
                correct_predictions += (
                    (predictions[mask] == targets[mask]).sum().item()
                )
                total_predictions += mask.sum().item()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return (
            avg_loss,
            correct_predictions / total_predictions
            if total_predictions > 0
            else 0.0,
        )

    def evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for sample in self.val_data:
                inputs = sample["input"].unsqueeze(0).to(self.device)
                targets = sample["target"].to(self.device)

                # Forward pass
                logits = self.model(inputs)

                # Reshape for evaluation
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)

                # Compute accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = targets != 0  # Ignore padding
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
        self,
        max_epochs: int = 500,
        grokking_threshold: float = 0.95,
        patience: int = 50,
        quick_mode: bool = False,
    ) -> dict:
        """
        Train the model and track grokking

        Args:
            max_epochs: Maximum number of epochs
            grokking_threshold: Validation accuracy threshold for grokking
            patience: Early stopping patience
            quick_mode: Use more aggressive early stopping for quick experiments

        Returns:
            Training results dictionary
        """
        best_val_acc = 0.0
        patience_counter = 0

        # Adjust patience for quick mode
        if quick_mode:
            patience = min(patience, 20)  # More aggressive early stopping

        print(f"Starting training with {self.softmax_variant} softmax...")

        for epoch in tqdm(range(max_epochs), desc="Training"):
            # Train one epoch
            train_loss, train_acc = self.train_epoch()
            val_acc = self.evaluate()

            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Check for grokking
            if val_acc >= grokking_threshold and self.grokking_epoch is None:
                self.grokking_epoch = epoch
                print(f"\nGrokking detected at epoch {epoch}!")
                print(f"Validation accuracy: {val_acc:.4f}")
                # Stop training immediately when grokking is detected
                break

            # Early stopping (only if grokking hasn't been detected)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            # Print progress
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
                )

        # Compile results
        return {
            "grokking_epoch": self.grokking_epoch,
            "final_train_loss": self.train_losses[-1],
            "final_train_acc": self.train_accuracies[-1],
            "final_val_acc": self.val_accuracies[-1],
            "best_val_acc": best_val_acc,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "total_epochs": len(self.train_losses),
        }


@dataclass
class ExperimentConfig:
    """Configuration for running experiments"""

    task_type: str
    optimizer_type: str
    softmax_variant: str
    model_config: dict
    optimizer_config: dict
    device: str = "cpu"
    seed: int = 42
    quick_mode: bool = False


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single experiment

    Args:
        config: Experiment configuration

    Returns:
        Experiment results
    """
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Get task configuration
    task_configs = get_task_configs()
    task_config = task_configs[config.task_type]

    # Create dataset
    dataset = ModularArithmeticDataset(
        DatasetConfig(
            task_type=config.task_type,
            modulus=task_config["modulus"],
            train_split=task_config["train_split"],
            max_seq_len=config.model_config["max_seq_len"],
            seed=config.seed,
        )
    )

    # Create data loader
    train_loader = DataLoader(
        dataset, batch_size=config.model_config["batch_size"], shuffle=True
    )
    val_data = dataset.get_val_data()

    # Create model
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
    if config.optimizer_type == "muon":
        # Use original Muon implementation with proper parameter grouping
        # Hidden weights (2D+ parameters) use Muon, others use AdamW
        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]

        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=config.optimizer_config["lr"],
                momentum=0.95,  # Default Muon momentum
                weight_decay=config.optimizer_config["weight_decay"],
            ),
            dict(
                params=other_params,
                use_muon=False,
                lr=config.optimizer_config["lr"]
                * 0.1,  # Lower LR for non-hidden params
                betas=config.optimizer_config["betas"],
                weight_decay=config.optimizer_config["weight_decay"],
            ),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:  # AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.optimizer_config["lr"],
            betas=config.optimizer_config["betas"],
            weight_decay=config.optimizer_config["weight_decay"],
        )

    # Create trainer
    trainer = GrokkingTrainer(
        TrainerConfig(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_data=val_data,
            device=config.device,
            softmax_variant=config.softmax_variant,
        )
    )

    # Train
    results = trainer.train(
        max_epochs=config.model_config["max_epochs"],
        grokking_threshold=config.model_config["grokking_threshold"],
        quick_mode=config.quick_mode,
    )

    # Add metadata
    results.update(
        {
            "task_type": config.task_type,
            "optimizer_type": config.optimizer_type,
            "softmax_variant": config.softmax_variant,
            "model_config": config.model_config,
            "optimizer_config": config.optimizer_config,
            "seed": config.seed,
        }
    )

    return results


def run_comprehensive_experiments(
    device: str = "cpu", num_runs: int = 3, single_task: bool = False
) -> list[dict]:
    """
    Run comprehensive experiments as described in the paper

    Args:
        device: Device to run on
        num_runs: Number of runs per configuration
        single_task: Run only one task for the fastest possible test

    Returns:
        List of experiment results
    """
    model_config = {
        "hidden_size": 8,  # Extremely small model
        "num_layers": 1,  # Single layer
        "num_heads": 1,  # Single head
        "ff_size": 32,  # Very small feedforward
        "max_seq_len": 6,  # Very short sequences
        "batch_size": 4,  # Small batch
        "dropout": 0.5,  # High dropout
        "max_epochs": 50,  # More epochs to see grokking
        "grokking_threshold": 0.95,  # Match paper threshold
    }

    # Optimizer configurations - based on paper findings
    muon_config = {
        "lr": 0.02,  # Original Muon default learning rate
        "betas": (0.9, 0.95),  # For non-hidden parameters
        "weight_decay": 1e-2,  # Weight decay for both groups
    }

    adamw_config = {
        "lr": 5e-4,  # Standard learning rate
        "betas": (0.9, 0.98),
        "weight_decay": 1e-1,  # Higher weight decay needed for AdamW to achieve grokking
    }

    # Task types and softmax variants
    if single_task:
        # Single task: only 1 task, 1 softmax, 1 optimizer = 1 experiment total
        task_types = ["exp"]  # Exponentiation is the hardest task
        softmax_variants = ["standard"]
        optimizer_types = ["muon", "adamw"]  # Compare both optimizers
    else:
        # Full: all configurations
        task_types = ["gcd", "add", "div", "exp", "mul", "parity"]
        softmax_variants = ["standard", "stablemax", "sparsemax"]
        optimizer_types = ["muon", "adamw"]

    all_results = []

    print("Starting comprehensive experiments...")
    print(f"Mode: {'Single task' if single_task else 'Full'}")
    print(f"Tasks: {task_types}")
    print(f"Softmax variants: {softmax_variants}")
    print(f"Optimizers: {optimizer_types}")
    print(f"Number of runs per config: {num_runs}")
    print(f"Max epochs: {model_config['max_epochs']}")
    print(
        f"Total experiments: {len(task_types) * len(softmax_variants) * len(optimizer_types) * num_runs}"
    )

    for task_type in task_types:
        for softmax_variant in softmax_variants:
            for optimizer_type in optimizer_types:
                for run in range(num_runs):
                    print(
                        f"\nRunning: {task_type} + {softmax_variant} + {optimizer_type} (run {run + 1})"
                    )

                    # Select optimizer config
                    optimizer_config = (
                        muon_config
                        if optimizer_type == "muon"
                        else adamw_config
                    )

                    # Run experiment
                    results = run_experiment(
                        ExperimentConfig(
                            task_type=task_type,
                            optimizer_type=optimizer_type,
                            softmax_variant=softmax_variant,
                            model_config=model_config,
                            optimizer_config=optimizer_config,
                            device=device,
                            seed=42 + run,
                            quick_mode=single_task,
                        )
                    )

                    all_results.append(results)

                    # Print results
                    if results["grokking_epoch"] is not None:
                        print(
                            f"  Grokking at epoch: {results['grokking_epoch']}"
                        )
                    else:
                        print(
                            f"  No grokking detected (final val acc: {results['final_val_acc']:.4f})"
                        )

    return all_results


def save_results(results: list[dict], output_dir: str = "results"):
    """Save experiment results"""
    os.makedirs(output_dir, exist_ok=True)

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir, f"experiment_results_{timestamp}.json"
    )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create summary DataFrame
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
            }
        )

    df = pd.DataFrame(summary_data)
    summary_file = os.path.join(
        output_dir, f"experiment_summary_{timestamp}.csv"
    )
    df.to_csv(summary_file, index=False)

    print(f"Results saved to {output_dir}")
    print(f"Raw results: {results_file}")
    print(f"Summary: {summary_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run Muon Optimizer grokking experiments"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with fewer configurations",
    )
    parser.add_argument(
        "--single_task",
        action="store_true",
        help="Run only one task for the fastest possible test",
    )

    args = parser.parse_args()

    if args.quick_test:
        # Quick test with limited configurations
        print("Running quick test...")
        results = run_comprehensive_experiments(device=args.device, num_runs=1)
    elif args.single_task:
        # Single task test
        print("Running single task test...")
        results = run_comprehensive_experiments(
            device=args.device, num_runs=1, single_task=True
        )
    else:
        # Full experiments
        results = run_comprehensive_experiments(
            device=args.device, num_runs=args.num_runs
        )

    # Save results
    df = save_results(results, args.output_dir)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)

    # Compare Muon vs AdamW
    muon_results = df[df["optimizer_type"] == "muon"]
    adamw_results = df[df["optimizer_type"] == "adamw"]

    if len(muon_results) > 0 and len(adamw_results) > 0:
        muon_grokking = muon_results["grokking_epoch"].dropna()
        adamw_grokking = adamw_results["grokking_epoch"].dropna()

        if len(muon_grokking) > 0 and len(adamw_grokking) > 0:
            print(f"Muon average grokking epoch: {muon_grokking.mean():.2f}")
            print(f"AdamW average grokking epoch: {adamw_grokking.mean():.2f}")

            # Handle division by zero when Muon achieves grokking at epoch 0
            if muon_grokking.mean() == 0:
                print("Speedup: ∞x (Muon achieves immediate grokking)")
            else:
                print(
                    f"Speedup: {adamw_grokking.mean() / muon_grokking.mean():.2f}x"
                )
        else:
            print("No grokking detected in some configurations")

    print(f"Total experiments completed: {len(results)}")


if __name__ == "__main__":
    main()
