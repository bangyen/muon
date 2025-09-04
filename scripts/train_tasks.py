import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ModularArithmeticDataset, get_task_configs
from src.model import GrokkingTransformer, SoftmaxVariants
from src.optimizer import MuonOptimizer


class GrokkingTrainer:
    """
    Trainer class for grokking experiments
    Implements the training loop and evaluation as described in the paper
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_data: list[dict],
        device: str = "cpu",
        softmax_variant: str = "standard",
    ):
        """
        Initialize trainer

        Args:
            model: Transformer model
            optimizer: Optimizer (Muon or AdamW)
            train_loader: Training data loader
            val_data: Validation data
            device: Device to run on
            softmax_variant: Type of softmax ('standard', 'stablemax', 'sparsemax')
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_data = val_data
        self.device = device
        self.softmax_variant = softmax_variant

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
            SoftmaxVariants, f"{softmax_variant}_softmax"
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
        accuracy = (
            correct_predictions / total_predictions
            if total_predictions > 0
            else 0.0
        )

        return avg_loss, accuracy

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

        accuracy = (
            correct_predictions / total_predictions
            if total_predictions > 0
            else 0.0
        )
        return accuracy

    def train(
        self,
        max_epochs: int = 500,
        grokking_threshold: float = 0.95,
        patience: int = 50,
    ) -> dict:
        """
        Train the model and track grokking

        Args:
            max_epochs: Maximum number of epochs
            grokking_threshold: Validation accuracy threshold for grokking
            patience: Early stopping patience

        Returns:
            Training results dictionary
        """
        best_val_acc = 0.0
        patience_counter = 0

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

            # Early stopping
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
        results = {
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

        return results


def run_experiment(
    task_type: str,
    optimizer_type: str,
    softmax_variant: str,
    model_config: dict,
    optimizer_config: dict,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Run a single experiment

    Args:
        task_type: Type of arithmetic task
        optimizer_type: 'muon' or 'adamw'
        softmax_variant: Type of softmax
        model_config: Model hyperparameters
        optimizer_config: Optimizer hyperparameters
        device: Device to run on
        seed: Random seed

    Returns:
        Experiment results
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get task configuration
    task_configs = get_task_configs()
    task_config = task_configs[task_type]

    # Create dataset
    dataset = ModularArithmeticDataset(
        task_type=task_type,
        modulus=task_config["modulus"],
        train_split=task_config["train_split"],
        max_seq_len=model_config["max_seq_len"],
        seed=seed,
    )

    # Create data loader
    train_loader = DataLoader(
        dataset, batch_size=model_config["batch_size"], shuffle=True
    )
    val_data = dataset.get_val_data()

    # Create model
    model = GrokkingTransformer(
        vocab_size=dataset.vocab_size,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_size=model_config["ff_size"],
        max_seq_len=model_config["max_seq_len"],
        dropout=model_config["dropout"],
    )

    # Create optimizer
    if optimizer_type == "muon":
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=optimizer_config["betas"],
            weight_decay=optimizer_config["weight_decay"],
            spectral_norm_strength=optimizer_config["spectral_norm_strength"],
            second_order_interval=optimizer_config["second_order_interval"],
        )
    else:  # AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=optimizer_config["betas"],
            weight_decay=optimizer_config["weight_decay"],
        )

    # Create trainer
    trainer = GrokkingTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_data=val_data,
        device=device,
        softmax_variant=softmax_variant,
    )

    # Train
    results = trainer.train(
        max_epochs=model_config["max_epochs"],
        grokking_threshold=model_config["grokking_threshold"],
    )

    # Add metadata
    results.update(
        {
            "task_type": task_type,
            "optimizer_type": optimizer_type,
            "softmax_variant": softmax_variant,
            "model_config": model_config,
            "optimizer_config": optimizer_config,
            "seed": seed,
        }
    )

    return results


def run_comprehensive_experiments(
    device: str = "cpu", num_runs: int = 3
) -> list[dict]:
    """
    Run comprehensive experiments as described in the paper

    Args:
        device: Device to run on
        num_runs: Number of runs per configuration

    Returns:
        List of experiment results
    """
    # Model configuration
    model_config = {
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 8,
        "ff_size": 512,
        "max_seq_len": 10,
        "batch_size": 32,
        "dropout": 0.1,
        "max_epochs": 300,
        "grokking_threshold": 0.95,
    }

    # Optimizer configurations
    muon_config = {
        "lr": 1e-3,
        "betas": (0.9, 0.98),
        "weight_decay": 1e-2,
        "spectral_norm_strength": 0.1,
        "second_order_interval": 10,
    }

    adamw_config = {"lr": 1e-3, "betas": (0.9, 0.98), "weight_decay": 1e-2}

    # Task types and softmax variants
    task_types = ["gcd", "add", "div", "exp", "mul", "parity"]
    softmax_variants = ["standard", "stablemax", "sparsemax"]
    optimizer_types = ["muon", "adamw"]

    all_results = []

    print("Starting comprehensive experiments...")
    print(f"Tasks: {task_types}")
    print(f"Softmax variants: {softmax_variants}")
    print(f"Optimizers: {optimizer_types}")
    print(f"Number of runs per config: {num_runs}")

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
                        task_type=task_type,
                        optimizer_type=optimizer_type,
                        softmax_variant=softmax_variant,
                        model_config=model_config,
                        optimizer_config=optimizer_config,
                        device=device,
                        seed=42 + run,
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

    args = parser.parse_args()

    if args.quick_test:
        # Quick test with limited configurations
        print("Running quick test...")
        results = run_comprehensive_experiments(device=args.device, num_runs=1)
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
            print(
                f"Speedup: {adamw_grokking.mean() / muon_grokking.mean():.2f}x"
            )
        else:
            print("No grokking detected in some configurations")

    print(f"Total experiments completed: {len(results)}")


if __name__ == "__main__":
    main()
