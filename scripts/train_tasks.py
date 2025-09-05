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
    run_number: int = 1


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
        self.run_number = config.run_number

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.grokking_epoch = None

        if config.softmax_variant == "stablemax":
            self.softmax_fn = SoftmaxVariants.stablemax
        elif config.softmax_variant == "sparsemax":
            self.softmax_fn = SoftmaxVariants.sparsemax
        else:
            self.softmax_fn = SoftmaxVariants.standard_softmax

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch"""
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
        self,
        max_epochs: int = 500,
        grokking_threshold: float = 0.95,
        patience: int = 100,  # Increased patience for delayed grokking
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

        if quick_mode:
            patience = min(patience, 50)

        pbar = tqdm(range(max_epochs), desc=f"Run {self.run_number}")
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            val_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Update progress bar postfix with current metrics
            pbar.set_postfix(
                {"loss": f"{train_loss:.3f}", "val_acc": f"{val_acc:.3f}"}
            )

            if (
                val_acc >= grokking_threshold
                and train_acc >= 0.99
                and self.grokking_epoch is None
            ):
                self.grokking_epoch = epoch
                # Complete progress bar and break
                pbar.total = epoch
                pbar.refresh()
                break

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # Complete progress bar and break
                pbar.total = epoch
                pbar.refresh()
                break

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
    run_number: int = 1


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single experiment

    Args:
        config: Experiment configuration

    Returns:
        Experiment results
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    task_configs = get_task_configs()
    task_config = task_configs[config.task_type]

    dataset = ModularArithmeticDataset(
        DatasetConfig(
            task_type=config.task_type,
            modulus=task_config["modulus"],
            train_split=task_config["train_split"],
            max_seq_len=config.model_config["max_seq_len"],
            seed=config.seed,
        )
    )

    train_loader = DataLoader(
        dataset, batch_size=config.model_config["batch_size"], shuffle=True
    )
    val_data = dataset.get_val_data()

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

    if config.optimizer_type == "muon":
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
                lr=config.optimizer_config["lr"],
                momentum=0.95,
                weight_decay=config.optimizer_config["weight_decay"],
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=config.optimizer_config["lr"] * 0.1,
                betas=config.optimizer_config["betas"],
                weight_decay=config.optimizer_config["weight_decay"],
            ),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.optimizer_config["lr"],
            betas=config.optimizer_config["betas"],
            weight_decay=config.optimizer_config["weight_decay"],
        )

    trainer = GrokkingTrainer(
        TrainerConfig(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_data=val_data,
            device=config.device,
            softmax_variant=config.softmax_variant,
            run_number=config.run_number,
        )
    )

    results = trainer.train(
        max_epochs=config.model_config["max_epochs"],
        grokking_threshold=config.model_config["grokking_threshold"],
        quick_mode=config.quick_mode,
    )

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
    if single_task:
        model_config = {
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "ff_size": 128,
            "max_seq_len": 10,
            "batch_size": 32,
            "dropout": 0.2,
            "max_epochs": 10,
            "grokking_threshold": 0.95,
        }
    else:
        model_config = {
            "hidden_size": 256,
            "num_layers": 6,
            "num_heads": 8,
            "ff_size": 1024,
            "max_seq_len": 10,
            "batch_size": 32,
            "dropout": 0.2,
            "max_epochs": 1000,
            "grokking_threshold": 0.95,
        }

    muon_config = {
        "lr": 0.005,
        "betas": (
            0.9,
            0.98,
        ),
        "weight_decay": 1e-2,
    }

    adamw_config = {
        "lr": 0.0005,
        "betas": (0.9, 0.98),
        "weight_decay": 1e-2,
    }

    if single_task:
        task_types = ["exp"]
        softmax_variants = ["standard"]
        optimizer_types = ["muon", "adamw"]
    else:
        task_types = ["gcd", "add", "div", "exp", "mul", "parity"]
        softmax_variants = ["standard", "stablemax", "sparsemax"]
        optimizer_types = ["muon", "adamw"]

    all_results = []

    total_experiments = (
        len(task_types)
        * len(softmax_variants)
        * len(optimizer_types)
        * num_runs
    )
    print(f"🚀 Running {total_experiments} experiments ({num_runs} runs each)")

    for task_type in task_types:
        for softmax_variant in softmax_variants:
            print(f"\n=== {task_type.upper()} + {softmax_variant.upper()} ===")

            # Run all Muon experiments first
            print("\n--- Muon Optimizer ---")
            muon_results = []
            for run in range(num_runs):
                results = run_experiment(
                    ExperimentConfig(
                        task_type=task_type,
                        optimizer_type="muon",
                        softmax_variant=softmax_variant,
                        model_config=model_config,
                        optimizer_config=muon_config,
                        device=device,
                        seed=42 + run,
                        quick_mode=single_task,
                        run_number=run + 1,
                    )
                )

                muon_results.append(results)
                all_results.append(results)

            # Muon summary
            muon_grokking = [
                r["grokking_epoch"]
                for r in muon_results
                if r["grokking_epoch"] is not None
            ]
            muon_success_rate = len(muon_grokking) / len(muon_results) * 100
            if muon_grokking:
                muon_avg = sum(muon_grokking) / len(muon_grokking)
                grokking_epochs = [str(epoch) for epoch in muon_grokking]
                print(
                    f"📈 Muon: {muon_success_rate:.0f}% success, epochs: [{', '.join(grokking_epochs)}] (avg: {muon_avg:.1f})"
                )
            else:
                print(
                    f"📈 Muon: {muon_success_rate:.0f}% success, no grokking achieved"
                )

            # Run all AdamW experiments
            print("\n--- AdamW Optimizer ---")
            adamw_results = []
            for run in range(num_runs):
                results = run_experiment(
                    ExperimentConfig(
                        task_type=task_type,
                        optimizer_type="adamw",
                        softmax_variant=softmax_variant,
                        model_config=model_config,
                        optimizer_config=adamw_config,
                        device=device,
                        seed=42 + run,
                        quick_mode=single_task,
                        run_number=run + 1,
                    )
                )

                adamw_results.append(results)
                all_results.append(results)

            # AdamW summary
            adamw_grokking = [
                r["grokking_epoch"]
                for r in adamw_results
                if r["grokking_epoch"] is not None
            ]
            adamw_success_rate = len(adamw_grokking) / len(adamw_results) * 100
            if adamw_grokking:
                adamw_avg = sum(adamw_grokking) / len(adamw_grokking)
                grokking_epochs = [str(epoch) for epoch in adamw_grokking]
                print(
                    f"📈 AdamW: {adamw_success_rate:.0f}% success, epochs: [{', '.join(grokking_epochs)}] (avg: {adamw_avg:.1f})"
                )
            else:
                print(
                    f"📈 AdamW: {adamw_success_rate:.0f}% success, no grokking achieved"
                )

            # Show comparison for this task/variant
            muon_grokking = [
                r["grokking_epoch"]
                for r in muon_results
                if r["grokking_epoch"] is not None
            ]
            adamw_grokking = [
                r["grokking_epoch"]
                for r in adamw_results
                if r["grokking_epoch"] is not None
            ]

            if muon_grokking and adamw_grokking:
                muon_avg = sum(muon_grokking) / len(muon_grokking)
                adamw_avg = sum(adamw_grokking) / len(adamw_grokking)
                speedup = adamw_avg / muon_avg
                print(
                    f"\n📊 Comparison: Muon {muon_avg:.1f} epochs vs AdamW {adamw_avg:.1f} epochs (speedup: {speedup:.1f}x)"
                )
            elif muon_grokking:
                muon_avg = sum(muon_grokking) / len(muon_grokking)
                print(
                    f"\n📊 Muon grokked in {muon_avg:.1f} epochs, AdamW failed to grok"
                )
            elif adamw_grokking:
                adamw_avg = sum(adamw_grokking) / len(adamw_grokking)
                print(
                    f"\n📊 AdamW grokked in {adamw_avg:.1f} epochs, Muon failed to grok"
                )
            else:
                print(
                    "\n📊 Neither optimizer achieved grokking for this configuration"
                )

    return all_results


def save_results(results: list[dict], output_dir: str = "results"):
    """Save experiment results"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir, f"experiment_results_{timestamp}.json"
    )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

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

    print(f"💾 Results saved to {output_dir}/")

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
        results = run_comprehensive_experiments(device=args.device, num_runs=1)
    elif args.single_task:
        results = run_comprehensive_experiments(
            device=args.device, num_runs=args.num_runs, single_task=True
        )
    else:
        results = run_comprehensive_experiments(
            device=args.device, num_runs=args.num_runs
        )

    df = save_results(results, args.output_dir)

    print("\n📊 FINAL RESULTS")
    print("=" * 30)

    muon_results = df[df["optimizer_type"] == "muon"]
    adamw_results = df[df["optimizer_type"] == "adamw"]

    if len(muon_results) > 0 and len(adamw_results) > 0:
        muon_grokking = muon_results["grokking_epoch"].dropna()
        adamw_grokking = adamw_results["grokking_epoch"].dropna()

        if len(muon_grokking) > 0 and len(adamw_grokking) > 0:
            muon_avg = muon_grokking.mean()
            adamw_avg = adamw_grokking.mean()
            speedup = adamw_avg / muon_avg if muon_avg > 0 else float("inf")

            print(f"⚡ Muon: {muon_avg:.1f} epochs")
            print(f"🐌 AdamW: {adamw_avg:.1f} epochs")
            print(f"🚀 Speedup: {speedup:.1f}x")
        else:
            print("⚠️  No grokking detected in some configurations")

    print(f"✅ {len(results)} experiments completed")


if __name__ == "__main__":
    main()
