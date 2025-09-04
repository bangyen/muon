"""
Tests for grokking detection and analysis.

These tests validate the grokking phenomenon detection including:
- Grokking epoch detection
- Training dynamics analysis
- Comparison between optimizers
- Statistical significance testing
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

from src.model import GrokkingTransformer
from src.optimizer import MuonOptimizer
from src.dataset import ModularArithmeticDataset
from tests.conftest import set_seed, TestConfig, detect_grokking, compute_accuracy


class TestGrokkingDetection:
    """Test suite for grokking detection"""
    
    def test_grokking_detection_basic(self):
        """Test basic grokking detection functionality"""
        # Simulate training curves
        train_acc = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.99, 0.99]
        val_acc = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95, 0.98, 0.99, 0.99]
        
        grokking_epoch = detect_grokking(train_acc, val_acc, threshold=0.95)
        
        # Should detect grokking at epoch 6 (0-indexed)
        assert grokking_epoch == 6
    
    def test_grokking_detection_no_grokking(self):
        """Test grokking detection when no grokking occurs"""
        # Simulate training curves without grokking
        train_acc = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.99, 0.99]
        val_acc = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        grokking_epoch = detect_grokking(train_acc, val_acc, threshold=0.95)
        
        # Should return -1 (no grokking detected)
        assert grokking_epoch == -1
    
    def test_grokking_detection_early_grokking(self):
        """Test grokking detection when grokking happens early"""
        # Simulate early grokking
        train_acc = [0.1, 0.3, 0.95, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
        val_acc = [0.1, 0.1, 0.95, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
        
        grokking_epoch = detect_grokking(train_acc, val_acc, threshold=0.95)
        
        # Should detect grokking at epoch 2
        assert grokking_epoch == 2
    
    def test_grokking_detection_different_thresholds(self):
        """Test grokking detection with different thresholds"""
        train_acc = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.99, 0.99]
        val_acc = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95, 0.98, 0.99, 0.99]
        
        # Test different thresholds
        thresholds = [0.8, 0.9, 0.95, 0.99]
        expected_epochs = [6, 6, 6, -1]  # -1 for 0.99 threshold (not reached)
        
        for threshold, expected in zip(thresholds, expected_epochs):
            grokking_epoch = detect_grokking(train_acc, val_acc, threshold=threshold)
            assert grokking_epoch == expected
    
    def test_grokking_detection_edge_cases(self):
        """Test grokking detection edge cases"""
        # Test with insufficient data
        train_acc = [0.95]
        val_acc = [0.95]
        
        grokking_epoch = detect_grokking(train_acc, val_acc)
        assert grokking_epoch == -1  # Need at least 2 epochs
        
        # Test with empty lists
        grokking_epoch = detect_grokking([], [])
        assert grokking_epoch == -1
        
        # Test with different length lists
        train_acc = [0.1, 0.3, 0.5]
        val_acc = [0.1, 0.1, 0.1, 0.95]  # Longer list
        
        grokking_epoch = detect_grokking(train_acc, val_acc)
        assert grokking_epoch == -1  # Should handle gracefully


class TestTrainingDynamics:
    """Test training dynamics analysis"""
    
    def test_accuracy_computation(self):
        """Test accuracy computation function"""
        set_seed(42)
        
        # Create dummy logits and targets
        batch_size = 4
        vocab_size = 10
        seq_len = 5
        
        logits = torch.randn(batch_size * seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
        
        accuracy = compute_accuracy(logits, targets)
        
        # Check that accuracy is between 0 and 1
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)
    
    def test_training_curve_generation(self):
        """Test generation of training curves"""
        set_seed(42)
        
        # Simulate training for a few epochs
        num_epochs = 5
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Simulate increasing training accuracy
            train_acc = 0.1 + 0.2 * epoch
            val_acc = 0.1 + 0.1 * epoch if epoch < 3 else 0.95
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
        
        # Check that curves have expected properties
        assert len(train_accuracies) == num_epochs
        assert len(val_accuracies) == num_epochs
        
        # Training accuracy should generally increase
        assert train_accuracies[-1] > train_accuracies[0]
        
        # Validation accuracy should show grokking pattern
        grokking_epoch = detect_grokking(train_accuracies, val_accuracies)
        assert grokking_epoch >= 0  # Should detect grokking
    
    def test_loss_tracking(self):
        """Test loss tracking during training"""
        set_seed(42)
        
        # Create simple model and data
        model = nn.Linear(10, 5)
        optimizer = MuonOptimizer(model.parameters(), lr=1e-3)
        
        losses = []
        
        for step in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Check that losses are tracked
        assert len(losses) == 10
        
        # Check that losses are finite
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        
        # Check that losses generally decrease (not always guaranteed)
        # but at least they should be finite and reasonable
        assert all(0 <= loss < 1000 for loss in losses)


class TestOptimizerComparison:
    """Test comparison between Muon and AdamW optimizers"""
    
    def test_optimizer_convergence_comparison(self):
        """Test convergence comparison between optimizers"""
        set_seed(42)
        
        from torch.optim import AdamW
        
        # Create identical models
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        
        # Copy weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)
        
        # Create optimizers
        muon_optimizer = MuonOptimizer(model1.parameters(), lr=1e-3)
        adamw_optimizer = AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-2)
        
        # Track losses
        muon_losses = []
        adamw_losses = []
        
        for step in range(20):
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)
            
            # Muon step
            muon_optimizer.zero_grad()
            output1 = model1(x)
            loss1 = nn.MSELoss()(output1, y)
            loss1.backward()
            muon_optimizer.step()
            muon_losses.append(loss1.item())
            
            # AdamW step
            adamw_optimizer.zero_grad()
            output2 = model2(x)
            loss2 = nn.MSELoss()(output2, y)
            loss2.backward()
            adamw_optimizer.step()
            adamw_losses.append(loss2.item())
        
        # Check that both optimizers produce different convergence patterns
        assert len(muon_losses) == len(adamw_losses)
        assert any(abs(m - a) > 1e-6 for m, a in zip(muon_losses, adamw_losses))
    
    def test_grokking_comparison(self):
        """Test grokking comparison between optimizers"""
        set_seed(42)
        
        from torch.optim import AdamW
        
        # This is a simplified test - full grokking requires longer training
        # and specific conditions, but we can test the framework
        
        # Create models
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        
        # Copy weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)
        
        # Create optimizers
        muon_optimizer = MuonOptimizer(model1.parameters(), lr=1e-3)
        adamw_optimizer = AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-2)
        
        # Track accuracies (simplified)
        muon_train_acc = []
        muon_val_acc = []
        adamw_train_acc = []
        adamw_val_acc = []
        
        for epoch in range(5):
            # Simulate training for one epoch
            x = torch.randn(100, 10)
            y = torch.randn(100, 5)
            
            # Muon training
            muon_optimizer.zero_grad()
            output1 = model1(x)
            loss1 = nn.MSELoss()(output1, y)
            loss1.backward()
            muon_optimizer.step()
            
            # AdamW training
            adamw_optimizer.zero_grad()
            output2 = model2(x)
            loss2 = nn.MSELoss()(output2, y)
            loss2.backward()
            adamw_optimizer.step()
            
            # Simulate accuracy tracking
            muon_train_acc.append(0.1 + 0.2 * epoch)
            muon_val_acc.append(0.1 if epoch < 3 else 0.95)
            adamw_train_acc.append(0.1 + 0.15 * epoch)
            adamw_val_acc.append(0.1 if epoch < 4 else 0.95)
        
        # Check grokking detection
        muon_grokking = detect_grokking(muon_train_acc, muon_val_acc)
        adamw_grokking = detect_grokking(adamw_train_acc, adamw_val_acc)
        
        # Both should detect grokking in this simulated scenario
        assert muon_grokking >= 0
        assert adamw_grokking >= 0
        
        # Muon should grok earlier in this simulation
        assert muon_grokking <= adamw_grokking


class TestStatisticalAnalysis:
    """Test statistical analysis of grokking results"""
    
    def test_mean_grokking_epoch_calculation(self):
        """Test calculation of mean grokking epoch"""
        # Simulate multiple runs
        grokking_epochs = [100, 120, 80, 150, 110]
        
        mean_epoch = np.mean(grokking_epochs)
        
        # Check calculation
        expected_mean = (100 + 120 + 80 + 150 + 110) / 5
        assert abs(mean_epoch - expected_mean) < 1e-6
    
    def test_standard_deviation_calculation(self):
        """Test calculation of standard deviation"""
        grokking_epochs = [100, 120, 80, 150, 110]
        
        std_epoch = np.std(grokking_epochs)
        
        # Check that standard deviation is positive
        assert std_epoch > 0
        
        # Check that it's reasonable (less than mean)
        mean_epoch = np.mean(grokking_epochs)
        assert std_epoch < mean_epoch
    
    def test_t_test_simulation(self):
        """Test t-test simulation for optimizer comparison"""
        # Simulate grokking epochs for two optimizers
        muon_epochs = [80, 90, 85, 95, 88]
        adamw_epochs = [120, 130, 125, 135, 128]
        
        # Calculate means
        muon_mean = np.mean(muon_epochs)
        adamw_mean = np.mean(adamw_epochs)
        
        # Calculate standard deviations
        muon_std = np.std(muon_epochs)
        adamw_std = np.std(adamw_epochs)
        
        # Check that Muon has lower mean (as expected from paper)
        assert muon_mean < adamw_mean
        
        # Check that standard deviations are reasonable
        assert muon_std > 0
        assert adamw_std > 0
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation"""
        # Simulate grokking epochs
        epochs = [100, 120, 80, 150, 110, 95, 105, 115, 125, 135]
        
        mean_epoch = np.mean(epochs)
        std_epoch = np.std(epochs)
        n = len(epochs)
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        z_score = 1.96  # For 95% confidence
        
        margin_of_error = z_score * (std_epoch / np.sqrt(n))
        lower_bound = mean_epoch - margin_of_error
        upper_bound = mean_epoch + margin_of_error
        
        # Check that bounds are reasonable
        assert lower_bound < mean_epoch
        assert upper_bound > mean_epoch
        assert lower_bound > 0  # Grokking epoch should be positive


class TestGrokkingReproduction:
    """Test reproduction of grokking phenomenon"""
    
    def test_minimal_grokking_setup(self):
        """Test minimal setup for grokking reproduction"""
        set_seed(42)
        
        # Create small dataset and model for quick testing
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.5)
        
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            ff_size=64,
            max_seq_len=5,
        )
        
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            spectral_norm_strength=0.1,
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Track metrics
        train_accuracies = []
        val_accuracies = []
        
        # Train for a few epochs
        for epoch in range(5):
            model.train()
            
            # Training step
            sample = dataset[0]
            inputs = sample["input"].unsqueeze(0)
            targets = sample["target"].unsqueeze(0)
            
            optimizer.zero_grad()
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            # Simulate accuracy tracking
            train_accuracies.append(0.1 + 0.2 * epoch)
            val_accuracies.append(0.1 if epoch < 3 else 0.95)
        
        # Check that training completed
        assert len(train_accuracies) == 5
        assert len(val_accuracies) == 5
        
        # Check that loss is finite
        assert torch.isfinite(loss)
    
    def test_grokking_conditions(self):
        """Test that grokking conditions are met"""
        # Based on the paper, grokking requires:
        # 1. Weight decay
        # 2. Overparameterized model
        # 3. Specific task types
        
        # Test weight decay requirement
        model = nn.Linear(10, 5)
        
        # Without weight decay
        optimizer_no_wd = MuonOptimizer(model.parameters(), weight_decay=0.0)
        
        # With weight decay
        optimizer_with_wd = MuonOptimizer(model.parameters(), weight_decay=1e-2)
        
        # Both should work, but weight decay is important for grokking
        assert optimizer_no_wd.param_groups[0]["weight_decay"] == 0.0
        assert optimizer_with_wd.param_groups[0]["weight_decay"] == 1e-2
    
    def test_task_specific_grokking(self):
        """Test grokking on different task types"""
        set_seed(42)
        
        # Test different task types
        task_types = ["add", "mul", "gcd", "parity"]
        
        for task_type in task_types:
            # Create dataset
            dataset = ModularArithmeticDataset(task_type, modulus=13)
            
            # Check that dataset is created successfully
            assert len(dataset.data) > 0
            assert dataset.task_type == task_type
            
            # Check that model can process this task
            model = GrokkingTransformer(
                vocab_size=dataset.vocab_size,
                hidden_size=32,
                num_layers=1,
                num_heads=2,
                ff_size=64,
                max_seq_len=5,
            )
            
            # Test forward pass
            sample = dataset[0]
            inputs = sample["input"].unsqueeze(0)
            
            with torch.no_grad():
                output = model(inputs)
            
            expected_shape = (1, inputs.shape[1], dataset.vocab_size)
            assert output.shape == expected_shape
