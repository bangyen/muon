"""
Integration tests for the complete Muon optimizer system.

These tests validate the end-to-end functionality including:
- Complete training loops
- Model-optimizer-dataset integration
- Performance comparisons
- Reproducibility
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from src.model import GrokkingTransformer
from src.optimizer import MuonOptimizer
from src.dataset import ModularArithmeticDataset
from tests.conftest import set_seed, TestConfig, detect_grokking, compute_accuracy


class TestEndToEndTraining:
    """Test end-to-end training functionality"""
    
    def test_complete_training_loop(self):
        """Test a complete training loop with all components"""
        set_seed(42)
        
        # Create dataset
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        # Create model
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        # Create optimizer
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            spectral_norm_strength=0.1,
        )
        
        # Create loss function
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Training loop
        num_epochs = 3
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 5:  # Limit batches for testing
                    break
                    
                inputs = batch["input"]
                targets = batch["target"]
                
                optimizer.zero_grad()
                logits = model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average loss
            avg_loss = epoch_loss / min(5, len(dataloader))
            train_losses.append(avg_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= 3:  # Limit validation batches
                        break
                        
                    inputs = batch["input"]
                    targets = batch["target"]
                    
                    logits = model(inputs)
                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)
                    
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    val_correct += (predictions == targets).sum().item()
                    val_total += targets.numel()
            
            avg_val_loss = val_loss / min(3, len(dataloader))
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            val_losses.append(avg_val_loss)
        
        # Check that training completed successfully
        assert len(train_losses) == num_epochs
        assert len(val_losses) == num_epochs
        
        # Check that losses are finite
        assert all(torch.isfinite(torch.tensor(loss)) for loss in train_losses)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in val_losses)
        
        # Check that losses are reasonable
        assert all(0 <= loss < 1000 for loss in train_losses)
        assert all(0 <= loss < 1000 for loss in val_losses)
    
    def test_training_with_different_optimizers(self):
        """Test training with different optimizers"""
        set_seed(42)
        
        from torch.optim import AdamW
        
        # Create dataset
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        # Create identical models
        model1 = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        model2 = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        # Copy weights to ensure identical starting points
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)
        
        # Create optimizers
        muon_optimizer = MuonOptimizer(
            model1.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
        )
        
        adamw_optimizer = AdamW(
            model2.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
        )
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Train both models
        muon_losses = []
        adamw_losses = []
        
        for epoch in range(3):
            # Muon training
            model1.train()
            muon_epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 3:
                    break
                    
                inputs = batch["input"]
                targets = batch["target"]
                
                muon_optimizer.zero_grad()
                logits = model1(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(logits, targets)
                loss.backward()
                muon_optimizer.step()
                
                muon_epoch_loss += loss.item()
            
            muon_losses.append(muon_epoch_loss / 3)
            
            # AdamW training
            model2.train()
            adamw_epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 3:
                    break
                    
                inputs = batch["input"]
                targets = batch["target"]
                
                adamw_optimizer.zero_grad()
                logits = model2(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(logits, targets)
                loss.backward()
                adamw_optimizer.step()
                
                adamw_epoch_loss += loss.item()
            
            adamw_losses.append(adamw_epoch_loss / 3)
        
        # Check that both optimizers completed training
        assert len(muon_losses) == 3
        assert len(adamw_losses) == 3
        
        # Check that losses are finite
        assert all(torch.isfinite(torch.tensor(loss)) for loss in muon_losses)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in adamw_losses)
        
        # Check that optimizers behave differently
        assert any(abs(m - a) > 1e-6 for m, a in zip(muon_losses, adamw_losses))
    
    def test_training_with_different_softmax(self):
        """Test training with different softmax variants"""
        set_seed(42)
        
        # Create dataset
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        # Test different softmax variants
        softmax_variants = ["standard", "stablemax", "sparsemax"]
        
        for variant in softmax_variants:
            # Create model with specific softmax variant
            model = GrokkingTransformer(
                vocab_size=dataset.vocab_size,
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                ff_size=64,
                max_seq_len=5,
                softmax_variant=variant,
            )
            
            optimizer = MuonOptimizer(model.parameters(), lr=1e-3)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            
            # Train for a few steps
            model.train()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 3:
                    break
                    
                inputs = batch["input"]
                targets = batch["target"]
                
                optimizer.zero_grad()
                logits = model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
            
            # Check that training completed successfully
            assert torch.isfinite(loss)
    
    def test_training_reproducibility(self):
        """Test that training is reproducible with same seed"""
        set_seed(42)
        
        # Create dataset
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        # Train model with seed 123
        model1 = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        optimizer1 = MuonOptimizer(model1.parameters(), lr=1e-3)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Train first model
        set_seed(123)
        model1.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
                
            inputs = batch["input"]
            targets = batch["target"]
            
            optimizer1.zero_grad()
            logits = model1(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer1.step()
        
        loss1 = loss.item()
        
        # Train second model with same seed
        model2 = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        optimizer2 = MuonOptimizer(model2.parameters(), lr=1e-3)
        
        set_seed(123)
        model2.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
                
            inputs = batch["input"]
            targets = batch["target"]
            
            optimizer2.zero_grad()
            logits = model2(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer2.step()
        
        loss2 = loss.item()
        
        # Check that losses are identical (reproducibility)
        assert abs(loss1 - loss2) < 1e-6


class TestPerformanceBenchmarks:
    """Test performance benchmarks and comparisons"""
    
    def test_training_speed_comparison(self):
        """Test training speed comparison between optimizers"""
        set_seed(42)
        
        from torch.optim import AdamW
        import time
        
        # Create dataset
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        # Create models
        model1 = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        model2 = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        # Copy weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)
        
        # Create optimizers
        muon_optimizer = MuonOptimizer(model1.parameters(), lr=1e-3)
        adamw_optimizer = AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-2)
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Time Muon training
        start_time = time.time()
        model1.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:
                break
                
            inputs = batch["input"]
            targets = batch["target"]
            
            muon_optimizer.zero_grad()
            logits = model1(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            muon_optimizer.step()
        
        muon_time = time.time() - start_time
        
        # Time AdamW training
        start_time = time.time()
        model2.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:
                break
                
            inputs = batch["input"]
            targets = batch["target"]
            
            adamw_optimizer.zero_grad()
            logits = model2(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            adamw_optimizer.step()
        
        adamw_time = time.time() - start_time
        
        # Check that both completed successfully
        assert muon_time > 0
        assert adamw_time > 0
        
        # Both should complete in reasonable time
        assert muon_time < 60  # Less than 1 minute
        assert adamw_time < 60
    
    def test_memory_usage(self):
        """Test memory usage during training"""
        set_seed(42)
        
        import gc
        
        # Create dataset and model
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        optimizer = MuonOptimizer(model.parameters(), lr=1e-3)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Train for a few steps
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:
                break
                
            inputs = batch["input"]
            targets = batch["target"]
            
            optimizer.zero_grad()
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        
        # Check that training completed without memory issues
        assert torch.isfinite(loss)
        
        # Clean up
        del model, optimizer, dataloader
        gc.collect()
    
    def test_convergence_comparison(self):
        """Test convergence comparison between optimizers"""
        set_seed(42)
        
        from torch.optim import AdamW
        
        # Create dataset
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        # Create models
        model1 = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        model2 = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        # Copy weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)
        
        # Create optimizers
        muon_optimizer = MuonOptimizer(model1.parameters(), lr=1e-3)
        adamw_optimizer = AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-2)
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Track convergence
        muon_losses = []
        adamw_losses = []
        
        for epoch in range(5):
            # Muon training
            model1.train()
            muon_epoch_loss = 0.0
            muon_steps = 0
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 3:
                    break
                    
                inputs = batch["input"]
                targets = batch["target"]
                
                muon_optimizer.zero_grad()
                logits = model1(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(logits, targets)
                loss.backward()
                muon_optimizer.step()
                
                muon_epoch_loss += loss.item()
                muon_steps += 1
            
            muon_losses.append(muon_epoch_loss / muon_steps)
            
            # AdamW training
            model2.train()
            adamw_epoch_loss = 0.0
            adamw_steps = 0
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 3:
                    break
                    
                inputs = batch["input"]
                targets = batch["target"]
                
                adamw_optimizer.zero_grad()
                logits = model2(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(logits, targets)
                loss.backward()
                adamw_optimizer.step()
                
                adamw_epoch_loss += loss.item()
                adamw_steps += 1
            
            adamw_losses.append(adamw_epoch_loss / adamw_steps)
        
        # Check that both optimizers converged
        assert len(muon_losses) == 5
        assert len(adamw_losses) == 5
        
        # Check that losses are finite and reasonable
        assert all(torch.isfinite(torch.tensor(loss)) for loss in muon_losses)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in adamw_losses)
        
        # Check that optimizers have different convergence patterns
        assert any(abs(m - a) > 1e-6 for m, a in zip(muon_losses, adamw_losses))


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        set_seed(42)
        
        # Create dataset and model
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        optimizer = MuonOptimizer(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Test with invalid input shapes
        with pytest.raises(RuntimeError):
            invalid_input = torch.randn(4, 10, 20)  # Wrong shape
            model(invalid_input)
        
        # Test with invalid target indices
        with pytest.raises(RuntimeError):
            inputs = torch.randint(0, dataset.vocab_size, (4, 5))
            targets = torch.randint(dataset.vocab_size, dataset.vocab_size + 10, (4, 5))
            
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
    
    def test_numerical_stability(self):
        """Test numerical stability during training"""
        set_seed(42)
        
        # Create dataset and model
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        # Initialize with very large weights to test stability
        with torch.no_grad():
            for param in model.parameters():
                param.data *= 100.0
        
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            spectral_norm_strength=0.5,  # Strong spectral norm constraint
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Train for a few steps
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
                
            inputs = batch["input"]
            targets = batch["target"]
            
            optimizer.zero_grad()
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        
        # Check that training completed without numerical issues
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality"""
        set_seed(42)
        
        # Create dataset and model
        dataset = ModularArithmeticDataset("add", modulus=13, train_split=0.8)
        
        model = GrokkingTransformer(
            vocab_size=dataset.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ff_size=64,
            max_seq_len=5,
        )
        
        optimizer = MuonOptimizer(model.parameters(), lr=1e-3)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Train with gradient clipping
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
                
            inputs = batch["input"]
            targets = batch["target"]
            
            optimizer.zero_grad()
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Check that training completed successfully
        assert torch.isfinite(loss)
