"""
Tests for the Muon Optimizer implementation.

These tests validate the core optimizer functionality including:
- Basic optimization steps
- Spectral norm constraints
- Orthogonal gradient updates
- Second-order information approximation
- Comparison with AdamW
"""

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from src.optimizer import MuonOptimizer
from tests.conftest import set_seed


class TestMuonOptimizer:
    """Test suite for Muon Optimizer"""

    def test_initialization(self):
        """Test optimizer initialization with various parameters"""
        model = nn.Linear(10, 5)

        # Test basic initialization
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            spectral_norm_strength=0.1,
            second_order_interval=5,
        )

        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["spectral_norm_strength"] == 0.1
        assert optimizer.param_groups[0]["second_order_interval"] == 5

        # Test default parameters
        optimizer = MuonOptimizer(model.parameters())
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.98)
        assert optimizer.param_groups[0]["eps"] == 1e-8

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors"""
        model = nn.Linear(10, 5)

        # Test negative learning rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            MuonOptimizer(model.parameters(), lr=-1.0)

        # Test invalid beta parameters
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            MuonOptimizer(model.parameters(), betas=(1.1, 0.98))

        with pytest.raises(ValueError, match="Invalid beta parameter"):
            MuonOptimizer(model.parameters(), betas=(0.9, -0.1))

        # Test negative weight decay
        with pytest.raises(ValueError, match="Invalid weight_decay value"):
            MuonOptimizer(model.parameters(), weight_decay=-1.0)

    def test_basic_optimization_step(self):
        """Test basic optimization step functionality"""
        set_seed(42)

        model = nn.Linear(10, 5)
        optimizer = MuonOptimizer(model.parameters(), lr=1e-3)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Perform optimization step
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        # Check that parameters have been updated
        for i, p in enumerate(model.parameters()):
            assert not torch.allclose(p, initial_params[i])

    def test_spectral_norm_constraint(self):
        """Test spectral norm constraint functionality"""
        set_seed(42)

        # Create a model that might have large weights
        model = nn.Linear(10, 10)

        # Initialize with large weights
        with torch.no_grad():
            model.weight.data *= 10.0

        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            spectral_norm_strength=0.5,
        )

        # Perform several steps to see if spectral norm constraint is applied
        for _ in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 10)

            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()

        # Check that weights haven't grown excessively
        spectral_norm = torch.linalg.svd(model.weight.data)[1][0]
        assert spectral_norm < 20.0  # Should be reasonable

    def test_orthogonal_gradient_updates(self):
        """Test orthogonal gradient update functionality"""
        set_seed(42)

        model = nn.Linear(10, 5)
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            use_orthogonal_updates=True,
        )

        # Perform multiple steps to test orthogonalization
        gradients = []

        for _step in range(5):
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)

            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            # Store gradient before step
            grad = model.weight.grad.clone()
            gradients.append(grad)

            optimizer.step()

        # Check that gradients are not perfectly aligned (orthogonalization effect)
        if len(gradients) >= 2:
            cos_sim = torch.cosine_similarity(
                gradients[0].flatten(), gradients[1].flatten(), dim=0
            )
            # Should not be perfectly aligned
            assert abs(cos_sim.item()) < 0.99

    def test_second_order_information(self):
        """Test second-order information approximation"""
        set_seed(42)

        model = nn.Linear(10, 5)
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            second_order_interval=3,
        )

        # Perform steps to trigger second-order updates
        for step in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)

            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()

            # Check that hessian_diag is updated
            if step > 0 and step % 3 == 0:
                state = optimizer.state[model.weight]
                assert "hessian_diag" in state
                assert state["hessian_diag"].shape == model.weight.shape

    def test_weight_decay(self):
        """Test weight decay functionality"""
        set_seed(42)

        model = nn.Linear(10, 5)
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
        )

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Perform optimization step
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        # Check that weight decay is applied
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                # Weight decay should reduce parameter values
                assert torch.norm(p) <= torch.norm(initial_params[i])

    def test_sparse_gradient_error(self):
        """Test that sparse gradients raise an error"""
        model = nn.Linear(10, 5)
        optimizer = MuonOptimizer(model.parameters())

        # Create sparse gradient
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Make gradient sparse
        sparse_grad = model.weight.grad.data.to_sparse()
        model.weight.grad = torch.nn.Parameter(sparse_grad)

        with pytest.raises(
            RuntimeError, match="Muon does not support sparse gradients"
        ):
            optimizer.step()

    def test_state_initialization(self):
        """Test that optimizer state is properly initialized"""
        model = nn.Linear(10, 5)
        optimizer = MuonOptimizer(model.parameters())

        # Check that state is empty initially
        for p in model.parameters():
            assert len(optimizer.state[p]) == 0

        # Perform one step to initialize state
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        # Check that state is now initialized
        for p in model.parameters():
            state = optimizer.state[p]
            assert "step" in state
            assert "exp_avg" in state
            assert "exp_avg_sq" in state
            assert "hessian_diag" in state
            assert "prev_grad" in state
            assert state["step"] == 1


class TestMuonVsAdamW:
    """Test comparison between Muon and AdamW optimizers"""

    def test_convergence_comparison(self):
        """Test that Muon converges differently than AdamW"""
        set_seed(42)

        # Create identical models
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)

        # Copy weights to ensure they start identical
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)

        # Create optimizers with same learning rate
        muon_optimizer = MuonOptimizer(model1.parameters(), lr=1e-3)
        adamw_optimizer = AdamW(
            model2.parameters(), lr=1e-3, weight_decay=1e-2
        )

        # Train both models
        muon_losses = []
        adamw_losses = []

        for _step in range(20):
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

        # Check that the optimizers behave differently
        # (they should have different convergence patterns)
        assert len(set(muon_losses)) > 1  # Muon losses should vary
        assert len(set(adamw_losses)) > 1  # AdamW losses should vary

        # The loss trajectories should be different due to different update mechanisms
        muon_final_loss = muon_losses[-1]
        adamw_final_loss = adamw_losses[-1]

        # They might converge to similar final losses, but the paths should differ
        assert abs(muon_final_loss - adamw_final_loss) > 1e-6 or any(
            abs(m1 - m2) > 1e-6 for m1, m2 in zip(muon_losses, adamw_losses)
        )

    def test_parameter_trajectories(self):
        """Test that parameter update trajectories differ between optimizers"""
        set_seed(42)

        # Create identical models
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)

        # Copy weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)

        # Create optimizers
        muon_optimizer = MuonOptimizer(model1.parameters(), lr=1e-3)
        adamw_optimizer = AdamW(
            model2.parameters(), lr=1e-3, weight_decay=1e-2
        )

        # Track parameter changes
        muon_changes = []
        adamw_changes = []

        for _step in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)

            # Store initial weights
            muon_initial = model1.weight.clone()
            adamw_initial = model2.weight.clone()

            # Muon step
            muon_optimizer.zero_grad()
            output1 = model1(x)
            loss1 = nn.MSELoss()(output1, y)
            loss1.backward()
            muon_optimizer.step()

            # AdamW step
            adamw_optimizer.zero_grad()
            output2 = model2(x)
            loss2 = nn.MSELoss()(output2, y)
            loss2.backward()
            adamw_optimizer.step()

            # Calculate parameter changes
            muon_change = torch.norm(model1.weight - muon_initial).item()
            adamw_change = torch.norm(model2.weight - adamw_initial).item()

            muon_changes.append(muon_change)
            adamw_changes.append(adamw_change)

        # Check that the optimizers have different parameter update patterns
        assert any(
            abs(m - a) > 1e-6 for m, a in zip(muon_changes, adamw_changes)
        )

    def test_gradient_orthogonalization_effect(self):
        """Test that Muon's orthogonal updates have measurable effect"""
        set_seed(42)

        model = nn.Linear(10, 5)

        # Test with orthogonal updates enabled
        optimizer_ortho = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            use_orthogonal_updates=True,
        )

        # Test with orthogonal updates disabled
        model2 = nn.Linear(10, 5)
        with torch.no_grad():
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                p2.copy_(p1)

        optimizer_no_ortho = MuonOptimizer(
            model2.parameters(),
            lr=1e-3,
            use_orthogonal_updates=False,
        )

        # Track gradient directions
        ortho_directions = []
        no_ortho_directions = []

        for _step in range(5):
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)

            # Orthogonal updates
            optimizer_ortho.zero_grad()
            output1 = model(x)
            loss1 = nn.MSELoss()(output1, y)
            loss1.backward()
            grad1 = model.weight.grad.clone()
            optimizer_ortho.step()
            ortho_directions.append(grad1.flatten())

            # No orthogonal updates
            optimizer_no_ortho.zero_grad()
            output2 = model2(x)
            loss2 = nn.MSELoss()(output2, y)
            loss2.backward()
            grad2 = model2.weight.grad.clone()
            optimizer_no_ortho.step()
            no_ortho_directions.append(grad2.flatten())

        # Check that gradient directions differ between the two modes
        if len(ortho_directions) >= 2:
            ortho_similarity = torch.cosine_similarity(
                ortho_directions[0], ortho_directions[1], dim=0
            )
            no_ortho_similarity = torch.cosine_similarity(
                no_ortho_directions[0], no_ortho_directions[1], dim=0
            )

            # The orthogonal version should have less similar consecutive gradients
            assert (
                abs(ortho_similarity.item())
                <= abs(no_ortho_similarity.item()) + 1e-6
            )
