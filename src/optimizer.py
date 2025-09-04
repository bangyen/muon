import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from torch.optim import Optimizer


@dataclass
class OptimizerConfig:
    """Configuration for the MuonOptimizer"""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    spectral_norm_strength: float = 0.1
    second_order_interval: int = 10
    use_orthogonal_updates: bool = True


class MuonOptimizer(Optimizer):
    """
    Muon Optimizer implementation based on the paper "Muon Optimizer Accelerates Grokking"

    Key features:
    1. Spectral norm constraints to prevent runaway weights
    2. Orthogonalized gradient updates for broader exploration
    3. Second-order information approximation
    4. Layer-wise update scaling
    """

    def __init__(
        self,
        params: Union[torch.Tensor, list[torch.Tensor]],
        config: Optional[Union[OptimizerConfig, dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Muon optimizer

        Args:
            params: Model parameters
            config: Optimizer configuration object or dict
            **kwargs: Additional configuration parameters
        """
        # Handle both config object and individual parameters
        if isinstance(config, OptimizerConfig):
            optimizer_config = config
        elif isinstance(config, dict):
            optimizer_config = OptimizerConfig(**config)
        else:
            # Extract parameters from kwargs
            optimizer_config = OptimizerConfig(
                lr=kwargs.get("lr", 1e-3),
                betas=kwargs.get("betas", (0.9, 0.98)),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=kwargs.get("weight_decay", 1e-2),
                spectral_norm_strength=kwargs.get(
                    "spectral_norm_strength", 0.1
                ),
                second_order_interval=kwargs.get("second_order_interval", 10),
                use_orthogonal_updates=kwargs.get(
                    "use_orthogonal_updates", True
                ),
            )

        min_value = 0.0
        max_beta = 1.0
        if not optimizer_config.lr >= min_value:
            raise ValueError(f"Invalid learning rate: {optimizer_config.lr}")
        if not optimizer_config.eps >= min_value:
            raise ValueError(f"Invalid epsilon value: {optimizer_config.eps}")
        if not min_value <= optimizer_config.betas[0] < max_beta:
            raise ValueError(
                f"Invalid beta parameter at index 0: {optimizer_config.betas[0]}"
            )
        if not min_value <= optimizer_config.betas[1] < max_beta:
            raise ValueError(
                f"Invalid beta parameter at index 1: {optimizer_config.betas[1]}"
            )
        if not optimizer_config.weight_decay >= min_value:
            raise ValueError(
                f"Invalid weight_decay value: {optimizer_config.weight_decay}"
            )

        defaults = dict(
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            spectral_norm_strength=optimizer_config.spectral_norm_strength,
            second_order_interval=optimizer_config.second_order_interval,
            use_orthogonal_updates=optimizer_config.use_orthogonal_updates,
        )
        super().__init__(params, defaults)

        self.step_count = 0

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Performs a single optimization step

        Args:
            closure: Optional closure that reevaluates the model and returns the loss

        Returns:
            The loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Muon does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Second-order information approximation
                    state["hessian_diag"] = torch.ones_like(p.data)
                    # Previous gradient for orthogonalization
                    state["prev_grad"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                hessian_diag = state["hessian_diag"]
                prev_grad = state["prev_grad"]

                beta1, beta2 = group["betas"]
                weight_decay = group["weight_decay"]
                spectral_norm_strength = group["spectral_norm_strength"]
                second_order_interval = group["second_order_interval"]
                use_orthogonal_updates = group["use_orthogonal_updates"]

                state["step"] += 1

                # Apply weight decay (AdamW style)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * group["lr"])

                # Orthogonalize gradient updates if enabled
                if use_orthogonal_updates and state["step"] > 1:
                    grad = self._orthogonalize_gradient(grad, prev_grad)

                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Update second-order information periodically
                if state["step"] % second_order_interval == 0:
                    hessian_diag = self._update_hessian_diag(
                        grad, exp_avg_sq, hessian_diag
                    )

                # Compute bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Apply spectral norm constraints
                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                # Compute adaptive learning rate with second-order information
                adaptive_lr = step_size / (
                    bias_correction2_sqrt * torch.sqrt(exp_avg_sq)
                    + group["eps"]
                )
                adaptive_lr = adaptive_lr / (hessian_diag + group["eps"])

                # Apply spectral norm constraint
                if spectral_norm_strength > 0:
                    adaptive_lr = self._apply_spectral_norm_constraint(
                        adaptive_lr, p.data, spectral_norm_strength
                    )

                # Update parameters using adaptive learning rate (Muon style)
                p.data.add_(-adaptive_lr * exp_avg)

                # Store current gradient for next iteration
                prev_grad.copy_(grad)

        return loss

    def _orthogonalize_gradient(
        self, current_grad: torch.Tensor, prev_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Orthogonalize current gradient with respect to previous gradient.

        This promotes broader exploration by reducing redundancy in updates.

        Args:
            current_grad: Current gradient tensor
            prev_grad: Previous gradient tensor

        Returns:
            Orthogonalized gradient tensor
        """
        epsilon_threshold = 1e-8
        if torch.norm(prev_grad) < epsilon_threshold:
            return current_grad

        # Project current gradient onto the space orthogonal to previous gradient
        proj_coeff = torch.sum(current_grad * prev_grad) / torch.sum(
            prev_grad * prev_grad
        )
        return current_grad - proj_coeff * prev_grad

    def _update_hessian_diag(
        self,
        grad: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        hessian_diag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update diagonal Hessian approximation using gradient information.

        This provides second-order information for better update directions.

        Args:
            grad: Current gradient tensor
            exp_avg_sq: Exponential moving average of squared gradients
            hessian_diag: Current diagonal Hessian approximation

        Returns:
            Updated diagonal Hessian approximation
        """
        # Improved diagonal Hessian approximation based on paper's description
        # The paper mentions this helps "reach generalization faster with fewer steps"
        grad_magnitude = torch.abs(grad)
        exp_avg_sqrt = torch.sqrt(exp_avg_sq + 1e-8)

        # Combine gradient magnitude with exponential moving average
        new_hessian = grad_magnitude + 0.1 * exp_avg_sqrt

        # Smooth update with momentum
        alpha = 0.95  # Increased from 0.9 for more stability
        return alpha * hessian_diag + (1 - alpha) * new_hessian

    def _apply_spectral_norm_constraint(
        self, adaptive_lr: torch.Tensor, param: torch.Tensor, strength: float
    ) -> torch.Tensor:
        """
        Apply spectral norm constraint to prevent runaway weights.

        This keeps training stable and avoids "softmax collapse".

        Args:
            adaptive_lr: Adaptive learning rate tensor
            param: Parameter tensor to constrain
            strength: Strength of the spectral norm constraint

        Returns:
            Constrained adaptive learning rate tensor
        """
        min_dimensions_for_svd = 2
        # Compute spectral norm of the parameter
        if param.dim() >= min_dimensions_for_svd:
            # For matrices, compute the largest singular value
            try:
                u, s, v = torch.svd(param, compute_uv=False)
                spectral_norm = (
                    s[0]
                    if len(s) > 0
                    else torch.tensor(0.0, device=param.device)
                )
            except RuntimeError:
                # Fallback to L2 norm if SVD fails
                spectral_norm = torch.norm(param, p=2)
        else:
            # For vectors, use L2 norm (scalar)
            spectral_norm = torch.norm(param, p=2)

        # Ensure spectral_norm is a scalar
        if spectral_norm.dim() > 0:
            spectral_norm = spectral_norm.mean()  # Take mean if it's a tensor

        # Apply constraint if spectral norm is too large (paper mentions preventing "softmax collapse")
        max_spectral_norm = 2.0  # Increased from 1.0 to allow more flexibility
        if spectral_norm.item() > max_spectral_norm:
            # Scale down the learning rate to prevent further growth
            scale_factor = 1.0 / (
                1.0 + strength * (spectral_norm.item() - max_spectral_norm)
            )
            adaptive_lr = adaptive_lr * scale_factor

        return adaptive_lr
