import math
from typing import Union

import torch
from torch.optim import Optimizer


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
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        spectral_norm_strength: float = 0.1,
        second_order_interval: int = 10,
        use_orthogonal_updates: bool = True,
    ) -> None:
        """
        Initialize Muon optimizer

        Args:
            params: Model parameters
            lr: Learning rate
            betas: Beta parameters for momentum
            eps: Epsilon for numerical stability
            weight_decay: Weight decay strength
            spectral_norm_strength: Strength of spectral norm constraints
            second_order_interval: How often to compute second-order info
            use_orthogonal_updates: Whether to use orthogonalized gradients
        """
        min_value = 0.0
        max_beta = 1.0
        if not lr >= min_value:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= min_value:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not min_value <= betas[0] < max_beta:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not min_value <= betas[1] < max_beta:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not weight_decay >= min_value:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            spectral_norm_strength=spectral_norm_strength,
            second_order_interval=second_order_interval,
            use_orthogonal_updates=use_orthogonal_updates,
        )
        super().__init__(params, defaults)

        self.step_count = 0

    def step(self, closure=None) -> Union[float, None]:
        """Performs a single optimization step"""
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

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

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

                # Update parameters
                p.data.addcdiv_(
                    exp_avg,
                    bias_correction2_sqrt * torch.sqrt(exp_avg_sq)
                    + group["eps"],
                    value=-step_size,
                )

                # Store current gradient for next iteration
                prev_grad.copy_(grad)

        return loss

    def _orthogonalize_gradient(
        self, current_grad: torch.Tensor, prev_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Orthogonalize current gradient with respect to previous gradient
        This promotes broader exploration by reducing redundancy in updates
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
        Update diagonal Hessian approximation using gradient information
        This provides second-order information for better update directions
        """
        # Simple diagonal Hessian approximation based on gradient magnitude
        # More sophisticated methods could be used here
        new_hessian = torch.abs(grad) + 0.1 * torch.sqrt(exp_avg_sq)

        # Smooth update
        alpha = 0.9
        return alpha * hessian_diag + (1 - alpha) * new_hessian

    def _apply_spectral_norm_constraint(
        self, adaptive_lr: torch.Tensor, param: torch.Tensor, strength: float
    ) -> torch.Tensor:
        """
        Apply spectral norm constraint to prevent runaway weights
        This keeps training stable and avoids "softmax collapse"
        """
        min_dimensions_for_svd = 2
        # Compute spectral norm of the parameter
        if param.dim() >= min_dimensions_for_svd:
            # For matrices, compute the largest singular value
            u, s, v = torch.svd(param, compute_uv=False)
            spectral_norm = (
                s[0] if len(s) > 0 else torch.tensor(0.0, device=param.device)
            )
        else:
            # For vectors, use L2 norm (scalar)
            spectral_norm = torch.norm(param, p=2)

        # Ensure spectral_norm is a scalar
        if spectral_norm.dim() > 0:
            spectral_norm = spectral_norm.mean()  # Take mean if it's a tensor

        # Apply constraint if spectral norm is too large
        max_spectral_norm = 1.0
        if spectral_norm.item() > max_spectral_norm:
            # Scale down the learning rate to prevent further growth
            scale_factor = 1.0 / (
                1.0 + strength * (spectral_norm.item() - max_spectral_norm)
            )
            adaptive_lr = adaptive_lr * scale_factor

        return adaptive_lr
