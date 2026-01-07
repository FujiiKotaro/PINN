"""
Adaptive loss weighting mechanism for PINN training.
"""

import numpy as np
from typing import Dict, Optional


class AdaptiveLossWeighting:
    """
    Adaptive loss weighting using gradient statistics.

    This class implements various adaptive weighting strategies to balance
    different loss components during PINN training.
    """

    def __init__(
        self,
        method: str = "grad_norm",
        alpha: float = 0.9,
        update_interval: int = 100,
        initial_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize adaptive loss weighting.

        Args:
            method: Weighting method ("grad_norm", "softmax", "relobralo")
            alpha: Exponential moving average coefficient
            update_interval: Update weights every N iterations
            initial_weights: Initial loss weights
        """
        self.method = method
        self.alpha = alpha
        self.update_interval = update_interval
        self.weights = initial_weights or {}
        self.loss_history = {}
        self.iteration = 0

    def update_weights(self, loss_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Update loss weights based on current losses.

        Args:
            loss_dict: Dictionary of loss components

        Returns:
            Updated weight dictionary
        """
        self.iteration += 1

        # Store loss history
        for key, value in loss_dict.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value)

        # Update weights at specified interval
        if self.iteration % self.update_interval == 0:
            if self.method == "grad_norm":
                self._update_grad_norm()
            elif self.method == "softmax":
                self._update_softmax()
            elif self.method == "relobralo":
                self._update_relobralo()

        return self.weights

    def _update_grad_norm(self):
        """Update weights based on gradient norm balancing."""
        if len(self.loss_history) < 2:
            return

        # Calculate mean losses over recent history
        window_size = min(self.update_interval, len(next(iter(self.loss_history.values()))))
        mean_losses = {}

        for key, history in self.loss_history.items():
            if len(history) >= window_size:
                mean_losses[key] = np.mean(history[-window_size:])

        if not mean_losses:
            return

        # Calculate new weights inversely proportional to loss magnitude
        # Higher loss -> lower weight (to prevent domination)
        total_inverse = sum(1.0 / (loss + 1e-8) for loss in mean_losses.values())

        for key, loss in mean_losses.items():
            new_weight = (1.0 / (loss + 1e-8)) / total_inverse * len(mean_losses)

            # Smooth update using exponential moving average
            if key in self.weights:
                self.weights[key] = self.alpha * self.weights[key] + (1 - self.alpha) * new_weight
            else:
                self.weights[key] = new_weight

    def _update_softmax(self):
        """Update weights using softmax temperature scaling."""
        if len(self.loss_history) < 2:
            return

        window_size = min(self.update_interval, len(next(iter(self.loss_history.values()))))
        mean_losses = {}

        for key, history in self.loss_history.items():
            if len(history) >= window_size:
                mean_losses[key] = np.mean(history[-window_size:])

        if not mean_losses:
            return

        # Softmax with temperature
        temperature = 1.0
        loss_array = np.array(list(mean_losses.values()))
        exp_losses = np.exp(-loss_array / temperature)
        softmax_weights = exp_losses / np.sum(exp_losses)

        for i, key in enumerate(mean_losses.keys()):
            new_weight = softmax_weights[i] * len(mean_losses)

            if key in self.weights:
                self.weights[key] = self.alpha * self.weights[key] + (1 - self.alpha) * new_weight
            else:
                self.weights[key] = new_weight

    def _update_relobralo(self):
        """
        ReLoBRaLo: Relative Loss Balancing with Random Lookback.

        This method balances losses by looking at their relative magnitudes
        over a random lookback window.
        """
        if len(self.loss_history) < 2:
            return

        window_size = min(self.update_interval, len(next(iter(self.loss_history.values()))))

        # Random lookback within window
        lookback = np.random.randint(max(1, window_size // 2), window_size + 1)

        mean_losses = {}
        for key, history in self.loss_history.items():
            if len(history) >= lookback:
                mean_losses[key] = np.mean(history[-lookback:])

        if not mean_losses:
            return

        # Calculate weights to balance losses
        max_loss = max(mean_losses.values())

        for key, loss in mean_losses.items():
            target_weight = max_loss / (loss + 1e-8)

            if key in self.weights:
                self.weights[key] = self.alpha * self.weights[key] + (1 - self.alpha) * target_weight
            else:
                self.weights[key] = target_weight

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()

    def reset(self):
        """Reset the weighting mechanism."""
        self.loss_history = {}
        self.iteration = 0
