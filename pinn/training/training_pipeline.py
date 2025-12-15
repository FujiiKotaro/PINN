"""Training Pipeline Service for GPU-accelerated PINN training."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Any, List


class TrainingPipelineService:
    """Orchestrate PINN training with GPU acceleration and monitoring."""

    def __init__(self):
        """Initialize training pipeline service."""
        self.device = self._detect_device()

    def train(
        self,
        model: Any,
        config: Any,
        output_dir: Path,
        callbacks: Optional[List[Any]] = None
    ) -> tuple[Any, dict[str, list[float]]]:
        """Execute PINN training with monitoring.

        Args:
            model: Compiled DeepXDE model
            config: Training configuration (TrainingConfig object)
            output_dir: Directory for checkpoints and logs
            callbacks: Optional list of callbacks to register

        Returns:
            tuple[Any, dict[str, list[float]]]: Tuple of (trained_model, training_history)

        Training history dict contains:
            - "total_loss": List of total loss values per epoch
            - "L_data": Data fitting loss component
            - "L_pde": PDE residual loss component
            - "L_bc": Boundary condition loss component
            - "L2_error": Validation error vs. analytical solution (if ValidationCallback used)
        """
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Register callbacks if provided
        if callbacks:
            registered_callbacks = self.register_callbacks(callbacks, model)
        else:
            registered_callbacks = []

        # Train model with DeepXDE
        # Note: DeepXDE's model.train() handles the training loop internally
        # and calls callbacks via the internal callback system
        model.train(
            epochs=config.epochs if hasattr(config, 'epochs') else 10000,
            callbacks=registered_callbacks
        )

        # Extract training history from callbacks
        training_history = {}
        for callback in registered_callbacks:
            if hasattr(callback, 'history'):
                training_history.update(callback.history)
            if hasattr(callback, 'errors'):
                training_history['L2_errors'] = callback.errors
            if hasattr(callback, 'relative_errors'):
                training_history['relative_errors'] = callback.relative_errors

        return model, training_history

    def register_callbacks(
        self,
        callbacks: List[Any],
        model: Any
    ) -> List[Any]:
        """Register callbacks with model.

        Args:
            callbacks: List of callback instances
            model: DeepXDE Model instance

        Returns:
            List[Any]: List of registered callbacks
        """
        # Assign model to each callback
        for callback in callbacks:
            callback.model = model

        return callbacks

    def extract_loss_components(self, model: Any) -> dict[str, float]:
        """Extract individual loss components from DeepXDE model state.

        Args:
            model: DeepXDE Model instance with train_state attribute

        Returns:
            dict[str, float]: Dictionary with 'L_data', 'L_pde', 'L_bc', 'total_loss'
        """
        loss_train = model.train_state.loss_train

        # DeepXDE stores losses as: [L_data, L_pde, L_bc, ...]
        return {
            "L_data": float(loss_train[0]),
            "L_pde": float(loss_train[1]),
            "L_bc": float(loss_train[2]),
            "total_loss": float(np.sum(loss_train))
        }

    def compute_weighted_loss(
        self,
        loss_components: dict[str, float],
        weights: dict[str, float]
    ) -> float:
        """Compute weighted total loss: w_data*L_data + w_pde*L_pde + w_bc*L_bc.

        Args:
            loss_components: Dictionary with 'L_data', 'L_pde', 'L_bc'
            weights: Dictionary with 'data', 'pde', 'bc' weights

        Returns:
            float: Weighted total loss
        """
        return (
            weights["data"] * loss_components["L_data"] +
            weights["pde"] * loss_components["L_pde"] +
            weights["bc"] * loss_components["L_bc"]
        )

    def validate_loss_weights(self, weights: dict[str, float]) -> bool:
        """Validate loss weights format and values.

        Args:
            weights: Dictionary with 'data', 'pde', 'bc' keys

        Returns:
            bool: True if weights are valid, False otherwise
        """
        required_keys = {"data", "pde", "bc"}

        # Check all required keys present
        if not required_keys.issubset(weights.keys()):
            return False

        # Check all weights are non-negative
        if any(w < 0 for w in weights.values()):
            return False

        return True

    def _detect_device(self) -> torch.device:
        """Detect CUDA availability and return appropriate device.

        Returns:
            torch.device: CUDA device if available, otherwise CPU

        Side Effects:
            Logs warning message if CUDA unavailable
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("WARNING: CUDA not available, falling back to CPU")
            return torch.device("cpu")

    def log_gpu_memory(self) -> dict[str, float]:
        """Log current GPU memory usage in MB.

        Returns:
            dict[str, float]: Dictionary with 'allocated_mb' and 'reserved_mb' keys
                             Empty dict if CUDA unavailable
        """
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1e6,
                "reserved_mb": torch.cuda.memory_reserved() / 1e6
            }
        return {}
