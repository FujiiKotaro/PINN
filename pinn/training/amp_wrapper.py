"""AMP Wrapper Service for GPU-accelerated mixed precision training."""

from contextlib import contextmanager

import torch


class AMPWrapperService:
    """Enable PyTorch automatic mixed precision for memory optimization and speedup."""

    def __init__(self, enabled: bool = True):
        """Initialize AMP wrapper.

        Args:
            enabled: Whether to enable AMP (only works with CUDA available)
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.enabled else None

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision forward pass.

        Yields:
            Context with AMP enabled if CUDA available, otherwise no-op
        """
        if self.enabled:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield  # No-op for CPU

    def scale_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> None:
        """Scale loss, backward pass, optimizer step with gradient scaling.

        Args:
            loss: Loss tensor to backpropagate
            optimizer: Optimizer instance
        """
        if self.enabled:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

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
