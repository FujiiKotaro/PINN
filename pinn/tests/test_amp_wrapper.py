"""Unit tests for AMP Wrapper Service."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pinn.training.amp_wrapper import AMPWrapperService


class TestAMPWrapper:
    """Test automatic mixed precision wrapper functionality."""

    def test_amp_enabled_when_cuda_available_and_flag_true(self):
        """Test AMP is enabled when CUDA available and config flag is True."""
        with patch('torch.cuda.is_available', return_value=True):
            wrapper = AMPWrapperService(enabled=True)
            assert wrapper.enabled is True
            assert wrapper.scaler is not None

    def test_amp_disabled_when_cuda_unavailable(self):
        """Test AMP is disabled when CUDA unavailable even if flag True."""
        with patch('torch.cuda.is_available', return_value=False):
            wrapper = AMPWrapperService(enabled=True)
            assert wrapper.enabled is False
            assert wrapper.scaler is None

    def test_amp_disabled_when_flag_false(self):
        """Test AMP is disabled when config flag is False."""
        with patch('torch.cuda.is_available', return_value=True):
            wrapper = AMPWrapperService(enabled=False)
            assert wrapper.enabled is False
            assert wrapper.scaler is None

    def test_autocast_context_manager_with_amp_enabled(self):
        """Test autocast context manager when AMP enabled."""
        with patch('torch.cuda.is_available', return_value=True):
            wrapper = AMPWrapperService(enabled=True)

            # Verify context manager can be used
            with wrapper.autocast():
                pass  # Context should work without errors

    def test_autocast_context_manager_with_amp_disabled(self):
        """Test autocast context manager is no-op when AMP disabled."""
        with patch('torch.cuda.is_available', return_value=False):
            wrapper = AMPWrapperService(enabled=True)

            # Should still work as no-op
            with wrapper.autocast():
                pass

    def test_scale_and_step_with_amp_enabled(self):
        """Test gradient scaling when AMP enabled."""
        with patch('torch.cuda.is_available', return_value=True):
            wrapper = AMPWrapperService(enabled=True)

            # Mock loss, optimizer, and scaler
            loss = Mock(spec=torch.Tensor)
            optimizer = Mock(spec=torch.optim.Optimizer)

            # Mock GradScaler methods
            wrapper.scaler = Mock()
            wrapper.scaler.scale = Mock(return_value=Mock(backward=Mock()))

            wrapper.scale_and_step(loss, optimizer)

            # Verify scaler was used
            wrapper.scaler.scale.assert_called_once_with(loss)
            wrapper.scaler.step.assert_called_once_with(optimizer)
            wrapper.scaler.update.assert_called_once()

    def test_scale_and_step_with_amp_disabled(self):
        """Test gradient scaling falls back to standard backprop when AMP disabled."""
        with patch('torch.cuda.is_available', return_value=False):
            wrapper = AMPWrapperService(enabled=True)

            # Create real tensors for testing CPU path
            x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            loss = x.sum()
            optimizer = torch.optim.SGD([x], lr=0.1)

            # Should execute standard backprop without errors
            wrapper.scale_and_step(loss, optimizer)

            # Verify gradient computed
            assert x.grad is not None

    def test_log_gpu_memory_with_cuda_available(self):
        """Test GPU memory logging when CUDA available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=5_000_000):
                with patch('torch.cuda.memory_reserved', return_value=10_000_000):
                    wrapper = AMPWrapperService(enabled=True)
                    memory_info = wrapper.log_gpu_memory()

                    assert memory_info["allocated_mb"] == pytest.approx(5.0, rel=0.01)
                    assert memory_info["reserved_mb"] == pytest.approx(10.0, rel=0.01)

    def test_log_gpu_memory_without_cuda(self):
        """Test GPU memory logging returns empty dict without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            wrapper = AMPWrapperService(enabled=True)
            memory_info = wrapper.log_gpu_memory()

            assert memory_info == {}
