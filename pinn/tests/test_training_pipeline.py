"""Unit tests for Training Pipeline Service."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pinn.training.training_pipeline import TrainingPipelineService


class TestLossComputation:
    """Test multi-component loss computation and tracking."""

    def test_extract_loss_components_from_model_state(self):
        """Test extraction of individual loss components from DeepXDE model."""
        service = TrainingPipelineService()

        # Mock DeepXDE model with train_state
        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.loss_train = np.array([0.5, 0.3, 0.2])  # [L_data, L_pde, L_bc]

        loss_components = service.extract_loss_components(mock_model)

        assert "L_data" in loss_components
        assert "L_pde" in loss_components
        assert "L_bc" in loss_components
        assert "total_loss" in loss_components

        assert loss_components["L_data"] == pytest.approx(0.5)
        assert loss_components["L_pde"] == pytest.approx(0.3)
        assert loss_components["L_bc"] == pytest.approx(0.2)
        assert loss_components["total_loss"] == pytest.approx(1.0)

    def test_compute_total_loss_with_weights(self):
        """Test total loss computation: w_data*L_data + w_pde*L_pde + w_bc*L_bc."""
        service = TrainingPipelineService()

        loss_components = {
            "L_data": 0.5,
            "L_pde": 0.3,
            "L_bc": 0.2
        }
        weights = {
            "data": 2.0,
            "pde": 1.0,
            "bc": 1.0
        }

        total_loss = service.compute_weighted_loss(loss_components, weights)

        # Expected: 2.0*0.5 + 1.0*0.3 + 1.0*0.2 = 1.0 + 0.3 + 0.2 = 1.5
        assert total_loss == pytest.approx(1.5)

    def test_validate_loss_weights_format(self):
        """Test loss weights validation."""
        service = TrainingPipelineService()

        # Valid weights
        valid_weights = {"data": 1.0, "pde": 1.0, "bc": 1.0}
        assert service.validate_loss_weights(valid_weights) is True

        # Missing keys
        invalid_weights = {"data": 1.0, "pde": 1.0}
        assert service.validate_loss_weights(invalid_weights) is False

        # Negative weights
        negative_weights = {"data": -1.0, "pde": 1.0, "bc": 1.0}
        assert service.validate_loss_weights(negative_weights) is False


class TestGPUDetection:
    """Test GPU device detection and management."""

    def test_detect_device_with_cuda_available(self):
        """Test device detection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            service = TrainingPipelineService()
            device = service._detect_device()

            assert device.type == "cuda"

    def test_detect_device_without_cuda(self):
        """Test device detection when CUDA is unavailable (fallback to CPU)."""
        with patch('torch.cuda.is_available', return_value=False):
            service = TrainingPipelineService()
            device = service._detect_device()

            assert device.type == "cpu"

    def test_log_gpu_memory_when_cuda_available(self):
        """Test GPU memory logging when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=1_000_000):
                with patch('torch.cuda.memory_reserved', return_value=2_000_000):
                    service = TrainingPipelineService()
                    memory_info = service.log_gpu_memory()

                    assert "allocated_mb" in memory_info
                    assert "reserved_mb" in memory_info
                    assert memory_info["allocated_mb"] == pytest.approx(1.0, rel=0.01)
                    assert memory_info["reserved_mb"] == pytest.approx(2.0, rel=0.01)

    def test_log_gpu_memory_when_cuda_unavailable(self):
        """Test GPU memory logging returns empty dict when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            service = TrainingPipelineService()
            memory_info = service.log_gpu_memory()

            assert memory_info == {}


class TestTrainingPipeline:
    """Test complete training pipeline orchestration."""

    def test_train_pipeline_initialization(self):
        """Test training pipeline can be created and initialized."""
        service = TrainingPipelineService()

        assert service.device is not None
        assert hasattr(service, 'train')

    def test_train_method_signature(self):
        """Test train method has correct signature."""
        from pathlib import Path
        from unittest.mock import Mock
        import inspect

        service = TrainingPipelineService()

        # Check method exists and signature
        assert hasattr(service, 'train')
        sig = inspect.signature(service.train)

        # Should accept model, config, and output_dir
        assert 'model' in sig.parameters
        assert 'config' in sig.parameters
        assert 'output_dir' in sig.parameters

    def test_register_callbacks(self):
        """Test callback registration functionality."""
        service = TrainingPipelineService()
        from unittest.mock import Mock

        # Mock callbacks
        callback1 = Mock()
        callback2 = Mock()

        callbacks = service.register_callbacks(
            callbacks=[callback1, callback2],
            model=Mock()
        )

        assert len(callbacks) == 2
        assert callback1 in callbacks
        assert callback2 in callbacks
