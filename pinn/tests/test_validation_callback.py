"""Unit tests for Validation Callback."""

from unittest.mock import Mock, patch

import numpy as np

from pinn.training.callbacks import ValidationCallback
from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService
from pinn.validation.error_metrics import ErrorMetricsService


class TestValidationCallback:
    """Test validation callback with analytical solution comparison."""

    def test_initialization(self):
        """Test callback initialization."""
        analytical_solver = AnalyticalSolutionGeneratorService()
        error_metrics = ErrorMetricsService()

        callback = ValidationCallback(
            analytical_solver=analytical_solver,
            error_metrics=error_metrics,
            validation_interval=500,
            domain_config={"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
            wave_speed=1.0
        )

        assert callback.validation_interval == 500
        assert callback.analytical_solver is analytical_solver
        assert callback.error_metrics is error_metrics
        assert len(callback.errors) == 0
        assert len(callback.relative_errors) == 0

    def test_validation_at_correct_intervals(self):
        """Test that validation is computed at specified intervals."""
        analytical_solver = AnalyticalSolutionGeneratorService()
        error_metrics = ErrorMetricsService()

        callback = ValidationCallback(
            analytical_solver=analytical_solver,
            error_metrics=error_metrics,
            validation_interval=500,
            domain_config={"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
            wave_speed=1.0
        )

        # Mock model
        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 500

        # Mock model prediction
        x_test = np.linspace(0, 1, 10).reshape(-1, 1)
        t_test = np.ones((10, 1)) * 0.5
        test_points = np.hstack([x_test, t_test])

        mock_model.predict = Mock(return_value=np.random.rand(10, 1))

        callback.model = mock_model
        callback.test_points = test_points

        # Execute callback
        callback.on_epoch_end()

        # Should have computed one validation error
        assert len(callback.errors) == 1
        assert len(callback.relative_errors) == 1

    def test_no_validation_between_intervals(self):
        """Test that validation is not computed between intervals."""
        analytical_solver = AnalyticalSolutionGeneratorService()
        error_metrics = ErrorMetricsService()

        callback = ValidationCallback(
            analytical_solver=analytical_solver,
            error_metrics=error_metrics,
            validation_interval=500,
            domain_config={"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
            wave_speed=1.0
        )

        # Mock model at epoch 250 (not a multiple of 500)
        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 250

        callback.model = mock_model
        callback.on_epoch_end()

        # Should not have computed validation
        assert len(callback.errors) == 0

    def test_high_error_warning(self):
        """Test warning when relative error exceeds threshold."""
        analytical_solver = AnalyticalSolutionGeneratorService()
        error_metrics = ErrorMetricsService()

        callback = ValidationCallback(
            analytical_solver=analytical_solver,
            error_metrics=error_metrics,
            validation_interval=500,
            domain_config={"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
            wave_speed=1.0,
            error_threshold=0.05  # 5%
        )

        # Mock model
        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 500

        # Create test points
        x_test = np.linspace(0, 1, 10).reshape(-1, 1)
        t_test = np.ones((10, 1)) * 0.5
        test_points = np.hstack([x_test, t_test])

        # Mock prediction with high error (very different from analytical)
        mock_model.predict = Mock(return_value=np.ones((10, 1)) * 10.0)

        callback.model = mock_model
        callback.test_points = test_points

        # Should trigger warning (captured in stdout)
        with patch('builtins.print') as mock_print:
            callback.on_epoch_end()

            # Check if warning was printed
            assert mock_print.called
            warning_msg = str(mock_print.call_args)
            assert "WARNING" in warning_msg or "High" in warning_msg

    def test_multiple_validation_events(self):
        """Test multiple validation events accumulate error history."""
        analytical_solver = AnalyticalSolutionGeneratorService()
        error_metrics = ErrorMetricsService()

        callback = ValidationCallback(
            analytical_solver=analytical_solver,
            error_metrics=error_metrics,
            validation_interval=500,
            domain_config={"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
            wave_speed=1.0
        )

        mock_model = Mock()
        callback.model = mock_model

        x_test = np.linspace(0, 1, 10).reshape(-1, 1)
        t_test = np.ones((10, 1)) * 0.5
        test_points = np.hstack([x_test, t_test])
        callback.test_points = test_points

        # First validation at epoch 500
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 500
        mock_model.predict = Mock(return_value=np.random.rand(10, 1))
        callback.on_epoch_end()

        # Second validation at epoch 1000
        mock_model.train_state.epoch = 1000
        mock_model.predict = Mock(return_value=np.random.rand(10, 1))
        callback.on_epoch_end()

        # Should have two validation errors
        assert len(callback.errors) == 2
        assert len(callback.relative_errors) == 2
