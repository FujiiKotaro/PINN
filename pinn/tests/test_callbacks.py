"""Unit tests for Custom Training Callbacks."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import json
from pinn.training.callbacks import (
    LossLoggingCallback,
    CheckpointCallback,
    DivergenceDetectionCallback
)


class TestLossLoggingCallback:
    """Test loss logging callback functionality."""

    def test_initialization(self):
        """Test callback initialization with log interval."""
        callback = LossLoggingCallback(log_interval=100)

        assert callback.log_interval == 100
        assert "L_data" in callback.history
        assert "L_pde" in callback.history
        assert "L_bc" in callback.history
        assert "total_loss" in callback.history
        assert len(callback.history["L_data"]) == 0

    def test_logging_at_correct_intervals(self):
        """Test that losses are logged at specified intervals."""
        callback = LossLoggingCallback(log_interval=10)

        # Mock model with train_state
        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 10
        mock_model.train_state.loss_train = np.array([0.5, 0.3, 0.2])

        callback.model = mock_model
        callback.on_epoch_end()

        # Should log at epoch 10
        assert len(callback.history["L_data"]) == 1
        assert callback.history["L_data"][0] == pytest.approx(0.5)
        assert callback.history["L_pde"][0] == pytest.approx(0.3)
        assert callback.history["L_bc"][0] == pytest.approx(0.2)
        assert callback.history["total_loss"][0] == pytest.approx(1.0)

    def test_no_logging_between_intervals(self):
        """Test that losses are not logged between intervals."""
        callback = LossLoggingCallback(log_interval=10)

        # Mock model at epoch 5 (not a multiple of 10)
        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 5
        mock_model.train_state.loss_train = np.array([0.5, 0.3, 0.2])

        callback.model = mock_model
        callback.on_epoch_end()

        # Should not log
        assert len(callback.history["L_data"]) == 0

    def test_multiple_logging_events(self):
        """Test multiple logging events accumulate history."""
        callback = LossLoggingCallback(log_interval=10)
        mock_model = Mock()
        callback.model = mock_model

        # First log at epoch 10
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 10
        mock_model.train_state.loss_train = np.array([0.5, 0.3, 0.2])
        callback.on_epoch_end()

        # Second log at epoch 20
        mock_model.train_state.epoch = 20
        mock_model.train_state.loss_train = np.array([0.4, 0.25, 0.15])
        callback.on_epoch_end()

        assert len(callback.history["L_data"]) == 2
        assert callback.history["L_data"][1] == pytest.approx(0.4)
        assert callback.history["total_loss"][1] == pytest.approx(0.8)


class TestCheckpointCallback:
    """Test checkpoint saving callback functionality."""

    def test_initialization(self):
        """Test callback initialization with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = CheckpointCallback(output_dir=output_dir, save_interval=100)

            assert callback.output_dir == output_dir
            assert callback.save_interval == 100
            assert callback.best_loss == float("inf")

    def test_checkpoint_saved_at_interval(self):
        """Test that checkpoint is saved at specified interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = CheckpointCallback(output_dir=output_dir, save_interval=100)

            # Mock model
            mock_model = Mock()
            mock_model.train_state = Mock()
            mock_model.train_state.epoch = 100
            mock_model.train_state.loss_train = np.array([0.5, 0.3, 0.2])
            mock_model.save = Mock()

            callback.model = mock_model
            callback.on_epoch_end()

            # Verify save was called
            mock_model.save.assert_called_once()
            save_path = mock_model.save.call_args[0][0]
            assert "checkpoint_epoch_100" in str(save_path)

    def test_no_checkpoint_between_intervals(self):
        """Test that checkpoint is not saved between intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = CheckpointCallback(output_dir=output_dir, save_interval=100)

            # Mock model at epoch 50 (not a multiple of 100)
            mock_model = Mock()
            mock_model.train_state = Mock()
            mock_model.train_state.epoch = 50
            mock_model.save = Mock()

            callback.model = mock_model
            callback.on_epoch_end()

            # Save should not be called
            mock_model.save.assert_not_called()

    def test_best_checkpoint_tracking(self):
        """Test that best checkpoint is tracked by lowest loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = CheckpointCallback(output_dir=output_dir, save_interval=100)

            mock_model = Mock()
            mock_model.save = Mock()
            callback.model = mock_model

            # First checkpoint at epoch 100 with loss=1.0
            mock_model.train_state = Mock()
            mock_model.train_state.epoch = 100
            mock_model.train_state.loss_train = np.array([0.5, 0.3, 0.2])
            callback.on_epoch_end()

            assert callback.best_loss == pytest.approx(1.0)

            # Second checkpoint at epoch 200 with loss=0.5 (better)
            mock_model.train_state.epoch = 200
            mock_model.train_state.loss_train = np.array([0.2, 0.15, 0.15])
            callback.on_epoch_end()

            assert callback.best_loss == pytest.approx(0.5)
            assert mock_model.save.call_count == 2


class TestDivergenceDetectionCallback:
    """Test loss divergence detection callback."""

    def test_initialization(self):
        """Test callback initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = DivergenceDetectionCallback(
                output_dir=output_dir,
                nan_threshold=1e10
            )

            assert callback.output_dir == output_dir
            assert callback.nan_threshold == 1e10
            assert callback.divergence_detected is False

    def test_nan_detection(self):
        """Test detection of NaN in loss values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = DivergenceDetectionCallback(output_dir=output_dir)

            # Mock model with NaN loss
            mock_model = Mock()
            mock_model.train_state = Mock()
            mock_model.train_state.epoch = 100
            mock_model.train_state.loss_train = np.array([np.nan, 0.3, 0.2])
            mock_model.stop_training = False

            callback.model = mock_model
            callback.on_epoch_end()

            # Should halt training
            assert mock_model.stop_training is True
            assert callback.divergence_detected is True

    def test_threshold_exceeded_detection(self):
        """Test detection when loss exceeds threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = DivergenceDetectionCallback(
                output_dir=output_dir,
                nan_threshold=100.0
            )

            # Mock model with excessive loss
            mock_model = Mock()
            mock_model.train_state = Mock()
            mock_model.train_state.epoch = 100
            mock_model.train_state.loss_train = np.array([500.0, 300.0, 200.0])
            mock_model.stop_training = False

            callback.model = mock_model
            callback.on_epoch_end()

            # Should halt training
            assert mock_model.stop_training is True
            assert callback.divergence_detected is True

    def test_no_halt_on_normal_loss(self):
        """Test no halt when loss is normal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = DivergenceDetectionCallback(output_dir=output_dir)

            # Mock model with normal loss
            mock_model = Mock()
            mock_model.train_state = Mock()
            mock_model.train_state.epoch = 100
            mock_model.train_state.loss_train = np.array([0.5, 0.3, 0.2])
            mock_model.stop_training = False

            callback.model = mock_model
            callback.on_epoch_end()

            # Should not halt training
            assert mock_model.stop_training is False
            assert callback.divergence_detected is False

    def test_diagnostic_save_on_divergence(self):
        """Test diagnostic information saved on divergence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            callback = DivergenceDetectionCallback(output_dir=output_dir)

            # Mock model with NaN loss
            mock_model = Mock()
            mock_model.train_state = Mock()
            mock_model.train_state.epoch = 100
            mock_model.train_state.loss_train = np.array([np.nan, 0.3, 0.2])
            mock_model.stop_training = False

            callback.model = mock_model
            callback.on_epoch_end()

            # Check diagnostic file was created
            diagnostic_file = output_dir / "divergence_diagnostic.json"
            assert diagnostic_file.exists()

            # Check diagnostic content
            with open(diagnostic_file, 'r') as f:
                diagnostics = json.load(f)

            assert diagnostics["epoch"] == 100
            assert "loss_values" in diagnostics
            assert diagnostics["divergence_detected"] is True
