"""Tests for R² validation callback (Task 4.2).

Test-Driven Development: Tests written before implementation.
Tests cover R² monitoring during training with warning thresholds.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from pinn.training.callbacks import R2ValidationCallback


class TestR2ValidationCallback:
    """Test R² validation callback for 2D PINN training."""

    def test_initialization(self):
        """Test callback initialization with default parameters."""
        val_x = np.random.randn(100, 5)  # 100 samples, 5D input
        val_y = np.random.randn(100, 4)  # 100 samples, 4D output

        callback = R2ValidationCallback(
            val_x=val_x,
            val_y=val_y,
            r2_threshold=0.9,
            log_interval=1000
        )

        assert callback.r2_threshold == 0.9
        assert callback.log_interval == 1000
        assert callback.val_x.shape == (100, 5)
        assert callback.val_y.shape == (100, 4)
        assert len(callback.r2_history) == 0

    def test_validation_at_correct_intervals(self):
        """Test that R² is computed at specified intervals."""
        val_x = np.random.randn(50, 5)
        val_y = np.random.randn(50, 4)

        callback = R2ValidationCallback(
            val_x=val_x,
            val_y=val_y,
            r2_threshold=0.9,
            log_interval=500
        )

        # Mock model
        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 500

        # Mock prediction (good quality)
        mock_model.predict = Mock(return_value=val_y + 0.01 * np.random.randn(50, 4))

        callback.model = mock_model

        # Execute callback
        callback.on_epoch_end()

        # Should have computed one R² entry
        assert len(callback.r2_history) == 1
        assert 'epoch' in callback.r2_history[0]
        assert 'scores' in callback.r2_history[0]
        assert callback.r2_history[0]['epoch'] == 500

    def test_no_validation_between_intervals(self):
        """Test that R² is not computed between intervals."""
        val_x = np.random.randn(50, 5)
        val_y = np.random.randn(50, 4)

        callback = R2ValidationCallback(
            val_x=val_x,
            val_y=val_y,
            log_interval=500
        )

        # Mock model at epoch 250 (not multiple of 500)
        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 250

        callback.model = mock_model
        callback.on_epoch_end()

        # Should not have computed R²
        assert len(callback.r2_history) == 0

    def test_r2_scores_per_field(self):
        """Test that R² is computed for each output field."""
        val_x = np.random.randn(50, 5)
        val_y = np.random.randn(50, 4)

        callback = R2ValidationCallback(
            val_x=val_x,
            val_y=val_y,
            log_interval=100
        )

        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 100
        mock_model.predict = Mock(return_value=val_y + 0.01 * np.random.randn(50, 4))

        callback.model = mock_model
        callback.on_epoch_end()

        # Should have R² for all 4 fields
        scores = callback.r2_history[0]['scores']
        assert set(scores.keys()) == {'T1', 'T3', 'Ux', 'Uy'}

        # All should be floats
        for field, r2 in scores.items():
            assert isinstance(r2, float)

    def test_warning_when_r2_below_threshold(self):
        """Test warning is emitted when R² < threshold."""
        val_x = np.random.randn(50, 5)
        val_y = np.random.randn(50, 4)

        callback = R2ValidationCallback(
            val_x=val_x,
            val_y=val_y,
            r2_threshold=0.9,
            log_interval=100
        )

        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 100

        # Poor prediction (random noise)
        mock_model.predict = Mock(return_value=np.random.randn(50, 4))

        callback.model = mock_model

        # Should trigger warning
        with patch('builtins.print') as mock_print:
            callback.on_epoch_end()

            # Check if warning was printed
            assert mock_print.called
            warning_calls = [str(call) for call in mock_print.call_args_list]
            warning_text = ' '.join(warning_calls)
            assert 'WARNING' in warning_text or 'warning' in warning_text.lower()

    def test_no_warning_when_r2_above_threshold(self):
        """Test no warning when R² >= threshold."""
        val_x = np.random.randn(50, 5)
        val_y = np.random.randn(50, 4)

        callback = R2ValidationCallback(
            val_x=val_x,
            val_y=val_y,
            r2_threshold=0.9,
            log_interval=100
        )

        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 100

        # Good prediction (very close to truth)
        mock_model.predict = Mock(return_value=val_y + 0.001 * np.random.randn(50, 4))

        callback.model = mock_model

        # Should not trigger warning for high R²
        with patch('builtins.print') as mock_print:
            callback.on_epoch_end()

            # WARNING should not be in output
            if mock_print.called:
                warning_calls = [str(call) for call in mock_print.call_args_list]
                warning_text = ' '.join(warning_calls)
                # Either no WARNING or only info prints
                if 'WARNING' in warning_text:
                    # If WARNING appears, it should be for a different reason
                    pytest.fail("Unexpected WARNING for high R² scores")

    def test_multiple_validation_events(self):
        """Test multiple validation events accumulate history."""
        val_x = np.random.randn(50, 5)
        val_y = np.random.randn(50, 4)

        callback = R2ValidationCallback(
            val_x=val_x,
            val_y=val_y,
            log_interval=500
        )

        mock_model = Mock()
        callback.model = mock_model

        # First validation at epoch 500
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 500
        mock_model.predict = Mock(return_value=val_y + 0.01 * np.random.randn(50, 4))
        callback.on_epoch_end()

        # Second validation at epoch 1000
        mock_model.train_state.epoch = 1000
        mock_model.predict = Mock(return_value=val_y + 0.01 * np.random.randn(50, 4))
        callback.on_epoch_end()

        # Third validation at epoch 1500
        mock_model.train_state.epoch = 1500
        mock_model.predict = Mock(return_value=val_y + 0.01 * np.random.randn(50, 4))
        callback.on_epoch_end()

        # Should have three R² entries
        assert len(callback.r2_history) == 3
        assert callback.r2_history[0]['epoch'] == 500
        assert callback.r2_history[1]['epoch'] == 1000
        assert callback.r2_history[2]['epoch'] == 1500

    def test_warning_identifies_low_scoring_fields(self):
        """Test warning message identifies which fields have low R²."""
        val_x = np.random.randn(50, 5)
        val_y = np.random.randn(50, 4)

        callback = R2ValidationCallback(
            val_x=val_x,
            val_y=val_y,
            r2_threshold=0.9,
            log_interval=100
        )

        mock_model = Mock()
        mock_model.train_state = Mock()
        mock_model.train_state.epoch = 100

        # Create prediction where one field is poor
        y_pred = val_y.copy()
        y_pred[:, 2] = np.random.randn(50)  # Ux field is poor

        mock_model.predict = Mock(return_value=y_pred)
        callback.model = mock_model

        with patch('builtins.print') as mock_print:
            callback.on_epoch_end()

            # Should mention specific field
            warning_calls = [str(call) for call in mock_print.call_args_list]
            warning_text = ' '.join(warning_calls)
            # Either Ux is mentioned, or field index is shown
            assert 'Ux' in warning_text or 'field' in warning_text.lower()

    def test_callback_compatible_with_deepxde(self):
        """Test callback follows DeepXDE callback interface."""
        val_x = np.random.randn(50, 5)
        val_y = np.random.randn(50, 4)

        callback = R2ValidationCallback(val_x=val_x, val_y=val_y)

        # Should have required methods
        assert hasattr(callback, 'set_model')
        assert hasattr(callback, 'on_train_begin')
        assert hasattr(callback, 'on_epoch_end')
        assert hasattr(callback, 'on_train_end')

        # Test set_model
        mock_model = Mock()
        callback.set_model(mock_model)
        assert callback.model is mock_model
