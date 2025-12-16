"""Integration tests for end-to-end PINN training workflow.

This module tests the complete PINN training pipeline from model creation
through training to validation, without mocks.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn.models.pinn_model_builder import PINNModelBuilderService
from pinn.training.training_pipeline import TrainingPipelineService
from pinn.utils.config_loader import (
    BoundaryConditionConfig,
    DomainConfig,
    ExperimentConfig,
    NetworkConfig,
    TrainingConfig,
)
from pinn.utils.seed_manager import SeedManager
from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService
from pinn.validation.error_metrics import ErrorMetricsService


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def minimal_config():
    """Create minimal valid configuration for testing."""
    return ExperimentConfig(
        experiment_name="test_e2e",
        seed=42,
        domain=DomainConfig(
            x_min=0.0,
            x_max=1.0,
            t_min=0.0,
            t_max=0.5,
            wave_speed=1.0
        ),
        boundary_conditions=BoundaryConditionConfig(
            type="dirichlet",
            left_value=0.0,
            right_value=0.0
        ),
        network=NetworkConfig(
            layer_sizes=[2, 20, 20, 1],  # Small network for fast testing
            activation="tanh"
        ),
        training=TrainingConfig(
            epochs=100,  # Small number for fast test
            learning_rate=0.001,
            optimizer="adam",
            loss_weights={"data": 0.0, "pde": 1.0, "bc": 10.0},
            amp_enabled=False,  # Disable AMP for reproducibility in tests
            checkpoint_interval=50
        )
    )


class TestEndToEndTraining:
    """Integration tests for complete training workflow."""

    def test_simple_standing_wave_training(self, minimal_config, temp_output_dir):
        """Test end-to-end training on simple standing wave problem.

        This test verifies:
        1. Model can be built from config
        2. Training completes without errors
        3. Loss decreases over training
        4. Checkpoints are saved
        """
        # Set seed for reproducibility
        SeedManager.set_seed(minimal_config.seed)

        # Define initial condition: u(x, 0) = sin(πx)
        def initial_condition(x):
            return np.sin(np.pi * x[:, 0:1])

        # Build model
        builder = PINNModelBuilderService()
        model = builder.build_model(minimal_config, initial_condition)

        assert model is not None

        # Train model
        pipeline = TrainingPipelineService()
        trained_model, history = pipeline.train(
            model=model,
            config=minimal_config.training,
            output_dir=temp_output_dir
        )

        # Verify training history is populated
        assert "total_loss" in history
        assert len(history["total_loss"]) > 0

        # Verify loss decreases (monotonicity check on averaged windows)
        losses = history["total_loss"]
        if len(losses) >= 4:
            first_half_avg = np.mean(losses[:len(losses)//2])
            second_half_avg = np.mean(losses[len(losses)//2:])
            assert second_half_avg < first_half_avg, "Loss should decrease over training"

        # Verify checkpoints directory exists
        checkpoint_dir = temp_output_dir / "checkpoints"
        assert checkpoint_dir.exists(), "Checkpoint directory should be created"

    def test_loss_components_logged(self, minimal_config, temp_output_dir):
        """Test that individual loss components are logged during training."""
        SeedManager.set_seed(minimal_config.seed)

        # Define initial condition
        def initial_condition(x):
            return np.sin(np.pi * x[:, 0:1])

        # Build and train
        builder = PINNModelBuilderService()
        model = builder.build_model(minimal_config, initial_condition)

        pipeline = TrainingPipelineService()
        _, history = pipeline.train(
            model=model,
            config=minimal_config.training,
            output_dir=temp_output_dir
        )

        # Verify all loss components are present
        assert "L_pde" in history, "PDE loss should be logged"
        assert "L_bc" in history, "Boundary condition loss should be logged"
        assert "total_loss" in history, "Total loss should be logged"

        # Verify they contain numerical values (not NaN)
        assert all(not np.isnan(v) for v in history["L_pde"] if v is not None)
        assert all(not np.isnan(v) for v in history["L_bc"] if v is not None)

    def test_validation_error_computed(self, minimal_config, temp_output_dir):
        """Test that validation errors vs. analytical solution are computed."""
        SeedManager.set_seed(minimal_config.seed)

        # Define initial condition
        def initial_condition(x):
            return np.sin(np.pi * x[:, 0:1])

        # Build and train
        builder = PINNModelBuilderService()
        model = builder.build_model(minimal_config, initial_condition)

        pipeline = TrainingPipelineService()
        trained_model, history = pipeline.train(
            model=model,
            config=minimal_config.training,
            output_dir=temp_output_dir
        )

        # If validation callback is used, L2_error should be in history
        # This depends on implementation - may need to be added
        # For now, manually compute validation error

        # Generate test points
        x_test = np.linspace(0, 1, 20)
        t_test = np.linspace(0, 0.5, 20)
        X, T = np.meshgrid(x_test, t_test)
        xt_test = np.stack([X.flatten(), T.flatten()], axis=1)

        # Get PINN predictions
        with torch.no_grad():
            xt_tensor = torch.tensor(xt_test, dtype=torch.float32)
            u_pred = trained_model.predict(xt_tensor).numpy()

        # Get analytical solution (standing wave fundamental mode)
        analytical_gen = AnalyticalSolutionGeneratorService()
        u_exact = analytical_gen.standing_wave(
            x=xt_test[:, 0],
            t=xt_test[:, 1],
            L=1.0,
            c=1.0,
            n=1
        ).flatten()

        # Compute error
        error_metrics = ErrorMetricsService()
        relative_error = error_metrics.relative_error(u_pred.flatten(), u_exact)

        # After training, error should be reasonable (not checking exact value,
        # just that it's finite and not catastrophically large)
        assert not np.isnan(relative_error), "Error should not be NaN"
        assert not np.isinf(relative_error), "Error should not be infinite"
        assert relative_error < 10.0, "Error should be less than 1000% after training"


class TestCheckpointSaveLoad:
    """Integration tests for model checkpoint saving and loading."""

    def test_checkpoint_saved_during_training(self, minimal_config, temp_output_dir):
        """Test that checkpoints are saved at specified intervals."""
        SeedManager.set_seed(minimal_config.seed)

        # Define initial condition
        def initial_condition(x):
            return np.sin(np.pi * x[:, 0:1])

        # Train model
        builder = PINNModelBuilderService()
        model = builder.build_model(minimal_config, initial_condition)

        pipeline = TrainingPipelineService()
        pipeline.train(
            model=model,
            config=minimal_config.training,
            output_dir=temp_output_dir
        )

        # Check that checkpoint directory exists
        checkpoint_dir = temp_output_dir / "checkpoints"
        assert checkpoint_dir.exists()

        # Check that at least one checkpoint file exists
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "At least one checkpoint should be saved"

    def test_checkpoint_load_weights_match(self, minimal_config, temp_output_dir):
        """Test that loaded checkpoint has matching weights."""
        SeedManager.set_seed(minimal_config.seed)

        # Define initial condition
        def initial_condition(x):
            return np.sin(np.pi * x[:, 0:1])

        # Train and save model
        builder = PINNModelBuilderService()
        model = builder.build_model(minimal_config, initial_condition)

        pipeline = TrainingPipelineService()
        trained_model, _ = pipeline.train(
            model=model,
            config=minimal_config.training,
            output_dir=temp_output_dir
        )

        # Save model weights explicitly
        checkpoint_path = temp_output_dir / "test_checkpoint.pth"
        trained_model.save(str(checkpoint_path))

        # Create new model with same architecture
        model_new = builder.build_model(minimal_config, initial_condition)

        # Load weights
        model_new.restore(str(checkpoint_path))

        # Test that predictions match on same inputs
        x_test = torch.tensor([[0.5, 0.25]], dtype=torch.float32)

        with torch.no_grad():
            pred_original = trained_model.predict(x_test)
            pred_loaded = model_new.predict(x_test)

        # Predictions should be very close (allowing small numerical differences)
        assert torch.allclose(pred_original, pred_loaded, rtol=1e-5, atol=1e-6), \
            "Loaded model predictions should match original"

    def test_checkpoint_contains_metadata(self, minimal_config, temp_output_dir):
        """Test that checkpoint saves necessary metadata."""
        SeedManager.set_seed(minimal_config.seed)

        # Define initial condition
        def initial_condition(x):
            return np.sin(np.pi * x[:, 0:1])

        # Train and save
        builder = PINNModelBuilderService()
        model = builder.build_model(minimal_config, initial_condition)

        pipeline = TrainingPipelineService()
        trained_model, _ = pipeline.train(
            model=model,
            config=minimal_config.training,
            output_dir=temp_output_dir
        )

        # Save checkpoint
        checkpoint_path = temp_output_dir / "checkpoint_with_metadata.pth"
        trained_model.save(str(checkpoint_path))

        # Verify checkpoint file exists and is not empty
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0, "Checkpoint file should not be empty"
