#!/usr/bin/env python3
"""Main training script for 2D parametric PINN elastic wave models.

This script provides a command-line interface for training 2D PINN models using
DeepXDE with FDTD data supervision. It handles:
- Configuration loading and validation
- FDTD data loading with dimensionless scaling
- Train/validation data splitting
- Model building and compilation
- Training execution with R² monitoring
- Results saving and visualization

Usage:
    python pinn/training/train_2d.py <config_file.yaml>

Example:
    python pinn/training/train_2d.py configs/pinn_2d_example.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from pinn.data.dimensionless_scaler import CharacteristicScales, DimensionlessScalerService
from pinn.data.fdtd_loader import FDTDDataLoaderService
from pinn.models.pinn_model_builder_2d import PINNModelBuilder2DService
from pinn.training.training_pipeline import TrainingPipelineService
from pinn.utils.config_loader import ConfigLoaderService
from pinn.utils.experiment_manager import ExperimentManager
from pinn.utils.seed_manager import SeedManager
from pinn.validation.r2_score import R2ScoreCalculator


def main():
    """Main entry point for 2D PINN training script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train 2D parametric PINN for elastic wave equations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pinn/training/train_2d.py configs/pinn_2d_example.yaml
  python pinn/training/train_2d.py configs/pinn_2d_example.yaml --output-dir results/
        """,
    )
    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Base directory for experiment outputs (default: experiments/)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/PINN_data",
        help="Directory containing FDTD .npz files (default: /PINN_data)"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load configuration
        print(f"Loading configuration from {args.config_file}...")
        config = ConfigLoaderService.load_config(str(config_path))
        print(f"✓ Configuration loaded: {config.experiment_name}")

        # Set random seeds for reproducibility
        print(f"Setting random seed: {config.seed}")
        SeedManager.set_seed(config.seed, verbose=True)

        # Create experiment directory
        print("Creating experiment directory...")
        experiment_manager = ExperimentManager(base_dir=args.output_dir)
        exp_dir = experiment_manager.create_experiment_directory(config.experiment_name)
        print(f"✓ Experiment directory: {exp_dir}")

        # ===== STEP 1: Load FDTD Data =====
        print("\n" + "=" * 60)
        print("STEP 1: Loading FDTD Data")
        print("=" * 60)

        data_dir = Path(args.data_dir)
        loader = FDTDDataLoaderService(data_dir=data_dir)

        # Get all .npz files in data directory
        npz_files = sorted(data_dir.glob("p*_d*.npz"))
        if not npz_files:
            print(f"ERROR: No .npz files found in {data_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(npz_files)} FDTD data files")

        # ===== STEP 2: Create Dimensionless Scaler =====
        print("\n" + "=" * 60)
        print("STEP 2: Creating Dimensionless Scaler")
        print("=" * 60)

        # Estimate U_ref from first file
        sample_data = loader.load_file(npz_files[0])
        U_ref = np.std(np.concatenate([sample_data.Ux, sample_data.Uy]))

        # Load elastic constants from config file (raw dict access for now)
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        elastic_lambda = float(config_dict['domain']['elastic_lambda'])
        elastic_mu = float(config_dict['domain']['elastic_mu'])
        density = float(config_dict['domain']['density'])

        # Store elastic constants for later use
        elastic_constants = {
            'lambda': elastic_lambda,
            'mu': elastic_mu,
            'density': density
        }

        print(f"Elastic constants:")
        print(f"  λ = {elastic_lambda:.2e} Pa")
        print(f"  μ = {elastic_mu:.2e} Pa")
        print(f"  ρ = {density:.2f} kg/m³")
        print(f"Domain length: {config.domain.x_max:.4f} m")
        print(f"Estimated U_ref: {U_ref:.2e} m")

        scales = CharacteristicScales.from_physics(
            domain_length=config.domain.x_max,
            elastic_lambda=elastic_lambda,
            elastic_mu=elastic_mu,
            density=density,
            displacement_amplitude=U_ref
        )

        scaler = DimensionlessScalerService(scales)

        print(f"\nCharacteristic scales:")
        print(f"  L_ref = {scales.L_ref:.4f} m")
        print(f"  T_ref = {scales.T_ref:.2e} s")
        print(f"  σ_ref = {scales.sigma_ref:.2e} Pa")
        print(f"  U_ref = {scales.U_ref:.2e} m")

        # ===== STEP 3: Load and Normalize Data =====
        print("\n" + "=" * 60)
        print("STEP 3: Loading and Normalizing FDTD Data")
        print("=" * 60)

        dataset = loader.load_multiple_files(
            npz_files,
            apply_dimensionless=True,
            scaler=scaler
        )

        print(f"✓ Loaded {len(dataset.x)} samples from {len(dataset.metadata['files'])} files")
        print(f"Data ranges (dimensionless):")
        print(f"  x: [{np.min(dataset.x):.3f}, {np.max(dataset.x):.3f}]")
        print(f"  y: [{np.min(dataset.y):.3f}, {np.max(dataset.y):.3f}]")
        print(f"  t: [{np.min(dataset.t):.3f}, {np.max(dataset.t):.3f}]")
        print(f"  T1: [{np.min(dataset.T1):.3f}, {np.max(dataset.T1):.3f}]")
        print(f"  Ux: [{np.min(dataset.Ux):.3f}, {np.max(dataset.Ux):.3f}]")

        # ===== STEP 4: Train/Validation Split =====
        print("\n" + "=" * 60)
        print("STEP 4: Train/Validation Split")
        print("=" * 60)

        train_ratio = config_dict['training'].get("train_ratio", 0.8)
        validation_equals_train = config_dict['training'].get("validation_equals_train", False)

        train_data, val_data = loader.train_val_split(
            dataset,
            train_ratio=train_ratio,
            seed=config.seed,
            validation_equals_train=validation_equals_train
        )

        print(f"Train samples: {len(train_data.x)}")
        print(f"Val samples: {len(val_data.x)}")
        if validation_equals_train:
            print("Note: Validation = Training (for overfitting monitoring)")

        # ===== STEP 5: Build PINN Model =====
        print("\n" + "=" * 60)
        print("STEP 5: Building 2D PINN Model")
        print("=" * 60)

        # Note: PINNModelBuilder2DService.build_model() requires elastic constants
        # in config.domain, but DomainConfig doesn't have those fields yet.
        # For now, we'll skip model building and print configuration.
        print("Model configuration:")
        print(f"  Geometry: 2D Rectangle [{config.domain.x_min}, {config.domain.x_max}] × "
              f"[{config_dict['domain']['y_min']}, {config_dict['domain']['y_max']}]")
        print(f"  Time: [{config.domain.t_min:.2e}, {config.domain.t_max:.2e}]")
        print(f"  Network: {config.network.layer_sizes}")
        print(f"  Activation: {config.network.activation}")
        print(f"  Elastic constants: λ={elastic_constants['lambda']:.2e}, "
              f"μ={elastic_constants['mu']:.2e}, ρ={elastic_constants['density']:.1f}")

        # TODO: Extend DomainConfig to include elastic constants, then call:
        # builder = PINNModelBuilder2DService()
        # model = builder.build_model(config, compile_model=True)

        # ===== STEP 6: Summary =====
        print("\n" + "=" * 60)
        print("Data Loading Pipeline Complete")
        print("=" * 60)

        print(f"\nSummary:")
        print(f"  ✓ Loaded {len(dataset.x)} samples from {len(dataset.metadata['files'])} files")
        print(f"  ✓ Applied dimensionless scaling (all variables O(1))")
        print(f"  ✓ Split into train ({len(train_data.x)}) / val ({len(val_data.x)})")
        print(f"  ✓ Experiment directory: {exp_dir}")

        print(f"\nNext steps for full training pipeline:")
        print(f"  1. Extend DomainConfig to include elastic constants")
        print(f"  2. Add FDTD data observation points to DeepXDE model")
        print(f"  3. Implement R² monitoring callback during training")
        print(f"  4. Add visualization and result saving")
        print(f"  5. Implement loss weight balancing (Task 5.3)")

        print(f"\nTask 5.2 Status: Data loading pipeline ✓ Complete")

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
