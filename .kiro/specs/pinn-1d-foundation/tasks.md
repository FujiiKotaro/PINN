# Implementation Plan

## Overview
Implementation tasks for the PINN 1D Foundation feature, organized into logical phases: Project Setup → Model Layer → Training Infrastructure → Validation → Data Utilities → Integration & Testing.

---

## Tasks

### Phase 1: Project Setup & Configuration

- [ ] 1. Set up project structure and configuration management
- [x] 1.1 (P) Create project directory structure
  - Create `/pinn/` directory hierarchy: `models/`, `training/`, `validation/`, `data/`, `tuning/`, `utils/`, `tests/`
  - Create `/configs/` directory for YAML experiment configurations
  - Create `/experiments/` directory template for timestamped output directories
  - Set up `__init__.py` files for Python package structure
  - _Requirements: 9.1_

- [x] 1.2 (P) Implement configuration loading with Pydantic validation
  - Define Pydantic models for config sections: `DomainConfig`, `BoundaryConditionConfig`, `NetworkConfig`, `TrainingConfig`, `ExperimentConfig`
  - Implement `ConfigLoaderService.load_config()` to parse YAML files using PyYAML
  - Implement `ConfigLoaderService.save_config()` for reproducibility
  - Add validation rules: `wave_speed > 0`, `t_max > t_min`, `layer_sizes` non-empty list
  - Handle FileNotFoundError and ValidationError with clear error messages
  - _Requirements: 9.1, 9.2_

- [x] 1.3 (P) Set up seed management for reproducibility
  - Implement `SeedManager` utility to set seeds for NumPy, PyTorch, and Python's random module
  - Create function to log seed values to metadata JSON
  - Ensure seed is applied before any random operations (model init, data sampling, collocation points)
  - _Requirements: 9.2, 9.5_

### Phase 2: Model Layer - PINN Core

- [ ] 2. Build DeepXDE PINN model for 1D wave equation
- [x] 2.1 (P) Implement PDE definition for wave equation
  - Create `PDEDefinitionService.wave_equation_residual()` function
  - Use `dde.grad.jacobian()` to compute ∂u/∂x and ∂u/∂t
  - Use `dde.grad.hessian()` to compute ∂²u/∂x² and ∂²u/∂t²
  - Return residual: ∂²u/∂t² - c²∂²u/∂x²
  - Add docstring with mathematical formula and parameter descriptions
  - _Requirements: 1.1, 1.6_

- [x] 2.2 Define spatiotemporal geometry using DeepXDE
  - Create `_create_geometry()` method in `PINNModelBuilderService`
  - Use `dde.geometry.Interval(x_min, x_max)` for spatial domain
  - Use `dde.geometry.TimeDomain(t_min, t_max)` for temporal domain
  - Combine with `dde.geometry.GeometryXTime(geom, timedomain)`
  - Accept domain parameters from `DomainConfig` (x_min, x_max, t_min, t_max)
  - _Requirements: 1.3_

- [x] 2.3 (P) Implement boundary condition builders
  - Create `BoundaryConditionsService.create_dirichlet_bc()` for Dirichlet BCs (u=0 at boundaries)
  - Create `BoundaryConditionsService.create_neumann_bc()` for Neumann BCs (∂u/∂n=0)
  - Create helper functions to define boundary predicates (`on_boundary` lambda functions)
  - Support configurable BC types based on `BoundaryConditionConfig.type`
  - _Requirements: 2.3, 2.4, 2.5_

- [x] 2.4 (P) Implement initial condition builders
  - Create `BoundaryConditionsService.create_initial_condition()` for u(x, 0) = f(x)
  - Create `BoundaryConditionsService.create_initial_velocity()` for ∂u/∂t(x, 0) = g(x)
  - Use `dde.icbc.IC` for displacement IC
  - Use `dde.icbc.OperatorBC` for velocity IC (requires custom gradient operator)
  - _Requirements: 2.1, 2.2_

- [x] 2.5 Assemble complete PINN model
  - Implement `PINNModelBuilderService.build_model()` main entry point
  - Create feedforward neural network using `dde.nn.FNN` with configurable layer sizes from `NetworkConfig`
  - Combine geometry, PDE, BCs, ICs into `dde.data.TimePDE` data object
  - Instantiate `dde.Model(data, net)` and compile with optimizer
  - Accept wave speed `c` as configurable parameter
  - _Requirements: 1.2, 1.4, 1.5, 2.5_

### Phase 3: Training Infrastructure

- [ ] 3. Implement GPU-accelerated training pipeline
- [x] 3.1 (P) Build device detection and GPU management
  - Implement `TrainingPipelineService._detect_device()` to check CUDA availability with `torch.cuda.is_available()`
  - Log warning message if CUDA unavailable (fallback to CPU)
  - Move model parameters to detected device
  - Add GPU memory logging utility: `log_gpu_memory()` using `torch.cuda.memory_allocated()` and `memory_reserved()`
  - _Requirements: 5.1, 5.2, 5.4, 5.5_

- [x] 3.2 (P) Implement PyTorch AMP wrapper for mixed precision
  - Create `AMPWrapperService.__init__()` to check if AMP should be enabled (GPU + config flag)
  - Implement `autocast()` context manager wrapping `torch.cuda.amp.autocast()`
  - Implement `scale_and_step()` using `torch.cuda.amp.GradScaler()` for loss scaling
  - Handle AMP disable on NaN detection (fallback to full precision)
  - Add `log_gpu_memory()` to track memory reduction from mixed precision
  - _Requirements: 5.3, 5.5_

- [x] 3.3 Implement multi-component loss computation
  - Ensure DeepXDE model computes L_pde, L_bc, and L_data separately
  - Configure loss weights (w_data, w_pde, w_bc) via `dde.Model.compile(loss_weights=...)`
  - Verify loss computation formula: L_total = w_data × L_data + w_pde × L_pde + w_bc × L_bc
  - Extract individual loss components from `model.train_state.loss_train`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3.4 Build custom callbacks for training monitoring
  - Implement `LossLoggingCallback` to log L_data, L_pde, L_bc every N epochs
  - Store loss history in dictionary: `{"L_data": [], "L_pde": [], "L_bc": [], "total_loss": []}`
  - Implement `CheckpointCallback` to save model weights at configurable intervals
  - Save best checkpoint based on lowest validation error
  - _Requirements: 3.6, 7.1, 7.3_

- [x] 3.5 Implement validation callback with analytical solution comparison
  - Create `ValidationCallback` to compute L2 error every M epochs
  - Call `AnalyticalSolutionGenerator` to get ground truth solution
  - Call `ErrorMetrics.l2_error()` and `ErrorMetrics.relative_error()`
  - Log validation errors to history
  - Flag model if relative error > 5% threshold
  - _Requirements: 4.3, 4.4, 4.6, 7.2_

- [x] 3.6 Implement loss divergence detection and training halt
  - Add NaN detection in callback: check if `loss.isnan()` or `loss > threshold`
  - Halt training immediately with `self.model.stop_training = True`
  - Save diagnostic information: current loss values, model state, epoch number
  - Log error message with diagnostic file path
  - _Requirements: 7.4_

- [x] 3.7 Assemble complete training pipeline
  - Implement `TrainingPipelineService.train()` main entry point
  - Accept compiled `dde.Model`, `TrainingConfig`, and output directory
  - Register all callbacks: LossLogging, Validation, Checkpoint
  - Configure collocation point batch sizes from config
  - Execute `model.train(epochs=config.epochs, callbacks=[...])`
  - Return trained model and training history dict
  - _Requirements: 5.6, 7.5, 7.6_

### Phase 4: Validation Layer

- [x] 4. Build analytical solution validation infrastructure
- [x] 4.1 (P) Implement standing wave analytical solution generator
  - Create `AnalyticalSolutionGeneratorService.standing_wave()` function
  - Implement formula: u(x,t) = sin(nπx/L) cos(nπct/L)
  - Accept parameters: spatial coordinates `x`, temporal coordinates `t`, domain length `L`, wave speed `c`, mode number `n`
  - Return solution array with shape matching input meshgrid
  - Add docstring with mathematical derivation reference
  - _Requirements: 4.1_

- [x] 4.2 (P) Implement traveling wave analytical solution generator
  - Create `AnalyticalSolutionGeneratorService.traveling_wave()` function
  - Implement formula: u(x,t) = f(x - ct) + g(x + ct)
  - Accept initial condition function `f(x)` as callable parameter
  - Return solution array for d'Alembert's solution
  - _Requirements: 4.2_

- [x] 4.3 (P) Implement error metric computations
  - Create `ErrorMetricsService.l2_error()` using `np.linalg.norm(u_pred - u_exact)`
  - Create `ErrorMetricsService.relative_error()`: L2 error divided by `np.linalg.norm(u_exact)`
  - Create `ErrorMetricsService.max_absolute_error()` using `np.max(np.abs(...))`
  - Add input validation for shape compatibility
  - _Requirements: 4.3, 4.4_

- [x] 4.4 (P) Implement visualization utilities for solution comparison
  - Create `PlotGeneratorService.plot_training_curves()` for loss history (total loss, L_data, L_pde, L_bc vs. epochs)
  - Create `PlotGeneratorService.plot_solution_comparison()` for PINN vs. analytical at multiple time snapshots
  - Use Matplotlib/Seaborn for publication-quality plots
  - Save plots to experiment output directory with descriptive filenames
  - _Requirements: 4.5, 7.5_

### Phase 5: Data Layer (FDTD Utilities for Phase 2)

- [ ] 5. Build FDTD .npz data loading utilities
- [ ] 5.1 (P) Implement .npz file loader with metadata extraction
  - Create `FDTDDataLoaderService.load_file()` to read .npz files using `np.load()`
  - Extract spatiotemporal coordinates: x, y, t (flattened arrays)
  - Extract wave field data: T1, T3, Ux, Uy
  - Extract metadata: pitch, depth, width, seed, nx_sample, ny_sample, nt_sample
  - Return `FDTDData` dataclass container with all fields
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 5.2 Implement data validation for .npz files
  - Create `FDTDDataLoaderService.validate_data()` function
  - Check expected array shape: `(nt*nx*ny,)` for all coordinate/field arrays
  - Verify all required keys present: x, y, t, T1, T3, Ux, Uy, p, d, w, seed
  - Raise `ValueError` with descriptive message for malformed files
  - Handle `KeyError` for missing keys
  - _Requirements: 8.6_

- [ ] 5.3 (P) Implement multi-file loading with parameter filtering
  - Create `FDTDDataLoaderService.load_multiple()` function
  - Accept optional `pitch_range` and `depth_range` filters (tuples of min/max values)
  - Scan `/PINN_data/` directory for .npz files matching naming pattern `p{pitch}_d{depth}.npz`
  - Filter files by metadata before loading (parse filename to extract pitch/depth)
  - Concatenate data from multiple files into list of `FDTDData` objects
  - _Requirements: 8.5, 8.7_

- [ ] 5.4 (P) Implement tensor conversion utilities
  - Create `TensorConverterService.to_tensor()` to convert NumPy array to PyTorch tensor
  - Support configurable dtype: float32 (for AMP) or float64 (for precision)
  - Support configurable device: CPU or CUDA
  - Create `TensorConverterService.batch_convert()` to convert all `FDTDData` arrays at once
  - Return dict of tensors: `{"x": tensor, "y": tensor, "t": tensor, "T1": tensor, ...}`
  - _Requirements: 8.8_

### Phase 6: Tuning Framework

- [ ] 6. Implement automated loss weight tuning framework
- [ ] 6.1 Implement grid search for weight combinations
  - Create `WeightTuningFrameworkService._grid_search()` function
  - Accept `weight_ranges` dict: `{"data": [0.1, 1.0, 10.0], "pde": [0.1, 1.0, 10.0], "bc": [0.1, 1.0, 10.0]}`
  - Use `itertools.product()` to generate all combinations
  - Return list of weight dicts for training
  - _Requirements: 6.1_

- [ ] 6.2 (P) Implement random search for weight sampling
  - Create `WeightTuningFrameworkService._random_search()` function
  - Accept `weight_ranges` dict and `n_samples` parameter
  - Use `random.choice()` to sample weights from ranges
  - Return list of n_samples weight combinations
  - _Requirements: 6.2_

- [ ] 6.3 Implement tuning execution loop
  - Create `WeightTuningFrameworkService.run_tuning()` main entry point
  - Accept `base_config` (ExperimentConfig) and `tuning_config` (search type, ranges, n_samples)
  - Generate weight combinations using grid or random search
  - For each combination: train PINN model, compute validation error
  - Store results in `TuningResult` dataclass: (w_data, w_pde, w_bc, validation_error, training_time)
  - Identify best configuration by minimum validation error
  - _Requirements: 6.3, 6.4, 6.6_

- [ ] 6.4 (P) Implement tuning results logging
  - Save tuning results to JSON or CSV file
  - Include columns: w_data, w_pde, w_bc, validation_error, training_time, timestamp
  - Save to `output_path` from `TuningConfig`
  - Log progress during tuning (e.g., "Completed 5/27 configurations")
  - _Requirements: 6.5_

- [ ] 6.5 (P) Implement tuning visualization
  - Create `visualize_results()` function in `WeightTuningFrameworkService`
  - Generate loss landscape heatmap (2D projection of 3D weight space)
  - Generate Pareto frontier plot if multi-objective (validation error vs. training time)
  - Use Matplotlib/Seaborn for plotting
  - Save plots to output directory
  - _Requirements: 6.7_

### Phase 7: Reproducibility & Metadata

- [ ] 7. Implement experiment tracking and reproducibility
- [ ] 7.1 (P) Implement metadata logger for software versions
  - Create `MetadataLogger` utility to capture Python, PyTorch, DeepXDE, NumPy versions
  - Use `sys.version`, `torch.__version__`, `dde.__version__`, etc.
  - Save to `metadata.json` in experiment output directory
  - Include seed, timestamp, config hash for reproducibility
  - _Requirements: 9.3, 9.4_

- [ ] 7.2 Implement experiment directory organization
  - Create timestamped experiment directories: `/experiments/exp_{timestamp}/`
  - Save config YAML to experiment directory for reproducibility
  - Create subdirectories: `checkpoints/`, `logs/`, `plots/`
  - Generate experiment name from `config.experiment_name` or auto-generate
  - _Requirements: 9.6_

### Phase 8: Testing

- [ ] 8. Build comprehensive test suite
- [ ] 8.1 (P) Write unit tests for PDE residual computation
  - Test `wave_equation_residual()` with known analytical gradients
  - Verify residual → 0 for exact analytical solution
  - Test gradient computation with simple polynomial u(x,t)
  - Use pytest with tolerance for numerical errors
  - _Requirements: 10.1_

- [ ] 8.2 (P) Write unit tests for analytical solution generators
  - Test `standing_wave()` output against textbook formulas
  - Test `traveling_wave()` with known initial conditions
  - Verify solution satisfies wave equation (substitute into PDE)
  - Check boundary condition satisfaction
  - _Requirements: 10.1_

- [ ] 8.3 (P) Write unit tests for error metrics
  - Test `l2_error()` and `relative_error()` with synthetic arrays
  - Verify zero error for identical arrays
  - Test `max_absolute_error()` with known differences
  - Check edge cases: zero arrays, NaN handling
  - _Requirements: 10.1_

- [ ] 8.4 (P) Write unit tests for FDTD data loader
  - Create pytest fixture with mock .npz file (extract from `/PINN_data/`)
  - Test `load_file()` extracts all fields correctly
  - Test `validate_data()` catches malformed files (missing keys, wrong shapes)
  - Test `load_multiple()` filtering by pitch/depth ranges
  - _Requirements: 10.3_

- [ ] 8.5 (P) Write unit tests for config loader
  - Test `load_config()` with valid YAML files
  - Test Pydantic validation catches invalid configs (negative wave_speed, t_max < t_min)
  - Test `save_config()` round-trip consistency
  - Test FileNotFoundError handling
  - _Requirements: 10.1_

- [ ] 8.6 Write integration test for end-to-end training
  - Train PINN on simple standing wave (fundamental mode, 10 epochs)
  - Assert total loss decreases monotonically
  - Verify loss components (L_pde, L_bc) logged to history
  - Check checkpoint saved to output directory
  - _Requirements: 10.2_

- [ ] 8.7 (P) Write integration test for checkpoint save/load
  - Train model for 100 epochs, save checkpoint
  - Load checkpoint into new model instance
  - Verify weights match using `torch.allclose()`
  - Test checkpoint contains metadata (epoch, optimizer state)
  - _Requirements: 10.2_

- [ ] 8.8 (P) Write performance test for training time
  - Train 1D wave PINN with 10k collocation points for 1000 epochs on GPU
  - Measure elapsed time using `time.time()`
  - Assert training completes in <5 minutes (requirement: <5 min for 10k points)
  - Log GPU memory usage during training
  - _Requirements: Non-functional performance_

- [ ] 8.9 (P) Write performance test for data loading
  - Load single 4.7 MB .npz file from `/PINN_data/`
  - Measure elapsed time using `time.time()`
  - Assert loading completes in <2 seconds (requirement: <2 sec)
  - _Requirements: Non-functional performance_

- [ ] 8.10 (P) Write convergence test for AMP vs. full precision
  - Train identical PINN configuration with AMP (float16) and full precision (float32)
  - Compare final validation errors
  - Assert relative error difference <5% between AMP and full precision
  - If AMP degrades accuracy >5%, log warning and suggest CPU-only mode
  - _Requirements: 10.2, 5.3_

- [ ] 8.11 Configure pytest coverage measurement
  - Set up `pytest-cov` in `pyproject.toml`
  - Configure coverage targets: exclude tests/, venv/, migrations/
  - Run `pytest --cov=pinn --cov-report=html --cov-report=term`
  - Verify coverage meets 70% minimum threshold
  - _Requirements: 10.4_

- [ ] 8.12 Set up Ruff linting and formatting
  - Configure Ruff in `pyproject.toml` with project-specific rules
  - Run `ruff check .` to verify zero errors
  - Set up pre-commit hook (optional) for auto-formatting
  - Fix any existing linting violations
  - _Requirements: 10.5_

### Phase 9: Documentation & Integration

- [ ] 9. Create documentation and example workflows
- [ ] 9.1 (P) Write module docstrings following NumPy convention
  - Add docstrings to all public functions and classes
  - Include Parameters, Returns, Raises sections
  - Document mathematical formulas in PDE definition module
  - Add type hints to function signatures
  - _Requirements: 10.6, 10.7_

- [ ] 9.2 Create example YAML configuration files
  - Create `configs/standing_wave_example.yaml` for standing wave test case
  - Create `configs/traveling_wave_example.yaml` for traveling wave
  - Create `configs/dirichlet_bc_example.yaml` and `configs/neumann_bc_example.yaml`
  - Document config parameters with inline comments
  - _Requirements: 9.1_

- [ ] 9.3 (P) Create Jupyter notebook demonstration
  - Create `notebooks/wave_1d_demo.ipynb` showing end-to-end workflow
  - Load config, build model, train PINN, visualize results
  - Include analytical solution comparison plots
  - Add markdown cells explaining PINN methodology
  - Show loss weight tuning example (small grid search)
  - _Requirements: Non-functional documentation_

- [ ] 9.4 Create README for /pinn/ module
  - Document setup instructions: Poetry install, CUDA requirements
  - Provide usage examples: training script invocation, config file format
  - Include analytical validation results (L2 error, training time benchmarks)
  - List dependencies and version requirements (PyTorch 2.4.0, CUDA 12.4, DeepXDE 1.15.0)
  - _Requirements: Non-functional documentation_

### Phase 10: Final Integration

- [ ] 10. Integrate all components and validate end-to-end
- [ ] 10.1 Create main training script
  - Create `pinn/training/train.py` CLI script accepting config file path as argument
  - Load config, set seeds, build model, train pipeline
  - Save outputs to timestamped experiment directory
  - Log progress to console and JSON file
  - Handle exceptions gracefully with error messages
  - _Requirements: 9.3, 9.6_

- [ ] 10.2 Create weight tuning script
  - Create `pinn/tuning/tune_weights.py` CLI script accepting tuning config
  - Execute grid or random search over weight combinations
  - Save tuning results to CSV/JSON
  - Generate loss landscape visualizations
  - _Requirements: 6.1, 6.2, 6.3, 6.5, 6.7_

- [ ] 10.3 Run full integration validation
  - Execute training script with standing wave config: `python pinn/training/train.py configs/standing_wave_example.yaml`
  - Verify outputs: checkpoints saved, plots generated, metadata logged
  - Check validation error <5% threshold
  - Run weight tuning script with small grid (3×3×3)
  - Verify best weights identified and logged
  - _Requirements: All functional requirements_

- [ ] 10.4 Verify project structure matches steering conventions
  - Check directory layout: `/pinn/models/`, `/pinn/training/`, `/pinn/validation/`, `/pinn/data/`, `/pinn/tuning/`, `/pinn/utils/`, `/pinn/tests/`
  - Verify file naming: `snake_case.py` modules, `PascalCase` classes
  - Check import organization: stdlib → third-party → local
  - Ensure docstrings and type hints present
  - _Requirements: Non-functional maintainability_

---

## Requirements Coverage Summary

All 10 functional requirements mapped to tasks:

- **Requirement 1** (1D Wave Equation PINN Model): Tasks 2.1-2.5, 8.1
- **Requirement 2** (Boundary/Initial Conditions): Tasks 2.3-2.4, 8.2
- **Requirement 3** (Physics-Informed Loss): Tasks 3.3, 3.4
- **Requirement 4** (Analytical Validation): Tasks 4.1-4.4, 8.2
- **Requirement 5** (GPU Acceleration): Tasks 3.1, 3.2, 8.10
- **Requirement 6** (Loss Weight Tuning): Tasks 6.1-6.5, 10.2
- **Requirement 7** (Training Monitoring): Tasks 3.4-3.6, 4.4
- **Requirement 8** (FDTD Data Loading): Tasks 5.1-5.4, 8.4
- **Requirement 9** (Reproducibility): Tasks 1.2-1.3, 7.1-7.2
- **Requirement 10** (Code Quality/Testing): Tasks 8.1-8.12

Non-functional requirements (performance, maintainability, compatibility, documentation) covered in tasks 8.8-8.9, 9.1-9.4, 10.4.
