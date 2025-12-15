# Requirements Document

## Introduction

This document defines requirements for the foundational 1D PINN (Physics-Informed Neural Network) implementation using DeepXDE. The goal is to validate the PINN infrastructure, training pipeline, and GPU acceleration before advancing to 2D elastic wave physics. This phase serves as a framework validation step, testing against analytical solutions rather than FDTD data.

**Scope**: 1D wave equation solver, loss function tuning framework, GPU-accelerated training pipeline, and FDTD data loading utilities for future phases.

**Out of Scope**: 2D elastic wave physics, FDTD data comparison, crack geometry modeling, production deployment.

## Requirements

### Requirement 1: 1D Wave Equation PINN Model

**Objective:** As a physics simulation developer, I want a DeepXDE-based 1D wave equation solver, so that I can validate the PINN framework and training infrastructure before tackling 2D problems.

#### Acceptance Criteria

1. The PINN Model shall implement the 1D wave equation: ∂²u/∂t² = c² ∂²u/∂x²
2. The PINN Model shall use DeepXDE 1.15.0 framework for PDE definition and neural network construction
3. The PINN Model shall define spatial domain [0, L] and temporal domain [0, T] as geometry constraints
4. The PINN Model shall accept wave speed parameter c as configurable input
5. When model is initialized, the PINN Model shall create a feedforward neural network with configurable layer sizes
6. The PINN Model shall use automatic differentiation to compute PDE residuals (∂²u/∂t² - c² ∂²u/∂x²)

### Requirement 2: Boundary and Initial Conditions

**Objective:** As a physics simulation developer, I want proper boundary and initial conditions, so that the PINN solution matches analytical wave propagation behavior.

#### Acceptance Criteria

1. The PINN Model shall enforce initial condition u(x, 0) = f(x) using DeepXDE's IC constraint
2. The PINN Model shall enforce initial velocity condition ∂u/∂t(x, 0) = g(x) using DeepXDE's IC constraint
3. Where Dirichlet boundary conditions are specified, the PINN Model shall enforce u(0, t) = u(L, t) = 0
4. Where Neumann boundary conditions are specified, the PINN Model shall enforce ∂u/∂x at boundaries using DeepXDE's NBC constraint
5. The PINN Model shall support configurable boundary condition types (Dirichlet, Neumann, periodic)

### Requirement 3: Physics-Informed Loss Function

**Objective:** As a physics simulation developer, I want a tunable multi-component loss function, so that I can balance PDE constraints, data fitting, and boundary conditions.

#### Acceptance Criteria

1. The Training Pipeline shall compute total loss as: L_total = w_data × L_data + w_pde × L_pde + w_bc × L_bc
2. The Training Pipeline shall accept loss weight parameters (w_data, w_pde, w_bc) as configurable inputs
3. The Training Pipeline shall compute L_pde as mean squared residual of the wave equation PDE
4. The Training Pipeline shall compute L_bc as mean squared error on boundary condition violations
5. When training data points are provided, the Training Pipeline shall compute L_data as mean squared error between predictions and ground truth
6. The Training Pipeline shall log individual loss components (L_data, L_pde, L_bc) during training for monitoring

### Requirement 4: Analytical Solution Validation

**Objective:** As a physics simulation developer, I want to validate PINN predictions against analytical solutions, so that I can verify correctness before using the framework for complex problems.

#### Acceptance Criteria

1. The Validation Module shall generate analytical solution for standing wave: u(x,t) = sin(nπx/L) cos(nπct/L)
2. The Validation Module shall generate analytical solution for traveling wave: u(x,t) = f(x - ct) + g(x + ct)
3. When PINN training completes, the Validation Module shall compute L2 error between PINN predictions and analytical solution
4. The Validation Module shall compute relative error as ||u_PINN - u_analytical||₂ / ||u_analytical||₂
5. The Validation Module shall generate visualization plots comparing PINN vs. analytical solutions at multiple time snapshots
6. If relative error exceeds 5%, then the Validation Module shall flag the model as requiring hyperparameter tuning

### Requirement 5: GPU-Accelerated Training Pipeline

**Objective:** As a physics simulation developer, I want GPU-accelerated training using PyTorch CUDA backend, so that I can efficiently train PINNs and prepare for computationally intensive 2D models.

#### Acceptance Criteria

1. The Training Pipeline shall detect CUDA availability and automatically use GPU if present
2. The Training Pipeline shall move model parameters and training data to CUDA device
3. The Training Pipeline shall use PyTorch's automatic mixed precision (AMP) for memory efficiency
4. When CUDA is unavailable, the Training Pipeline shall fall back to CPU with warning message
5. The Training Pipeline shall log GPU memory usage and utilization metrics during training
6. The Training Pipeline shall support configurable batch sizes for collocation point sampling

### Requirement 6: Loss Weight Tuning Framework

**Objective:** As a physics simulation developer, I want an automated framework for tuning loss weights, so that I can systematically find optimal balance between data fitting and physics constraints.

#### Acceptance Criteria

1. The Tuning Framework shall support grid search over w_data, w_pde, w_bc parameter ranges
2. The Tuning Framework shall support random search sampling from specified weight distributions
3. When tuning is initiated, the Tuning Framework shall train multiple models with different weight combinations
4. The Tuning Framework shall evaluate each configuration using validation error metrics
5. The Tuning Framework shall log tuning results to structured format (JSON or CSV)
6. The Tuning Framework shall identify best-performing weight combination based on validation error
7. The Tuning Framework shall visualize tuning results as loss landscape or Pareto frontier plots

### Requirement 7: Training Monitoring and Logging

**Objective:** As a physics simulation developer, I want comprehensive training monitoring, so that I can diagnose convergence issues and track model performance.

#### Acceptance Criteria

1. The Training Pipeline shall log epoch number, total loss, and component losses (L_data, L_pde, L_bc) every N epochs
2. The Training Pipeline shall compute and log validation metrics (L2 error vs. analytical solution) every M epochs
3. The Training Pipeline shall save model checkpoints at configurable intervals
4. When loss diverges (NaN or exceeds threshold), the Training Pipeline shall halt training and save diagnostic information
5. The Training Pipeline shall generate training curve plots (loss vs. epoch) upon completion
6. The Training Pipeline shall save final model weights and hyperparameters to output directory

### Requirement 8: FDTD Data Loading Utilities

**Objective:** As a physics simulation developer, I want utilities to load FDTD .npz data files, so that I can prepare data pipelines for future 2D PINN training.

#### Acceptance Criteria

1. The Data Loader shall read .npz files from `/PINN_data/` directory
2. The Data Loader shall extract spatiotemporal coordinates (x, y, t) from .npz files
3. The Data Loader shall extract wave field data (T1, T3, Ux, Uy) from .npz files
4. The Data Loader shall extract metadata (pitch, depth, seed) from .npz files
5. When loading multiple files, the Data Loader shall concatenate data across parameter combinations
6. The Data Loader shall validate data shapes and raise errors for malformed .npz files
7. The Data Loader shall support filtering by parameter ranges (e.g., pitch ∈ [1.25, 2.0] mm)
8. The Data Loader shall convert data to PyTorch tensors with configurable dtype (float32/float64)

### Requirement 9: Reproducibility and Configuration Management

**Objective:** As a physics simulation developer, I want reproducible experiments with version-controlled configurations, so that I can replicate results and track hyperparameter evolution.

#### Acceptance Criteria

1. The Training Pipeline shall accept configuration via YAML or JSON file
2. The Training Pipeline shall set random seeds for NumPy, PyTorch, and Python's random module
3. The Training Pipeline shall log full configuration (hyperparameters, seeds, versions) to experiment directory
4. The Training Pipeline shall record software versions (Python, PyTorch, DeepXDE) in metadata
5. When experiment is rerun with same config and seed, the Training Pipeline shall produce identical results
6. The Training Pipeline shall support experiment naming and output directory organization by timestamp

### Requirement 10: Code Quality and Testing

**Objective:** As a physics simulation developer, I want well-tested and documented code, so that the foundation is reliable for future 2D development.

#### Acceptance Criteria

1. The PINN Module shall include unit tests for PDE residual computation using pytest
2. The PINN Module shall include integration tests for end-to-end training on simple analytical case
3. The Data Loader shall include unit tests for .npz file parsing with fixture data
4. The code shall achieve minimum 70% test coverage measured by pytest-cov
5. The code shall pass Ruff linting with zero errors
6. The code shall include docstrings for all public functions following NumPy docstring convention
7. The code shall include type hints for function signatures where practical

## Non-Functional Requirements

### Performance
- Training on 1D wave equation (10k collocation points) shall complete in under 5 minutes on GPU
- Data loading for single .npz file (4.7 MB) shall complete in under 2 seconds

### Maintainability
- Code shall follow project structure conventions defined in `.kiro/steering/structure.md`
- All modules shall use import organization patterns from tech stack guidelines

### Compatibility
- Code shall run on Python 3.11+ with PyTorch 2.4.0 and CUDA 12.4
- Code shall be compatible with DeepXDE 1.15.0 API

### Documentation
- README shall include setup instructions, usage examples, and analytical validation results
- Jupyter notebook shall demonstrate 1D wave equation training and visualization workflow
