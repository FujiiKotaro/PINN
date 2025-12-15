# Technology Stack

## Architecture

Physics-Informed Neural Network (PINN) implementation using differential equation constraints. FDTD-generated training data stored as `.npz` files, loaded for spatiotemporal sampling.

## Core Technologies

- **Language**: Python 3.11+
- **Framework**: DeepXDE 1.15.0 (PINN framework built on PyTorch)
- **Runtime**: PyTorch 2.4.0 with CUDA 12.4 (GPU acceleration required)
- **Computation**: CuPy 13.3.0 for GPU-accelerated array operations

## Key Libraries

- **PyTorch**: Neural network backbone, automatic differentiation for PDE constraints
- **DeepXDE**: High-level PINN API (geometry, PDE definition, loss functions)
- **NumPy/CuPy**: Numerical operations (NumPy CPU fallback, CuPy GPU acceleration)
- **SciPy**: Latin Hypercube Sampling (LHS) for spatiotemporal data sampling
- **Matplotlib/Seaborn**: Visualization of wave fields and training metrics

## Development Standards

### Type Safety
- NumPy array shapes documented in docstrings
- Use type hints for function signatures where practical

### Code Quality
- Ruff for linting and formatting (configured in pyproject.toml)
- Follow PEP 8 conventions

### Testing
- pytest for unit tests
- Test fixtures for loading sample `.npz` data

## Development Environment

### Required Tools
- **CUDA Toolkit**: 12.4 (for GPU acceleration)
- **Poetry**: Dependency management
- **Python**: 3.11+

### Common Commands
```bash
# Setup: poetry install
# Run FDTD simulation: python ref/PINN_FDTD3.py
# Run tests: pytest
# Lint: ruff check .
```

## Key Technical Decisions

### Why DeepXDE over raw PyTorch?
- **Domain-specific API**: Built-in support for PDE residuals, boundary conditions, and physics-informed loss
- **Proven for PINNs**: Established framework with examples for wave equations

### Why CuPy for FDTD?
- **GPU acceleration**: FDTD time-stepping is memory-bound, CuPy enables 10-100x speedup vs. NumPy
- **Drop-in replacement**: Minimal code changes from NumPy to CuPy

### Data format: .npz (NumPy compressed)
- **Efficiency**: Stores spatiotemporal arrays (T1, T3, Ux, Uy) + metadata in single file
- **Compatibility**: Direct load with `np.load()`, no parsing overhead

### Sampling strategy: Latin Hypercube Sampling (LHS)
- **Space-filling**: Ensures uniform coverage of spatiotemporal domain
- **Reproducibility**: Fixed seed (42) for consistent train/val splits

---
_Document standards and patterns, not every dependency_
