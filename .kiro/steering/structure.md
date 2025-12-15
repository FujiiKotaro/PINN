# Project Structure

## Organization Philosophy

**Reference-driven development**: Maintain `/ref/` with validated FDTD implementations, build new PINN modules in parallel. Data-centric workflow with `/PINN_data/` as ground truth source.

## Directory Patterns

### Reference Implementation
**Location**: `/ref/`
**Purpose**: Validated FDTD simulation code (CuPy-based), data generation utilities
**Example**: `PINN_FDTD3.py` (main FDTD loop), `sampling_utils.py` (LHS sampler), `sourse_new.py` (FFT/signal processing)
**Note**: Do not modify — serves as baseline for PINN validation

### Training Data
**Location**: `/PINN_data/`
**Purpose**: FDTD-generated `.npz` files with spatiotemporal wave field samples
**Naming**: `p{pitch_um}_d{depth_um}.npz` (e.g., `p1250_d100.npz` = 1.25mm pitch, 0.1mm depth)
**Contents**: Arrays (x, y, t, T1, T3, Ux, Uy) + metadata (pitch, depth, seed)

### PINN Modules (to be created)
**Location**: `/pinn/` (planned)
**Purpose**: DeepXDE-based PINN models, training scripts, evaluation utilities
**Example**: `pinn/models/wave_pinn.py`, `pinn/train.py`, `pinn/evaluate.py`

### Documentation
**Location**: Root directory
**Files**: `README.md` (project overview, 4-phase plan), `CLAUDE.md` (AI development guidelines)

## Naming Conventions

- **Files**: `snake_case.py` for Python modules (e.g., `sampling_utils.py`)
- **Classes**: `PascalCase` (e.g., `SpatioTemporalSampler`)
- **Functions**: `snake_case` (e.g., `sample_time_slices()`)
- **Data files**: `p{pitch}_d{depth}.npz` pattern for parametric sweep data

## Import Organization

```python
# Standard library
import os
from typing import Tuple

# Third-party (scientific stack)
import numpy as np
import cupy as cp
import torch

# DeepXDE and domain-specific
import deepxde as dde

# Local utilities
from sampling_utils import SpatioTemporalSampler
```

**Conventions**:
- Group imports: stdlib → third-party → local
- Use `cp` alias for CuPy (distinguishes from NumPy `np`)
- Avoid wildcard imports (`from module import *`)

## Code Organization Principles

### Physics Layer Separation
- **Data generation** (`/ref/`): FDTD simulations, ground truth creation
- **PINN modeling** (`/pinn/`): Neural network definitions, PDE constraints
- **Training/evaluation**: Separate scripts for training loops vs. inference

### GPU Memory Management
- **Explicit device control**: Use `cp.cuda.Device().synchronize()` after CuPy ops
- **Data transfer**: Minimize CPU↔GPU transfers (use `cp.asnumpy()` only when necessary)

### Reproducibility
- **Fixed seeds**: All sampling uses `seed=42` (LHS, train/test splits)
- **Metadata logging**: Save hyperparameters in `.npz` files alongside data

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
