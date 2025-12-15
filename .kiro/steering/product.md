# Product Overview

Physics-Informed Neural Network (PINN) simulation system for ultrasonic wave propagation in materials with surface defects. Replaces computationally expensive FDTD simulations with ML-accelerated forward models.

## Core Capabilities

- **Parametric crack modeling**: Simulate ultrasonic propagation for varying crack geometries (pitch: 1.25-2.0mm, depth: 0.1-0.3mm)
- **Physics-constrained learning**: Train neural networks that satisfy elastic wave PDEs (2D wave equation with boundary conditions)
- **FDTD data integration**: Load and train on spatiotemporal samples from reference FDTD simulations
- **Evaluation framework**: R² score comparison between PINN predictions and FDTD ground truth

## Target Use Cases

- **Non-destructive testing (NDT)**: Fast simulation of ultrasonic inspection scenarios for defect characterization
- **Surrogate modeling**: Replace expensive FDTD simulations (hours → minutes) for parameter studies
- **Design exploration**: Rapid evaluation of crack detection strategies across parameter spaces

## Value Proposition

- **Computational efficiency**: GPU-accelerated PINN inference vs. iterative FDTD time-stepping
- **Physics consistency**: Built-in PDE constraints ensure physically plausible predictions
- **Generalization**: Trained model can interpolate across unseen parameter combinations (pitch/depth pairs)

---
_Focus on patterns and purpose, not exhaustive feature lists_
