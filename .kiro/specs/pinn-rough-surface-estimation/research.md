# Research & Design Decisions

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design for the PINN-based surface parameter estimation feature.

**Usage**:
- Log research activities and outcomes during the discovery phase.
- Document design decision trade-offs that are too detailed for `design.md`.
- Provide references and evidence for future audits or reuse.
---

## Summary
- **Feature**: `pinn-rough-surface-estimation`
- **Discovery Scope**: New Feature (Greenfield)
- **Key Findings**:
  - `deepxde` directly supports inverse problem solving by defining parameters as `dde.Variable` and including them in `external_trainable_variables`.
  - The elastic wave equation, being a system of coupled PDEs, can be implemented as a custom PDE in `deepxde` where the network predicts a multi-component output (e.g., `Ux`, `Uy`).
  - Non-dimensionalization of the PDE system is a critical manual preprocessing step to ensure training stability and is not handled automatically by `deepxde`.
  - The reference FDTD code (`PINN_FDTD3.py`) contains the exact logic for defining the complex geometry of the rough surface, which can be translated into a boundary condition function for the PINN.

## Research Log

### Topic: `deepxde` for Inverse Problems
- **Context**: The core requirement is to estimate `pitch` and `depth` from reflection data, which is an inverse problem.
- **Sources Consulted**: `deepxde` documentation, official examples.
- **Findings**: The library provides a clear pattern: define unknown scalar values as `dde.Variable`, use them within the PDE definition, and pass them to `model.compile(external_trainable_variables=...)`. Observational data for the loss function is supplied via `dde.icbc.PointSetBC`.
- **Implications**: This confirms `deepxde` is a suitable framework. The design must include a component that manages these `dde.Variable` objects.

### Topic: Custom PDE for 2D Elastic Wave Equation
- **Context**: The physics are governed by a 2D elastic wave equation, which is a system of coupled second-order PDEs.
- **Sources Consulted**: `deepxde` documentation on custom PDEs, analysis of `PINN_FDTD3.py`.
- **Findings**: `deepxde` handles custom PDEs through a Python function returning the equation's residual. For a system, the network output `y` will be a vector (e.g., `(Ux, Uy)`), and the PDE function will return a list of residuals, one for each equation in the system. Derivatives like `∂Ux/∂t`, `∂²Ux/∂x²` are computed with `dde.grad.jacobian` and `dde.grad.hessian`.
- **Implications**: The design needs a component that correctly defines the multi-output network and the corresponding system of PDE residuals.

### Topic: Non-Dimensionalization Strategy
- **Context**: The `README.md` and FDTD code highlight that physical constants are very large, which can lead to numerical instability during training.
- **Sources Consulted**: General PINN best practices, `deepxde` community discussions.
- **Findings**: Non-dimensionalization is a standard technique to improve PINN training. It requires selecting characteristic scales (e.g., `L_char`, `T_char`, `sigma_char`) and reformulating the PDE and variables into a dimensionless form. This must be done manually before the problem is defined in `deepxde`.
- **Implications**: The design must include a data-handling or preprocessing component that is responsible for applying these transformations. The characteristic values can be derived from the constants in `PINN_FDTD3.py`.

## Architecture Pattern Evaluation
- **Option 1: Monolithic Script**: A single, large Python script containing all logic (data loading, model definition, training).
  - **Strengths**: Simple to write initially.
  - **Risks**: Poor reusability, difficult to test, hard to maintain and modify.
- **Option 2: Component-based Python Pipeline (Selected)**: A main script that orchestrates several distinct components (e.g., classes or modules) with clear responsibilities.
  - **Strengths**: Modular, testable, reusable, and easier to understand. Aligns with software engineering best practices.
  - **Risks**: Slightly more upfront effort to define the structure.
  - **Notes**: The selected architecture will consist of components for Data Loading, PINN Model Definition, Inverse Problem Configuration, and Training.

## Design Decisions

### Decision: Adopt Non-Dimensionalization
- **Context**: The physical parameters (Young's modulus, density) in the wave equation have vastly different scales, leading to a stiff PDE and potential training failure due to exploding/vanishing gradients.
- **Alternatives Considered**:
  1. Use raw physical values and rely on adaptive loss weighting. (Risk: High chance of training failure).
  2. Manual non-dimensionalization. (Risk: Requires careful implementation but is a proven technique).
- **Selected Approach**: Manually non-dimensionalize the entire system (PDE, variables, boundary conditions) before feeding it into `deepxde`.
- **Rationale**: This is a standard and recommended practice for training PINNs on problems with challenging physical scales. It promotes stability and faster convergence.
- **Trade-offs**: Adds a layer of complexity in preprocessing, but significantly de-risks the core training phase.

## Risks & Mitigations
- **Risk 1**: The inverse problem may not converge to the correct `pitch` and `depth` values or may get stuck in local minima.
  - **Mitigation**: Start with a good initial guess if possible. Employ a learning rate schedule. Consider using a hybrid optimization approach (e.g., Adam followed by L-BFGS), which `deepxde` supports.
- **Risk 2**: Tuning the weights of the different loss components (`L_data`, `L_pde`, `L_bc`) can be difficult and time-consuming.
  - **Mitigation**: Start with equal weights and observe the magnitude of each loss term. `deepxde` has features for automatic loss balancing that can be explored if manual tuning fails.
- **Risk 3**: Training may be very slow, especially with a large number of collocation points.
  - **Mitigation**: Start with a smaller number of points to verify the implementation. Leverage GPU acceleration. Ensure the non-dimensionalization is correctly implemented, as this can also improve convergence speed.

## References
- DeepXDE Documentation: [https://deepxde.readthedocs.io/](https://deepxde.readthedocs.io/)
- `PINN_FDTD3.py` (for physics and boundary logic)
