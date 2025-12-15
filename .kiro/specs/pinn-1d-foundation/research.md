# Research & Design Decisions

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design for the 1D PINN foundation feature.

**Usage**:
- Log research activities and outcomes during the discovery phase.
- Document design decision trade-offs that are too detailed for `design.md`.
- Provide references and evidence for future audits or reuse.
---

## Summary
- **Feature**: `pinn-1d-foundation`
- **Discovery Scope**: New Feature (greenfield PINN implementation)
- **Key Findings**:
  - DeepXDE 1.15.0 provides comprehensive 1D wave equation support with TimeDomain geometry and multiple boundary condition types
  - Adaptive loss weight balancing methods (ReLoBRaLo, gradient-based) significantly improve training performance over fixed weights
  - PyTorch AMP provides 2-3X speedup on CUDA GPUs with minimal code changes (3 lines)

## Research Log

### DeepXDE 1D Wave Equation Implementation
- **Context**: Need to validate DeepXDE API for 1D wave equation PDE definition and boundary conditions
- **Sources Consulted**:
  - [DeepXDE wave_1d.py example](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/wave_1d.py)
  - [FAU DCN PINNs Wave Equation Repository](https://github.com/DCN-FAU-AvH/PINNs_wave_equation)
  - [DeepXDE geometry.timedomain API](https://deepxde.readthedocs.io/en/latest/_modules/deepxde/geometry/timedomain.html)
  - [DeepXDE icbc boundary conditions](https://deepxde.readthedocs.io/en/latest/_modules/deepxde/icbc/boundary_conditions.html)
- **Findings**:
  - Official wave_1d.py example demonstrates ∂²u/∂t² = c²∂²u/∂x² implementation using DeepXDE
  - `dde.geometry.Interval()` + `dde.geometry.TimeDomain()` → `dde.geometry.GeometryXTime()` for spatiotemporal domain
  - Supports 5 BC types: Dirichlet (`DirichletBC`), Neumann (`NeumannBC`), Robin (`RobinBC`), periodic, and general operator BC
  - Initial conditions enforced via `IC` class with separate functions for u(x,0) and ∂u/∂t(x,0)
  - PDE residual defined as Python function using automatic differentiation (`dde.grad.jacobian`, `dde.grad.hessian`)
- **Implications**:
  - DeepXDE API is mature and well-suited for 1D wave validation phase
  - No custom PDE framework needed—can leverage built-in `dde.data.TimePDE` class
  - Analytical solution validation straightforward using FAU DCN reference implementation patterns

### Loss Function Weight Tuning Strategies
- **Context**: Requirements 3 and 6 mandate tunable loss weights (w_data, w_pde, w_bc) with automated tuning framework
- **Sources Consulted**:
  - [Self-adaptive loss balanced PINNs (ReLoBRaLo)](https://www.sciencedirect.com/science/article/abs/pii/S092523122200546X)
  - [Dynamic Weight Strategy for Navier-Stokes](https://pmc.ncbi.nlm.nih.gov/articles/PMC9497516/)
  - [Multi-Objective Loss Balancing](https://www.sciencedirect.com/science/article/pii/S0045782525001860)
  - [Adaptive Loss Balancing Tutorial](https://towardsdatascience.com/improving-pinns-through-adaptive-loss-balancing-55662759e701/)
- **Findings**:
  - **Challenge**: Different loss terms (L_data, L_pde, L_bc) have varying magnitudes → competing gradients during backpropagation
  - **Adaptive methods**: ReLoBRaLo (Relative Loss Balancing with Random Lookback) improves error by >10X vs. fixed weights
  - **Gradient-based balancing**: Automatically tunes weights to equalize back-propagated gradient magnitudes across loss components
  - **Alternative**: Reformulation methods eliminate need for weight tuning by incorporating BCs into network architecture (hard constraints)
- **Implications**:
  - Phase 1 should implement **fixed weights first** (manual tuning baseline) for Requirements 3
  - Requirement 6 tuning framework should support:
    1. Grid search (baseline)
    2. Gradient magnitude tracking for manual analysis
    3. (Future) Adaptive methods like ReLoBRaLo for Phase 2/3
  - Log individual loss components + gradient norms for diagnostic purposes

### PyTorch AMP for GPU Acceleration
- **Context**: Requirement 5 mandates GPU-accelerated training with memory efficiency for future 2D models
- **Sources Consulted**:
  - [PyTorch AMP Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
  - [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html)
  - [Mixed Precision Training Guide](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
  - [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- **Findings**:
  - **Performance**: 2-3X speedup on Volta/Turing/Ampere GPUs (CUDA 12.4 compatible)
  - **Memory**: Lower precision (float16) reduces memory footprint → larger batch sizes possible
  - **Implementation**: 3-line integration:
    ```python
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        # forward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    ```
  - **Stability**: GradScaler prevents gradient underflow by dynamic loss scaling
  - **Compatibility**: DeepXDE uses PyTorch backend → AMP directly compatible
- **Implications**:
  - Enable AMP by default for GPU training with CPU fallback
  - Log GPU memory usage to validate memory reduction claims
  - Test convergence with AMP vs. full precision to ensure no accuracy degradation

### FDTD Data Loading Utilities Design
- **Context**: Requirement 8 mandates .npz loader for future 2D PINN training (Phase 2/3)
- **Sources Consulted**: Existing `/ref/sampling_utils.py`, `/PINN_data/*.npz` file inspection
- **Findings**:
  - Existing `SpatioTemporalSampler` in `/ref/sampling_utils.py` provides LHS sampling logic (reusable)
  - `.npz` format: `np.load()` returns dict-like object with keys: `x, y, t, T1, T3, Ux, Uy, p, d, w, seed`
  - Data shapes: (nt*nx*ny,) flattened arrays → need reshaping to (nt, nx, ny) for 2D training
  - Metadata: pitch (p), depth (d), width (w), seed for reproducibility tracking
- **Implications**:
  - **Phase 1 scope**: Implement .npz loader but NOT used for 1D training (only analytical solutions)
  - **Reusability**: Design loader to return PyTorch tensors with configurable dtype (float32/float64)
  - **Validation**: Check array shapes, metadata presence, and raise errors for malformed files
  - **Future**: Loader will be critical for Phase 2 (2D PINN) to consume FDTD ground truth

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Modular Functional | Separate modules for model, training, validation, data loading | Clean separation of concerns, easy unit testing | May over-abstract for simple 1D case | Aligns with steering `/pinn/` structure |
| Monolithic Script | Single script with all logic (like `/ref/PINN_FDTD3.py`) | Fast to implement, fewer files | Hard to test, difficult to extend to 2D | Not recommended for foundation phase |
| DeepXDE Native | Use DeepXDE's built-in `Model.train()` and `Model.compile()` | Minimal boilerplate, framework-native | Less control over training loop, harder to add custom logging | Recommended for Phase 1 simplicity |

**Selected**: DeepXDE Native + Modular Utilities (hybrid)
- Core PINN model uses `dde.Model.compile()` and `dde.Model.train()` for simplicity
- Separate utility modules for: analytical solutions, visualization, data loading, config management
- Custom callbacks for logging loss components and GPU metrics

## Design Decisions

### Decision: DeepXDE Native Training Loop vs. Custom PyTorch Training
- **Context**: Requirements 3, 5, 7 mandate custom loss logging, GPU control, and checkpoint saving
- **Alternatives Considered**:
  1. **DeepXDE Native** — Use `dde.Model.compile(optimizer, loss, loss_weights)` + `dde.Model.train()`
  2. **Custom PyTorch Loop** — Manually implement training loop with full control over loss computation and logging
- **Selected Approach**: DeepXDE Native with custom callbacks
- **Rationale**:
  - DeepXDE handles PDE residual computation, BC enforcement, and collocation point sampling automatically
  - Custom callbacks (`dde.callbacks.Callback`) allow logging loss components without rewriting training loop
  - Phase 1 goal is framework validation, not custom PINN infrastructure
- **Trade-offs**:
  - **Benefits**: Faster implementation, fewer bugs, leverages DeepXDE's optimized backend
  - **Compromises**: Less flexibility for exotic training schemes (acceptable for 1D validation phase)
- **Follow-up**: If Phase 2 (2D PINN) requires advanced training (e.g., curriculum learning), revisit custom loop

### Decision: Fixed Loss Weights (Phase 1) with Manual Tuning Framework
- **Context**: Requirement 6 mandates automated weight tuning framework
- **Alternatives Considered**:
  1. **Adaptive methods (ReLoBRaLo)** — Implement state-of-the-art adaptive weight balancing
  2. **Grid/Random Search** — Exhaustive hyperparameter search over weight ranges
  3. **Manual tuning + gradient logging** — Provide tools for manual analysis, defer automation to Phase 2
- **Selected Approach**: Grid/Random Search (Requirement 6) + Gradient Logging for Phase 1
- **Rationale**:
  - Phase 1 scope is infrastructure validation, not optimal PINN training
  - Grid search is simple, reproducible, and provides baseline for future adaptive methods
  - Logging gradient magnitudes enables manual analysis of weight balance quality
- **Trade-offs**:
  - **Benefits**: Simple implementation, clear baseline metrics, educational value
  - **Compromises**: Computationally expensive for large grids (acceptable for 1D with fast training)
- **Follow-up**: Integrate ReLoBRaLo or gradient-based adaptive weighting in Phase 2 for 2D models

### Decision: Configuration Management via YAML
- **Context**: Requirement 9 mandates reproducible experiments with version-controlled configs
- **Alternatives Considered**:
  1. **Python dataclasses** — Type-safe config with IDE support
  2. **YAML files** — Human-readable, version-control friendly
  3. **JSON files** — Machine-readable, less human-friendly
- **Selected Approach**: YAML with Pydantic validation
- **Rationale**:
  - YAML is human-readable for manual editing (e.g., `wave_speed: 1.0`, `domain: [0, 1]`)
  - Pydantic provides runtime validation and type safety (best of both worlds)
  - Standard in ML research community (Hydra, OmegaConf patterns)
- **Trade-offs**:
  - **Benefits**: Easy manual edits, git diff-friendly, Pydantic catches errors early
  - **Compromises**: Requires YAML parsing library (PyYAML, already in Python stdlib)
- **Follow-up**: Consider Hydra for Phase 2 if multi-run experiments become complex

## Risks & Mitigations

### Risk 1: DeepXDE 1.15.0 API Incompatibility
- **Description**: Documentation found for 1.10.1 and 1.14.1, not exact version 1.15.0
- **Mitigation**:
  - Verify API stability by running official wave_1d.py example on local environment
  - If breaking changes found, use 1.14.1 (last documented stable version)
  - Document actual API usage in code comments for future reference

### Risk 2: AMP Convergence Degradation
- **Description**: Mixed precision may cause numerical instability in PDE residual computation
- **Mitigation**:
  - Implement convergence comparison test: full precision (float32) vs. AMP (float16)
  - If AMP causes >5% relative error increase, provide CPU-only fallback mode
  - Log loss divergence warnings and auto-disable AMP if NaN detected

### Risk 3: Grid Search Computational Cost
- **Description**: Exhaustive grid search over (w_data, w_pde, w_bc) may be slow for large grids
- **Mitigation**:
  - Start with coarse grid (3×3×3 = 27 configs), refine best region
  - Support random search for larger search spaces
  - Log early stopping if validation error plateaus (save compute time)

### Risk 4: .npz Data Loader Not Validated Until Phase 2
- **Description**: Requirement 8 loader won't be used in Phase 1 → untested until 2D PINN
- **Mitigation**:
  - Write comprehensive unit tests with fixture .npz files (extract from `/PINN_data/`)
  - Test edge cases: missing keys, malformed shapes, corrupted files
  - Validate loader in integration test (load → convert to tensor → basic shape checks)

## References

### DeepXDE Documentation
- [DeepXDE wave_1d.py Official Example](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/wave_1d.py) — Reference implementation for 1D wave equation
- [FAU DCN PINNs Wave Equation](https://github.com/DCN-FAU-AvH/PINNs_wave_equation) — Analytical validation examples with Neumann BCs
- [DeepXDE TimeDomain Geometry](https://deepxde.readthedocs.io/en/latest/_modules/deepxde/geometry/timedomain.html) — API for spatiotemporal domains
- [DeepXDE Boundary Conditions](https://deepxde.readthedocs.io/en/latest/_modules/deepxde/icbc/boundary_conditions.html) — Dirichlet, Neumann, Robin BC classes

### PINN Loss Function Tuning
- [ReLoBRaLo: Self-adaptive Loss Balancing](https://www.sciencedirect.com/science/article/abs/pii/S092523122200546X) — State-of-the-art adaptive weight method
- [Dynamic Weight Strategy (Navier-Stokes)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9497516/) — Gradient-based balancing approach
- [Adaptive Loss Balancing Tutorial](https://towardsdatascience.com/improving-pinns-through-adaptive-loss-balancing-55662759e701/) — Practical guide to implementation

### PyTorch AMP
- [PyTorch AMP Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — Official implementation guide
- [PyTorch AMP API](https://docs.pytorch.org/docs/stable/amp.html) — `torch.cuda.amp.autocast` and `GradScaler` documentation
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — Hardware-level optimization guide
