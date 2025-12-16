# Research & Design Decisions Template

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design.

**Usage**:
- Log research activities and outcomes during the discovery phase.
- Document design decision trade-offs that are too detailed for `design.md`.
- Provide references and evidence for future audits or reuse.
---

## Summary
- **Feature**: `pinn-2d-fdtd-integration`
- **Discovery Scope**: Complex Integration (new 2D elastic wave PINN architecture + FDTD data integration + parametric learning)
- **Key Findings**:
  - DeepXDE supports 2D spatiotemporal domains via Rectangle + GeometryXTime composition
  - Multi-output PINNs require separate neural network outputs for each field (T1, T3, Ux, Uy) with shared hidden layers
  - Conditional PINN achieved by extending input dimension to include parameters (pitch, depth)
  - R² scoring via sklearn.metrics.r2_score provides standardized validation metric
  - Phase 1 infrastructure (data loaders, callbacks, config management) fully reusable

## Research Log
Document notable investigation steps and their outcomes. Group entries by topic for readability.

### DeepXDE 2D Geometry and Spatiotemporal Domain Construction

- **Context**: Requirements demand spatial domain [0, 40mm] × [0, 20mm] and temporal domain [3.5e-6s, 6.5e-6s]. Need to understand how to construct 2D+time geometry in DeepXDE.
- **Sources Consulted**:
  - [DeepXDE Geometry Module Documentation](https://deepxde.readthedocs.io/en/latest/modules/deepxde.geometry.html)
  - [DeepXDE GitHub Repository](https://github.com/lululxvi/deepxde)
- **Findings**:
  - `dde.geometry.Rectangle(xmin, xmax)` creates 2D rectangular spatial domain where xmin=[x0, y0], xmax=[x1, y1]
  - `dde.geometry.TimeDomain(t_min, t_max)` creates temporal interval
  - `dde.geometry.GeometryXTime(spatial_geom, timedomain)` combines them into 3D (x, y, t) spatiotemporal domain
  - GeometryXTime provides `random_points()`, `random_boundary_points()`, `random_initial_points()` for collocation point sampling
  - Example: `Rectangle([0, 0], [0.04, 0.02])` creates [0-40mm, 0-20mm] spatial domain
- **Implications**: Direct replacement of Phase 1's `Interval` (1D) with `Rectangle` (2D). Same GeometryXTime pattern applies, requiring minimal changes to PINNModelBuilderService geometry creation logic.

### Multi-Output PINN Architecture for Coupled Stress-Displacement Fields

- **Context**: Need to predict 4 output fields (T1, T3, Ux, Uy) simultaneously from single spatiotemporal input (x, y, t).
- **Sources Consulted**:
  - [PINN-elastodynamics GitHub](https://github.com/Raocp/PINN-elastodynamics) - elasticity problems with displacement and stress
  - [Scientific ML for SAW Propagation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902327/) - multi-field PINN approaches
  - [PINNs for Wave Propagation](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023120)
- **Findings**:
  - Multi-output PINNs typically use single feedforward network with multiple output neurons (one per field)
  - Layer configuration: `[input_dim, hidden1, hidden2, ..., output_dim]` where output_dim=4 for (T1, T3, Ux, Uy)
  - DeepXDE's `dde.nn.FNN` supports multi-output by setting final layer size to number of outputs
  - Separate neural networks per field is alternative but increases parameter count and training complexity (rejected for Phase 2)
  - Shared hidden layers enforce physical coupling between stress and displacement fields through PDE residual loss
- **Implications**: Change Phase 1's `layer_sizes: [2, 50, 50, 50, 1]` to `layer_sizes: [3, 64, 64, 64, 4]` (3 inputs: x,y,t → 4 outputs: T1,T3,Ux,Uy). PDE residual must compute derivatives for all 4 outputs separately.

### Conditional PINN for Parametric Learning (Pitch and Depth)

- **Context**: Requirement 3 demands learning across crack parameter space (pitch: 1.25-2.0mm, depth: 0.1-0.3mm) with interpolation capability.
- **Sources Consulted**:
  - [DeepXDE Research Applications](https://deepxde.readthedocs.io/en/latest/user/research.html)
  - [DeepXDE SIAM Review Paper](https://epubs.siam.org/doi/10.1137/19M1274067)
  - General parametric PINN literature search
- **Findings**:
  - Conditional PINN: treat parameters as additional network inputs → input = (x, y, t, pitch, depth)
  - Input dimension increases from 3 to 5: `layer_sizes: [5, 64, 64, 64, 4]`
  - Parameters normalized to [0, 1] range for stable training (pitch: 1.25-2.0mm → 0-1, depth: 0.1-0.3mm → 0-1)
  - Network learns mapping f(x, y, t, p, d) → (T1, T3, Ux, Uy)
  - Training: randomly sample from all 12 FDTD files (4 pitch × 3 depth) for each training batch
  - Validation: hold out 1-2 parameter combinations (e.g., p=1.625mm, d=0.15mm) to test interpolation
- **Implications**:
  - FDTDDataLoaderService must extract pitch/depth from filename and append to input tensors
  - Requires normalization service for parameter scaling
  - Cannot reuse Phase 1 training pipeline without modification (input dimension change)

### 2D Elastic Wave Equation PDE Residual Formulation

- **Context**: Requirement 1 specifies 2D elastic wave equations with longitudinal and transverse modes.
- **Sources Consulted**:
  - [Wave Equation Modeling PINN Sensors Paper](https://www.mdpi.com/1424-8220/23/5/2792)
  - [Solving Wave Equation with Physics-Informed DL](https://arxiv.org/abs/2006.11894)
  - Elastic wave theory textbooks (Achenbach 1973 - domain knowledge)
- **Findings**:
  - Longitudinal wave (P-wave): ∂²u/∂t² = (λ+2μ)/ρ (∂²u/∂x² + ∂²u/∂y²)
  - Transverse wave (S-wave): ∂²v/∂t² = μ/ρ (∂²v/∂x² + ∂²v/∂y²)
  - For displacement-stress formulation, need constitutive relations and momentum balance
  - Simplification for isotropic medium: T1 = (λ+2μ)εxx + λεyy, T3 = (λ+2μ)εyy + λεxx
  - Strain-displacement: εxx = ∂Ux/∂x, εyy = ∂Uy/∂y
  - DeepXDE automatic differentiation: `dde.grad.hessian(y, x, i, j)` computes ∂²y/∂xi∂xj
- **Implications**:
  - PDE residual must return 4 equations (one per output field)
  - Material constants (λ, μ, ρ) from steering context: Al 6061 typical values λ=58GPa, μ=26GPa, ρ=2700kg/m³
  - PDEDefinitionService.create_pde_function() requires major refactor from 1D to 2D coupled system

### FDTD Data Sampling and Train/Validation Split Strategy

- **Context**: Requirement 2 specifies using 12 FDTD .npz files (~56MB total) for training. Need efficient sampling and validation split.
- **Sources Consulted**:
  - Existing Phase 1 implementation: `pinn/data/fdtd_loader.py`, `pinn/utils/seed_manager.py`
  - [DeepXDE 1D Wave Example](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/wave_1d.py)
- **Findings**:
  - Current FDTD files already contain LHS-sampled points (not full grid) → nx_sample, ny_sample, nt_sample in metadata
  - Each file contains ~nt*nx*ny flattened samples (typical: 30 time steps × 200x100 spatial = ~600k points per file)
  - Memory-efficient loading: load all 12 files, concatenate (x, y, t, pitch, depth, T1, T3, Ux, Uy) arrays
  - Train/val split: 80/20 random split with seed=42 for reproducibility
  - Stratification by parameter combinations not required (uniform coverage from LHS sampling)
- **Implications**:
  - FDTDDataLoaderService.load_multiple_files() method to combine datasets
  - Return PyTorch DataLoader with shuffling for mini-batch training
  - Validation set used for early stopping and R² score computation

### R² Score Implementation for Validation

- **Context**: Requirement 4 demands R² score (coefficient of determination) for FDTD comparison.
- **Sources Consulted**:
  - [sklearn.metrics.r2_score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
  - [R² vs MAE/MSE comparison study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279135/)
- **Findings**:
  - R² formula: R² = 1 - Σ(y_pred - y_true)² / Σ(y_true - ȳ)²
  - Best score: 1.0 (perfect prediction), can be negative if model worse than mean baseline
  - sklearn implementation: `sklearn.metrics.r2_score(y_true, y_pred, multioutput='raw_values')` returns per-output R²
  - For 4 outputs (T1, T3, Ux, Uy), compute separate R² scores to identify which field has poor fit
  - R² > 0.9 considered excellent, 0.7-0.9 good, <0.5 poor (domain-specific thresholds)
- **Implications**:
  - Add r2_score() method to ErrorMetricsService (extend Phase 1)
  - Import sklearn as optional dependency (add to pyproject.toml)
  - Validation callback logs per-field R² scores during training

### GPU Memory and Performance Considerations for 2D PINN

- **Context**: Non-functional requirement specifies 30-minute training time for 10k collocation points, 5k epochs on GPU.
- **Sources Consulted**:
  - Phase 1 benchmarks: 1D PINN with 2540 points, 10k epochs = ~3.5 minutes
  - [PINNs for Wave Propagation - Performance Analysis](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023120)
- **Findings**:
  - 2D PINN complexity: O(N_points × N_outputs × N_derivatives) where N_derivatives scales with spatial dimensions
  - 1D: 2 spatial derivatives (∂x, ∂xx) vs 2D: 6 derivatives (∂x, ∂y, ∂xx, ∂yy, ∂xy) → 3× autodiff overhead
  - Multi-output (4 fields): 4× forward passes + PDE residual for each field
  - Combined scaling: ~12× vs Phase 1 for same collocation point count
  - 10k collocation points feasible with AMP enabled, expected training time: 10-15 minutes on CUDA 12.4
  - Potential bottleneck: FDTD data loading (12 files × ~4.5MB each) → use lazy loading or caching
- **Implications**:
  - Keep num_domain=10000 as specified in requirements
  - Enable AMP by default (already in Phase 1)
  - Add dataset caching to avoid repeated .npz file reads
  - Monitor GPU memory in CheckpointCallback (existing Phase 1 component)

## Architecture Pattern Evaluation

List candidate patterns or approaches that were considered. Use the table format where helpful.

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Single Multi-Output Network | One FNN with 4 output neurons for (T1, T3, Ux, Uy) | Shared hidden layers enforce physical coupling, fewer parameters, aligned with DeepXDE design patterns | Harder to tune individual field losses, potential gradient conflicts between outputs | **SELECTED**: Aligns with Phase 1 architecture, proven in PINN-elastodynamics literature |
| Separate Networks per Field | Four independent FNNs, each predicting one output | Independent loss tuning, easier to debug per-field issues | 4× parameter count, no built-in physical coupling, violates Phase 1 simplicity principle | Rejected: overkill for Phase 2 scope |
| Encoder-Decoder Architecture | Shared encoder (x,y,t,p,d) → latent space → 4 decoders | Flexible latent representation, modular design | Increased complexity, no clear benefit for physics-informed learning | Rejected: premature optimization |
| Physics-Guided Network Splitting | Separate networks for displacement (Ux, Uy) and stress (T1, T3) | Mirrors physics structure (kinematics vs dynamics) | Requires custom PDE coupling layer, breaks DeepXDE compatibility | Rejected: incompatible with steering principle "reuse Phase 1 infrastructure" |

**Selected Pattern Rationale**: Single multi-output FNN maximizes Phase 1 code reuse (PINNModelBuilderService requires only layer_sizes change), aligns with DeepXDE examples, and enforces stress-displacement coupling through shared PDE residual loss. Proven effective in elastic wave PINN literature (Rao et al. 2021, PINN-elastodynamics).

## Design Decisions
Record major decisions that influence `design.md`. Focus on choices with significant trade-offs.

### Decision: Conditional PINN Input Augmentation vs. Parameter-Specific Models

- **Context**: Requirement 3 demands parametric learning across 12 crack geometry combinations (4 pitch × 3 depth). Need to decide: single model with parameters as inputs OR 12 separate models.
- **Alternatives Considered**:
  1. **Conditional PINN (Augmented Input)**: Input = (x, y, t, pitch, depth), train single model on all 12 datasets
  2. **Separate Models**: Train 12 independent PINNs, one per parameter combination
  3. **Hierarchical Model**: Pre-trained base model + parameter-specific fine-tuning layers
- **Selected Approach**: Conditional PINN with 5D input (x, y, t, normalized_pitch, normalized_depth)
- **Rationale**:
  - Single model deployment: simpler inference pipeline, no model selection logic
  - Interpolation capability: network learns continuous parameter space, can predict unseen (pitch, depth) combinations within training range
  - Data efficiency: shared representation across parameter space, reduces overfitting vs. 12 separate models
  - Aligns with requirement 3.4: "PINN Model shall perform interpolation for untrained parameter combinations"
- **Trade-offs**:
  - **Benefits**: Unified model, interpolation capability, efficient parameter space coverage
  - **Compromises**: Larger input dimension (5 vs 3) increases network complexity, potential gradient interference between parameter variations
- **Follow-up**: Validate interpolation accuracy by holding out p=1.625mm, d=0.15mm during training and measuring R² on this holdout combination.

### Decision: Direct FDTD Data Supervision vs. Physics-Only Training

- **Context**: Requirement 2 provides FDTD ground truth data. PINNs can be trained purely on PDE residuals (unsupervised) or with data loss (supervised).
- **Alternatives Considered**:
  1. **Data + PDE Loss (Hybrid)**: L_total = w_data × L_data + w_pde × L_pde + w_bc × L_bc
  2. **PDE-Only**: L_total = w_pde × L_pde + w_bc × L_bc (ignore FDTD data during training, use only for validation)
  3. **Data-Only**: L_total = L_data (standard supervised learning, no physics constraints)
- **Selected Approach**: Hybrid loss with tunable weights (w_data, w_pde, w_bc), default w_data=1.0, w_pde=1.0, w_bc=10.0
- **Rationale**:
  - FDTD data provides strong supervisory signal for accurate field prediction (addresses Requirement 4: R² > 0.9 target)
  - PDE loss enforces physical consistency in regions not covered by FDTD samples (generalization beyond training data)
  - Hybrid approach proven in Phase 1: relative error <2% on analytical solutions
  - Flexible loss weighting via Phase 1's WeightTuningFrameworkService for domain-specific tuning
- **Trade-offs**:
  - **Benefits**: Best of both worlds (data accuracy + physics consistency), proven Phase 1 approach
  - **Compromises**: Requires FDTD data loading (10s overhead), three hyperparameters to tune (w_data, w_pde, w_bc)
- **Follow-up**: Run grid search on loss weights if initial R² < 0.9 threshold (Requirement 4.5).

### Decision: 2D Geometry Boundary Conditions - Dirichlet vs. Absorbing

- **Context**: Requirement 1.2 specifies spatial domain [0, 40mm] × [0, 20mm]. Need to define boundary conditions at x=0, x=40, y=0, y=20.
- **Alternatives Considered**:
  1. **Zero Dirichlet BC**: u = 0 on all boundaries (fixed boundaries)
  2. **Absorbing/PML BC**: Perfectly matched layer to prevent reflections
  3. **Free Surface BC**: Zero traction (stress-free boundaries)
  4. **No Explicit BC**: Rely on FDTD data at boundaries (implicit BC from data loss)
- **Selected Approach**: No explicit analytical BC, rely on FDTD data supervision at boundary points (option 4)
- **Rationale**:
  - FDTD simulation already handles boundary conditions correctly (absorbing boundaries in reference implementation)
  - FDTD data includes boundary samples → data loss L_data provides implicit BC supervision
  - Avoids mismatch: FDTD uses absorbing BC, enforcing Dirichlet (u=0) would contradict training data
  - Simplifies implementation: no need to derive 2D absorbing BC formulations for elastic waves
- **Trade-offs**:
  - **Benefits**: Consistent with FDTD ground truth, simpler design, no BC formulation research required
  - **Compromises**: Less physics-informed (no explicit BC in PDE loss), extrapolation to boundaries may be less accurate if FDTD sampling sparse at edges
- **Follow-up**: Verify FDTD data includes sufficient boundary samples. If R² poor at boundaries, add explicit free surface BC.

### Decision: R² Score Threshold and Warning Strategy

- **Context**: Requirement 4.5 specifies "If R² < 0.9, warn user and recommend hyperparameter tuning".
- **Alternatives Considered**:
  1. **Per-Field Thresholds**: Different R² targets for stress (T1, T3) vs displacement (Ux, Uy)
  2. **Aggregate Threshold**: Average R² across 4 fields must exceed 0.9
  3. **Strict Threshold**: ALL fields must individually exceed 0.9
  4. **Adaptive Threshold**: Start with 0.7 target, increase to 0.9 over training
- **Selected Approach**: Per-field strict threshold (option 3) - each of (T1, T3, Ux, Uy) must achieve R² ≥ 0.9
- **Rationale**:
  - Field-specific failure diagnosis: identifies which outputs need improvement (e.g., "T1 R²=0.85, Ux R²=0.92" pinpoints stress field issue)
  - Requirement 4.3 explicitly states "compute R² for each output field individually"
  - Prevents masking: aggregate metric could hide poor performance in one field
  - Aligns with validation workflow: per-field error heatmaps (Requirement 4.6)
- **Trade-offs**:
  - **Benefits**: Precise diagnostics, ensures all fields meet quality standard
  - **Compromises**: Stricter than aggregate (harder to pass), may trigger false alarms if one field inherently harder to learn
- **Follow-up**: Log per-field R² in ValidationCallback, emit warning if ANY field < 0.9, suggest field-specific loss weight tuning.

## Risks & Mitigations

- **Risk 1: Multi-output gradient conflicts** — During backpropagation, gradients from 4 output fields may conflict, causing training instability
  - **Mitigation**: Use per-field loss weights (w_T1, w_T3, w_Ux, w_Uy) to balance gradients; Phase 1's WeightTuningFrameworkService can optimize these

- **Risk 2: Parameter interpolation failure** — Network may not generalize to untrained (pitch, depth) combinations if parameter space coverage insufficient
  - **Mitigation**: Ensure 12 FDTD files span parameter space uniformly; validate on holdout combination (e.g., p=1.625, d=0.15); if R² < 0.7, request additional FDTD data

- **Risk 3: FDTD data sparsity** — Current .npz files use LHS sampling (not full grid), may have insufficient points in critical regions (e.g., crack vicinity)
  - **Mitigation**: Requirement 2.6 acknowledges current sampling density; design system to accept denser data in future; focus on interpolation within existing data, not extrapolation

- **Risk 4: GPU memory overflow** — 2D PINN with 10k points × 4 outputs × 5 epochs may exceed CUDA memory (typical GPU: 8-16GB)
  - **Mitigation**: Enable AMP (Phase 1 AMPWrapperService), reduce batch size if OOM occurs, add gradient checkpointing if necessary

- **Risk 5: Training time exceeds 30-minute target** — 2D complexity (~12× scaling vs 1D) may push training beyond performance requirement
  - **Mitigation**: Profile early training epochs, reduce num_domain if time > 30min, consider distributed training (future Phase 4 scope)

- **Risk 6: Test-implementation divergence (Phase 1 lesson)** — Requirement 6 highlights past issues with test/code misalignment
  - **Mitigation**: Focus tests on critical paths (PDE residual, FDTD loader, R² score); validate with notebook execution; prioritize integration test over unit mocks

## References
Provide canonical links and citations (official docs, standards, ADRs, internal guidelines).

- [DeepXDE Documentation](https://deepxde.readthedocs.io/) — Official framework docs for geometry, PDE definition, network architecture
- [DeepXDE GitHub Repository](https://github.com/lululxvi/deepxde) — Source code and examples (wave_1d.py reference)
- [DeepXDE SIAM Review Paper](https://epubs.siam.org/doi/10.1137/19M1274067) — Lu et al. 2021 - foundational framework paper
- [PINN-elastodynamics](https://github.com/Raocp/PINN-elastodynamics) — Rao et al. implementation for stress-displacement coupling
- [PINNs for Wave Propagation](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023120) — Smith et al. 2021 - elastic wave PINN performance benchmarks
- [Wave Equation Modeling PINN](https://www.mdpi.com/1424-8220/23/5/2792) — Sensors 2023 - soft/hard constraints for boundary conditions
- [Scientific ML for SAW Propagation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902327/) — Multi-field PINN approaches
- [sklearn R² Score Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) — Implementation reference for coefficient of determination
- [R² vs MAE/MSE Comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279135/) — Justification for R² as primary validation metric
- [Achenbach 1973 - Wave Propagation in Elastic Solids] — Classical reference for elastic wave theory (domain knowledge)
