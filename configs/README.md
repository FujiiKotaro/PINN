# Configuration Files for 1D Wave Equation PINN

This directory contains YAML configuration files for different wave equation scenarios.

## Available Configurations

### 1. `standing_wave_example.yaml`
- **Boundary Condition**: Dirichlet (u=0 at boundaries)
- **Analytical Solution**: `u(x,t) = sin(nπx/L) cos(nπct/L)`
- **Use Case**: Clamped string, fundamental mode oscillation
- **Validation**: Enabled ✓

### 2. `dirichlet_bc_example.yaml`
- **Boundary Condition**: Dirichlet (u=0 at boundaries)
- **Analytical Solution**: `u(x,t) = sin(nπx/L) cos(nπct/L)`
- **Use Case**: Fixed boundary conditions, rigid wall reflections
- **Validation**: Enabled ✓

### 3. `neumann_bc_example.yaml`
- **Boundary Condition**: Neumann (∂u/∂x=0 at boundaries)
- **Analytical Solution**: `u(x,t) = cos(nπx/L) cos(nπct/L)`
- **Use Case**: Free boundaries, open-ended string
- **Validation**: Enabled ✓

### 4. `traveling_wave_example.yaml`
- **Boundary Condition**: Neumann (∂u/∂x=0 at boundaries)
- **Analytical Solution**: d'Alembert formula `u(x,t) = 0.5[f(x-ct) + f(x+ct)]`
- **Use Case**: Traveling wave with Gaussian pulse
- **Validation**: Disabled ⚠️

## Why is Validation Disabled for Traveling Wave?

The d'Alembert solution is derived for an **infinite domain** and does not account for boundary reflections. In a **finite domain with Neumann boundary conditions**:

1. The wave propagates according to d'Alembert initially
2. When the wave reaches the boundary (at time t ≈ L/(2c)), it reflects
3. After reflection, the d'Alembert solution **no longer satisfies the Neumann BC** (∂u/∂x = 0)
4. The PINN correctly learns the physics with boundary reflections
5. Comparing with d'Alembert solution produces high validation error (even though PINN is correct!)

**Solution**: Disable analytical validation for traveling wave. The PINN still learns the correct physics from:
- PDE residual minimization
- Boundary condition enforcement (Neumann BC)
- Initial condition fitting

To verify correctness, check that:
- PDE loss decreases to < 1e-3
- BC loss decreases to < 1e-3
- Wave propagation visually shows proper reflection at boundaries

## Modifying Configurations

Each YAML file contains:

```yaml
experiment_name: "name_of_experiment"
seed: 42

domain:
  x_min: 0.0
  x_max: 1.0
  t_min: 0.0
  t_max: 1.0
  wave_speed: 1.0

boundary_conditions:
  type: "dirichlet"  # or "neumann"
  left_value: null
  right_value: null

network:
  layer_sizes: [2, 50, 50, 50, 1]
  activation: "tanh"

training:
  epochs: 10000
  learning_rate: 0.001
  optimizer: "adam"
  loss_weights:
    data: 1.0
    pde: 1.0
    bc: 10.0
    ic: 10.0
  amp_enabled: true
  checkpoint_interval: 1000

analytical_solution:
  solution_type: "standing_wave"  # or "standing_wave_neumann", "traveling_wave"
  mode: 1
  initial_amplitude: 1.0
  enable_validation: true  # Set to false to disable analytical validation
```

## Tips for Hyperparameter Tuning

### Dirichlet BC (standing_wave_example.yaml, dirichlet_bc_example.yaml)
- BC weight: 10-50 (moderate to high)
- IC weight: 10-20 (moderate)
- Works well with standard Adam optimizer

### Neumann BC (neumann_bc_example.yaml)
- BC weight: 5-10 (moderate)
- IC weight: 20-500 (VERY HIGH - critical for anchoring solution!)
- Requires more epochs (12000+)
- Lower learning rate (1e-4) for stability
- Consider deeper network (4 hidden layers)

### Traveling Wave (traveling_wave_example.yaml)
- PDE weight: 2.0 (higher emphasis on physics)
- BC weight: 5.0
- IC weight: implicit through initial condition
- Disable analytical validation
- Monitor PDE and BC losses instead
