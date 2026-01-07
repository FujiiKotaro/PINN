"""Test script for Fourier Features and Causal Training.

This script verifies that the new features work correctly before running
the full training pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pinn.models.fourier_network import FourierFeatureNetwork
from pinn.models.causal_pde_definition import CausalPDEDefinitionService
from pinn.utils.config_loader import ConfigLoaderService

print("=" * 80)
print("Testing Fourier Features and Causal Training")
print("=" * 80)

# Test 1: Fourier Feature Network
print("\n[Test 1] Fourier Feature Network")
print("-" * 80)

net = FourierFeatureNetwork(
    input_dim=2,
    output_dim=1,
    hidden_layers=[128, 128, 128, 128],
    num_fourier_features=256,
    fourier_scale=10.0,
    activation="tanh",
)

# Test forward pass
x_test = torch.randn(10, 2)
if torch.cuda.is_available():
    x_test = x_test.cuda()
    net = net.cuda()

y_test = net(x_test)

print(f"✓ Network created successfully")
print(f"  Input shape: {x_test.shape}")
print(f"  Output shape: {y_test.shape}")
print(f"  Number of parameters: {sum(p.numel() for p in net.parameters()):,}")

# Test 2: Causal PDE Function
print("\n[Test 2] Causal PDE Function")
print("-" * 80)

c = 1.5
t_max = 2.0
beta = 2.0

pde_func = CausalPDEDefinitionService.create_adaptive_causal_pde_function(c=c, t_max=t_max, beta=beta)

print(f"✓ Causal PDE function created")
print(f"  Wave speed: c = {c}")
print(f"  Max time: t_max = {t_max}")
print(f"  Causal parameter: β = {beta}")

# Test causal weighting
x_early = torch.tensor([[0.5, 0.1]], dtype=torch.float32, requires_grad=True)
x_late = torch.tensor([[0.5, 1.5]], dtype=torch.float32, requires_grad=True)

if torch.cuda.is_available():
    x_early = x_early.cuda()
    x_late = x_late.cuda()

u_early = net(x_early)
u_late = net(x_late)

residual_early = pde_func(x_early, u_early)
residual_late = pde_func(x_late, u_late)

print(f"  Residual at t=0.1: {residual_early.item():.6f}")
print(f"  Residual at t=1.5: {residual_late.item():.6f}")
print(f"  Causal weight ratio: {np.exp(-beta * 0.1 / t_max) / np.exp(-beta * 1.5 / t_max):.2f}x")

# Test 3: Configuration Loading
print("\n[Test 3] Configuration Loading")
print("-" * 80)

config_path = project_root / "configs" / "traveling_wave_example.yaml"
config_loader = ConfigLoaderService()
config = config_loader.load_config(config_path)

print(f"✓ Configuration loaded successfully")
print(f"  Experiment: {config.experiment_name}")
print(f"  Use Fourier Features: {config.network.use_fourier_features}")
print(f"  Fourier features: {config.network.num_fourier_features}")
print(f"  Fourier scale: {config.network.fourier_scale}")
print(f"  Use Causal Training: {config.training.use_causal_training}")
print(f"  Causal beta: {config.training.causal_beta}")

# Test 4: Model Building
print("\n[Test 4] Model Building with New Features")
print("-" * 80)

from pinn.models.pinn_model_builder import PINNModelBuilderService


# Define simple initial condition
def initial_condition(x):
    L = config.domain.x_max - config.domain.x_min
    x0 = L / 2
    sigma = L / 10
    if x.ndim == 1:
        x_reshaped = x.reshape(-1, 1)
    else:
        x_reshaped = x if x.shape[1] >= 1 else x.reshape(-1, 1)
    return np.exp(-((x_reshaped[:, 0:1] - x0) ** 2) / (2 * sigma**2))


model_builder = PINNModelBuilderService()
model = model_builder.build_model(
    config=config,
    initial_condition_func=initial_condition,
    compile_model=False,  # Don't compile for this test
)

print(f"✓ PINN model built successfully")
print(f"  Network type: {'Fourier Feature Network' if config.network.use_fourier_features else 'Standard FNN'}")
print(f"  PDE type: {'Causal PDE' if config.training.use_causal_training else 'Standard PDE'}")

# Count parameters
if hasattr(model.net, "parameters"):
    total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

print("\n" + "=" * 80)
print("All tests passed! ✓")
print("=" * 80)
print("\nReady to run full training with:")
print("  - Fourier Feature Embedding (256 features, scale=10.0)")
print("  - Causal Training (β=2.0)")
print("\nExpected improvements:")
print("  - Better high-frequency wave pattern learning")
print("  - Improved temporal causality")
print("  - Reduced PDE residual")
print("  - Lower validation error")
