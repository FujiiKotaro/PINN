"""Tests for 2D PINN model builder service.

Test-Driven Development: Tests written before implementation.
Tests cover 5D input, 4D output neural network construction for parametric PINN.
"""

import pytest
import torch

import deepxde as dde

from pinn.models.pinn_model_builder_2d import PINNModelBuilder2DService


class TestPINNModelBuilder2DNetwork:
    """Test 5D input → 4D output neural network construction."""

    def test_create_network_has_correct_input_dimension(self):
        """Test that network accepts 5D input (x, y, t, pitch, depth)."""
        builder = PINNModelBuilder2DService()

        # Configuration: [5 input, 64, 64, 64, 4 output]
        layer_sizes = [5, 64, 64, 64, 4]
        activation = "tanh"

        net = builder._create_network(layer_sizes, activation)

        # Verify network is created
        assert net is not None
        assert isinstance(net, dde.nn.pytorch.fnn.FNN)

        # Verify input dimension by checking first layer
        # FNN stores layers as net.linears (ModuleList)
        first_layer = net.linears[0]  # First Linear layer
        assert first_layer.in_features == 5, \
            f"Expected input dim 5, got {first_layer.in_features}"

    def test_create_network_has_correct_output_dimension(self):
        """Test that network outputs 4D (T1, T3, Ux, Uy)."""
        builder = PINNModelBuilder2DService()

        layer_sizes = [5, 64, 64, 64, 4]
        activation = "tanh"

        net = builder._create_network(layer_sizes, activation)

        # Verify output dimension by checking last layer
        last_layer = net.linears[-1]  # Last Linear layer
        assert last_layer.out_features == 4, \
            f"Expected output dim 4, got {last_layer.out_features}"

    def test_create_network_has_correct_hidden_layers(self):
        """Test that network has correct hidden layer architecture."""
        builder = PINNModelBuilder2DService()

        layer_sizes = [5, 64, 64, 64, 4]
        activation = "tanh"

        net = builder._create_network(layer_sizes, activation)

        # FNN has len(layer_sizes) - 1 Linear layers
        # [5, 64, 64, 64, 4] → 4 Linear layers (5→64, 64→64, 64→64, 64→4)
        num_linear_layers = len(net.linears)

        expected_linear_layers = len(layer_sizes) - 1
        assert num_linear_layers == expected_linear_layers, \
            f"Expected {expected_linear_layers} Linear layers, got {num_linear_layers}"

        # Verify hidden layer sizes
        for i, layer in enumerate(net.linears):
            expected_in = layer_sizes[i]
            expected_out = layer_sizes[i + 1]
            assert layer.in_features == expected_in, \
                f"Layer {i}: expected in_features={expected_in}, got {layer.in_features}"
            assert layer.out_features == expected_out, \
                f"Layer {i}: expected out_features={expected_out}, got {layer.out_features}"

    def test_create_network_uses_tanh_activation(self):
        """Test that network uses tanh activation function."""
        builder = PINNModelBuilder2DService()

        layer_sizes = [5, 64, 64, 64, 4]
        activation = "tanh"

        net = builder._create_network(layer_sizes, activation)

        # Check activation by inspecting net.activation attribute
        # DeepXDE FNN stores activation as a function object
        assert net.activation is not None, "Network should have activation function"
        assert callable(net.activation), "Activation should be callable"
        # Check function name contains 'tanh'
        activation_name = getattr(net.activation, '__name__', str(net.activation))
        assert 'tanh' in activation_name.lower(), \
            f"Expected tanh activation, got {activation_name}"

    def test_create_network_forward_pass(self):
        """Test that network can perform forward pass with 5D input."""
        builder = PINNModelBuilder2DService()

        layer_sizes = [5, 64, 64, 64, 4]
        activation = "tanh"

        net = builder._create_network(layer_sizes, activation)

        # Create sample 5D input: (N=10, 5) [x, y, t, pitch_norm, depth_norm]
        batch_size = 10
        x = torch.randn(batch_size, 5, dtype=torch.float32)

        # Forward pass
        y = net(x)

        # Verify output shape
        assert y.shape == (batch_size, 4), \
            f"Expected output shape (10, 4), got {y.shape}"

        # Verify output is not NaN
        assert not torch.isnan(y).any(), "Output contains NaN values"

    def test_create_network_glorot_initialization(self):
        """Test that network uses Glorot normal initialization."""
        builder = PINNModelBuilder2DService()

        layer_sizes = [5, 64, 64, 64, 4]
        activation = "tanh"

        net = builder._create_network(layer_sizes, activation)

        # Check that weights are initialized (not zero)
        first_layer = net.linears[0]
        weight = first_layer.weight.detach()

        # Glorot initialization should have mean ≈ 0, std > 0
        mean = weight.mean().item()
        std = weight.std().item()

        assert abs(mean) < 0.1, f"Weight mean should be near 0, got {mean}"
        assert std > 0.01, f"Weight std should be > 0, got {std}"
        assert not torch.allclose(weight, torch.zeros_like(weight)), \
            "Weights should not be all zeros (initialization failed)"

    def test_create_network_with_different_architecture(self):
        """Test network creation with different layer configuration."""
        builder = PINNModelBuilder2DService()

        # Test with smaller architecture [5, 32, 32, 4]
        layer_sizes = [5, 32, 32, 4]
        activation = "tanh"

        net = builder._create_network(layer_sizes, activation)

        # Verify input/output dimensions
        first_layer = net.linears[0]
        last_layer = net.linears[-1]

        assert first_layer.in_features == 5, "Input dim should be 5"
        assert last_layer.out_features == 4, "Output dim should be 4"

        # Verify forward pass
        x = torch.randn(5, 5, dtype=torch.float32)
        y = net(x)
        assert y.shape == (5, 4), "Output shape mismatch"
