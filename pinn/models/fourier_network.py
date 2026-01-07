"""Fourier Feature Network for improved PINN performance.

This module implements Fourier Feature Embeddings to enhance the neural network's
ability to learn high-frequency patterns in wave equations.

Reference:
    Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import numpy as np


class FourierFeatureEmbedding(nn.Module):
    """Fourier Feature Embedding layer using random Fourier features.

    Maps input coordinates [x, t] to higher-dimensional feature space using:
        γ(v) = [cos(2π B v), sin(2π B v)]

    where B is a random matrix sampled from Gaussian distribution.
    """

    def __init__(self, input_dim: int, num_features: int = 256, scale: float = 10.0):
        """Initialize Fourier Feature Embedding.

        Args:
            input_dim: Input dimension (2 for [x, t])
            num_features: Number of Fourier features (output will be 2 * num_features)
            scale: Standard deviation of Gaussian distribution for frequency matrix
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_features = num_features
        self.scale = scale

        # Random Fourier feature matrix B ~ N(0, scale^2)
        # Fixed during training (not trainable)
        B = torch.randn(input_dim, num_features) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Fourier features (batch_size, 2 * num_features)
        """
        # Compute 2π B x
        x_proj = 2 * np.pi * x @ self.B

        # Concatenate [cos(2π B x), sin(2π B x)]
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class FourierFeatureNetwork(nn.Module):
    """Neural network with Fourier Feature Embedding for PINN.

    Architecture:
        Input [x, t] → Fourier Features → MLP → Output u(x, t)
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: list[int] = [128, 128, 128, 128],
        num_fourier_features: int = 256,
        fourier_scale: float = 10.0,
        activation: str = "tanh"
    ):
        """Initialize Fourier Feature Network.

        Args:
            input_dim: Input dimension (2 for [x, t])
            output_dim: Output dimension (1 for u)
            hidden_layers: List of hidden layer sizes
            num_fourier_features: Number of Fourier features
            fourier_scale: Scale parameter for Fourier features
            activation: Activation function ("tanh", "relu", "gelu")
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # DeepXDE compatibility attributes
        self.regularizer = None
        self.kernel_initializer = "Glorot normal"
        self.activation_fn = activation

        # Fourier feature embedding
        self.fourier_embedding = FourierFeatureEmbedding(
            input_dim=input_dim,
            num_features=num_fourier_features,
            scale=fourier_scale
        )

        # After Fourier embedding, dimension becomes 2 * num_fourier_features
        fourier_output_dim = 2 * num_fourier_features

        # Build MLP layers
        layers = []
        prev_dim = fourier_output_dim

        # Activation function
        if activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights using Glorot normal (Xavier normal)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Glorot normal initialization."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Apply Fourier feature embedding
        x_fourier = self.fourier_embedding(x)

        # Apply MLP
        return self.mlp(x_fourier)


class DeepXDEFourierFeatureNetwork:
    """Wrapper to make FourierFeatureNetwork compatible with DeepXDE.

    DeepXDE expects a specific interface for neural networks.
    This wrapper provides compatibility while using PyTorch's FourierFeatureNetwork.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: list[int] = [128, 128, 128, 128],
        num_fourier_features: int = 256,
        fourier_scale: float = 10.0,
        activation: str = "tanh"
    ):
        """Initialize DeepXDE-compatible Fourier Feature Network.

        Args:
            input_dim: Input dimension (2 for [x, t])
            output_dim: Output dimension (1 for u)
            hidden_layers: List of hidden layer sizes
            num_fourier_features: Number of Fourier features
            fourier_scale: Scale parameter for Fourier features
            activation: Activation function ("tanh", "relu", "gelu")
        """
        self.net = FourierFeatureNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            num_fourier_features=num_fourier_features,
            fourier_scale=fourier_scale,
            activation=activation
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (DeepXDE compatibility).

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.net(x)

    def parameters(self):
        """Return network parameters (DeepXDE compatibility)."""
        return self.net.parameters()

    def state_dict(self):
        """Return state dictionary (DeepXDE compatibility)."""
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dictionary (DeepXDE compatibility)."""
        self.net.load_state_dict(state_dict)

    def train(self):
        """Set network to training mode (DeepXDE compatibility)."""
        self.net.train()

    def eval(self):
        """Set network to evaluation mode (DeepXDE compatibility)."""
        self.net.eval()
