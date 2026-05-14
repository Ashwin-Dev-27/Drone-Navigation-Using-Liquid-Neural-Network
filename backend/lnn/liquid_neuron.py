"""
Closed-form Continuous-time (CfC) Neuron — Core LNN Building Block
===================================================================
CfC neurons are the heart of Liquid Neural Networks. Unlike standard RNNs,
they model continuous-time dynamics, making them ideal for physical systems
like drones where time matters.

Key property: x(t) = sigmoid(-f(x,I)*t) * x0 + (1 - sigmoid(-f(x,I)*t)) * g(x,I)
This gives the neuron a "liquid" behavior — it continuously flows toward
the input-driven attractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time Cell.
    
    A single recurrent cell that models time-varying dynamics.
    The state evolves according to a learned ODE that has a closed-form solution.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units (liquid state dimension)
        mode: 'default' uses full CfC, 'pure' uses pure gating
    """
    
    def __init__(self, input_size: int, hidden_size: int, mode: str = 'default'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        
        # Backbone: maps input+state → intermediate representation
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Time-constant network: determines how fast the neuron responds
        # Lower tau → faster response (good for sudden wind gusts)
        self.tau_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Softplus()  # ensures positive time constants
        )
        
        # Attractor network: where the neuron wants to go
        self.attractor_net = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Forward pass implementing CfC dynamics.
        
        Args:
            x: Input tensor [batch, input_size]
            h: Hidden state [batch, hidden_size]
            dt: Time step (represents real time elapsed, e.g., 0.05s for 20Hz)
        
        Returns:
            New hidden state [batch, hidden_size]
        """
        combined = torch.cat([x, h], dim=-1)
        
        # Compute time constants (how fast the neuron reacts)
        tau = self.tau_net(combined) + 0.001  # avoid zero
        
        # Compute the attractor (desired state given current input)
        attractor = torch.tanh(self.attractor_net(combined))
        
        # CfC closed-form update:
        # h_new = h * exp(-dt/tau) + attractor * (1 - exp(-dt/tau))
        # This is the exact solution to: dh/dt = -(h - attractor) / tau
        decay = torch.exp(-dt / tau)
        h_new = h * decay + attractor * (1.0 - decay)
        
        # Apply output projection with residual
        out = self.output_proj(h_new) + h_new
        return torch.tanh(out)


class LiquidNetwork(nn.Module):
    """
    Full Liquid Neural Network for drone control.
    
    Multiple CfC cells arranged in layers with wiring (sparse connections).
    
    Args:
        input_size: Drone state dimensions (position + velocity + orientation + sensors)
        hidden_size: Number of liquid neurons per layer
        output_size: Control commands (4 rotor thrusts)
        num_layers: Depth of the liquid network
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Input encoding: project raw sensor data into liquid space
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )
        
        # Stack of CfC cells
        self.cells = nn.ModuleList([
            CfCCell(
                input_size=hidden_size if i > 0 else hidden_size,
                hidden_size=hidden_size
            )
            for i in range(num_layers)
        ])
        
        # Output decoder: liquid state → motor commands
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()  # outputs in [0,1] representing throttle percentage
        )
    
    def forward(self, x: torch.Tensor, hidden_states: list = None, dt: float = 0.05):
        """
        Forward pass through all liquid layers.
        
        Args:
            x: Input [batch, input_size]
            hidden_states: List of hidden states for each layer (or None for zeros)
            dt: Time step
        
        Returns:
            (output, new_hidden_states)
        """
        batch_size = x.size(0)
        
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        
        # Encode input
        h = self.input_encoder(x)
        
        # Pass through liquid layers
        new_hidden = []
        for i, cell in enumerate(self.cells):
            h_new = cell(h, hidden_states[i], dt=dt)
            new_hidden.append(h_new)
            h = h_new
        
        # Decode to motor commands
        output = self.output_decoder(h)
        
        return output, new_hidden
    
    def get_param_count(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    model = LiquidNetwork(input_size=12, hidden_size=64, output_size=4, num_layers=2)
    print(f"LNN Parameters: {model.get_param_count():,}")
    
    x = torch.randn(1, 12)
    output, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape} (4 rotor thrusts)")
    print(f"Output values (throttle %): {output.detach().numpy()}")
