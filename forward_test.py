import deepxde as dde
import numpy as np
import torch
from src.data import DataLoader
from src.pinn import PINNModelBuilder, is_on_rough_surface

def get_source_func(f, wn, duration):
    """
    Recreates the source waveform from the FDTD simulation.
    """
    def source_func(t):
        # The input t to this function will be a tensor.
        # We need to use torch operations.
        wave2 = (1 - torch.cos(2 * np.pi * f * t / wn)) / 2
        wave3 = torch.sin(2 * np.pi * f * t)
        
        waveform = wave2 * wave3
        # Apply the waveform only for the duration
        return waveform * (t < duration)

    return source_func

class CustomFNN(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(torch.nn.BatchNorm1d(layer_sizes[i+1]))
                self.layers.append(torch.nn.ReLU())
        # Xavier uniform initialization
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        
        self.regularizer = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def main():
    # 1. Load Data
    loader = DataLoader("PINN_data/p1250_d100.npz")
    raw_data = loader.get_data()
    
    # 2. Define Characteristic Scales and Non-dimensionalize
    # From PINN_FDTD3.py
    L_char = 0.04 # y_length
    T_char = 10e-6 # t_max
    # U_char = L_char / T_char for velocity
    # Stress_char = rho * U_char**2
    rho = 7840
    Stress_char = rho * (L_char / T_char)**2

    # These are the dimensional parameters from the FDTD sim
    fdtd_params = {
        'x_length': 0.02, 'y_length': 0.04, 't_max': 10e-6,
        'rho': rho, 'E': 206e9, 'G': 80e9, 'V': 0.27,
        'f_pitch': raw_data['p'].item(), 'f_width': raw_data['w'].item(), 'f_depth': 0, #raw_data['d'].item(),
        'cl': 5960, # approx, should be calculated
        'f': 4.7e6, 'dt': 6.7e-10, 'wn': 2.5
    }

    # Create dimensionless params for the builder
    # The PINN will operate in a normalized [0,1] domain for simplicity
    pinn_params = {
        'x_domain': [0, fdtd_params['x_length'] / L_char], 
        'y_domain': [0, fdtd_params['y_length'] / L_char], 
        't_domain': [0, fdtd_params['t_max'] / T_char],
        'rho': 1, # rho / rho
        'c11': fdtd_params['E'] * (1 - fdtd_params['V']) / ((1 + fdtd_params['V']) * (1 - 2 * fdtd_params['V'])) / Stress_char,
        'c13': fdtd_params['E'] * fdtd_params['V'] / ((1 + fdtd_params['V']) * (1 - 2 * fdtd_params['V'])) / Stress_char,
        'c55': fdtd_params['G'] / Stress_char,
        'f_pitch': fdtd_params['f_pitch'] / L_char,
        'f_width': fdtd_params['f_width'] / L_char,
        'f_depth': fdtd_params['f_depth'] / L_char,
    }

    # 3. Instantiate PINNModelBuilder
    builder = PINNModelBuilder(pinn_params)

    # 4. Define source wave function
    # The duration of the source wave is wn * n * dt = wn * (1/f/dt) * dt = wn / f
    source_duration = fdtd_params['wn'] / fdtd_params['f']
    source_func = get_source_func(fdtd_params['f'], fdtd_params['wn'], source_duration)
    
    # 5. Get Boundary Conditions
    bcs = builder.define_boundary_conditions(source_func=source_func)
    
    # 6. Define Initial Conditions (zero displacement and velocity at t=0)
    geom = builder.get_geometry()
    ic_u = dde.icbc.IC(geom, lambda x: 0, lambda _, on_initial: on_initial, component=0)
    ic_v = dde.icbc.IC(geom, lambda x: 0, lambda _, on_initial: on_initial, component=1)
    
    # Need to also set initial velocity to zero, which means du/dt = 0 and dv/dt = 0 at t=0
    ic_u_t = dde.icbc.OperatorBC(geom, lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=2), lambda x, on_initial: on_initial)
    ic_v_t = dde.icbc.OperatorBC(geom, lambda x, y, _: dde.grad.jacobian(y, x, i=1, j=2), lambda x, on_initial: on_initial)

    all_conditions = bcs + [ic_u, ic_v, ic_u_t, ic_v_t]
    
    # 7. Create TimePDE data object
    data = dde.data.TimePDE(
        geom,
        builder.define_pde_system,
        all_conditions,
        num_domain=1000,
        num_boundary=500,
        num_initial=250,
    )
    
    print("Successfully created TimePDE data object.")

    # 8. Define and compile the model
    layer_sizes = [3] + [100] * 6 + [2]
    net = CustomFNN(layer_sizes)
    model = dde.Model(data, net)
    
    # 9. Train the model
    model.compile("adam", lr=1e-4)
    model.train(epochs=2000) # Reduced epochs to avoid timeout
    
    # 10. Predict the reflection waveform
    probe_y_center = fdtd_params['y_length'] / 2
    
    t_pred = np.linspace(0, fdtd_params['t_max'], 200)
    x_pred = np.zeros_like(t_pred)
    y_pred = np.full_like(t_pred, probe_y_center)
    
    pred_points = np.vstack((x_pred, y_pred, t_pred)).T
    
    # Predict displacements u and v
    uv_predicted = model.predict(pred_points)

    # Manually compute stress_xx from the displacement outputs
    pred_points_torch = torch.tensor(pred_points, dtype=torch.float32, requires_grad=True)
    
    # Can't compute gradient of a non-leaf tensor.
    # We need to re-run the model on the points that require gradients.
    uv_predicted_grad = model.net(pred_points_torch)
    
    u_x = dde.grad.jacobian(uv_predicted_grad, pred_points_torch, i=0, j=0)
    v_y = dde.grad.jacobian(uv_predicted_grad, pred_points_torch, i=1, j=1)

    c11_dim = fdtd_params['E'] * (1 - fdtd_params['V']) / ((1 + fdtd_params['V']) * (1 - 2 * fdtd_params['V']))
    c13_dim = fdtd_params['E'] * fdtd_params['V'] / ((1 + fdtd_params['V']) * (1 - 2 * fdtd_params['V']))
    
    predicted_stress_xx_dimensionless = (pinn_params['c11'] * u_x + pinn_params['c13'] * v_y)
    predicted_stress_xx = predicted_stress_xx_dimensionless.detach().cpu().numpy() * Stress_char

    # 11. Plot and save the results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(raw_data['t_probe'], raw_data['reflection'], label="Ground Truth (FDTD)")
    plt.plot(t_pred, predicted_stress_xx, label="PINN Prediction", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Stress (T1 / sigma_xx)")
    plt.title("Forward Problem: Predicted vs. Ground Truth Reflection")
    plt.legend()
    plt.grid(True)
    plt.savefig("forward_test_comparison.png")
    print("Saved plot to forward_test_comparison.png")

if __name__ == "__main__":
    main()

