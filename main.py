import argparse
import numpy as np
import os
import deepxde as dde
from src.data import DataLoader
from src.pinn import PINNModelBuilder, InverseProblemSolver
from src.training import Trainer
from src.evaluation import ResultsAnalyzer


def main():
    parser = argparse.ArgumentParser(description="PINN for Rough Surface Estimation")
    parser.add_argument("--data", type=str, default="PINN_data/p1250_d100.npz", help="Path to .npz data file")
    parser.add_argument("--epochs", type=int, default=1000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print(f"Starting PINN Rough Surface Estimation...")

    # 1. Load Data
    if not os.path.exists(args.data):
        print(f"Warning: Data file {args.data} not found. Using default/dummy constants.")
        # Proceed with caution or generate dummy data logic
        # For now, we'll try to instantiate DataLoader anyway as it might handle it or we expect the user to provide valid path

    try:
        loader = DataLoader(args.data)
        phys_consts = loader.get_physical_constants()
        true_params = loader.get_true_params()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Prepare Dimensions & Initial Guesses
    L_char = phys_consts["L_char"]
    T_char = phys_consts["T_char"]

    print(f"Physical Constants: L_char={L_char}, T_char={T_char}")

    # Initial guess for inverse parameters (using true values perturbed or fixed typical values)
    # Pitch: 1.25-2.0 mm. Depth: 0.1-0.3 mm.
    init_pitch = 1.5e-3
    init_depth = 0.2e-3

    # Calculate dimensionless PDE coefficients
    # Effective velocity squared in dimensionless units
    # v_p^2 = c11/rho. v_dim^2 = v_p^2 * (T/L)^2
    scale_factor = (T_char / L_char) ** 2 / phys_consts["rho"]

    # Note: FDTD y_length is 0.04m usually.
    y_max = 0.04

    dim_params = {
        "rho": 1.0,
        "c11": phys_consts["c11"] * scale_factor,
        "c13": phys_consts["c13"] * scale_factor,
        "c55": phys_consts["c55"] * scale_factor,
        "x_domain": [0, phys_consts["x_max"] / L_char],
        "y_domain": [0, y_max / L_char],
        "t_domain": [0, 1.0],
        # Geometry params for ModelBuilder (Static Boundary)
        # Using initial guess to define the domain mesh
        "f_pitch": init_pitch / L_char,
        "f_depth": init_depth / L_char,
        "f_width": (phys_consts.get("width", 0.25e-3) if phys_consts.get("width") else 0.25e-3) / L_char,
    }

    # 3. Build Model
    print("Building PINN model...")
    builder = PINNModelBuilder(dim_params)

    # 4. Setup Inverse Problem
    print("Setting up Inverse Problem...")
    solver = InverseProblemSolver(initial_pitch=init_pitch / L_char, initial_depth=init_depth / L_char)

    # 5. Train
    print("Starting training setup...")
    inverse_vars = solver.get_inverse_variables()

    geom = builder.get_geometry()
    pde = builder.define_pde_system
    bcs = builder.define_boundary_conditions()

    # Add observation BCs
    data = loader.get_data()
    if "reflection" in data and "t_probe" in data:
        print("Adding observation data...")
        reflection = data["reflection"]
        t_probe = data["t_probe"]

        # Probe location: x=0, y=y_mid (0.02)
        y_probe_phys = 0.02  # Center of y domain
        y_probe_dim = y_probe_phys / L_char
        x_probe_dim = 0.0

        t_dim = t_probe / T_char

        # Filter to domain bounds
        mask = (t_dim >= 0) & (t_dim <= dim_params["t_domain"][1])

        if np.sum(mask) > 0:
            t_selected = t_dim[mask]
            vals_selected = reflection[mask]

            # Subsample to reduce computational cost
            skip = max(1, len(t_selected) // 200)  # Aim for ~200 points
            t_selected = t_selected[::skip]
            vals_selected = vals_selected[::skip]

            points = np.zeros((len(t_selected), 3))
            points[:, 0] = x_probe_dim
            points[:, 1] = y_probe_dim
            points[:, 2] = t_selected

            values = vals_selected.reshape(-1, 1)

            # Note: We assign reflection data to component 0 (u).
            # In reality, reflection is stress, so this assumes u ~ stress or requires model adjustment.
            obs_bc = solver.define_observation_bc(points, values, component=0)
            bcs.append(obs_bc)
        else:
            print("Warning: No observation data within t_domain. Check T_char scaling.")

    data_pde = dde.data.TimePDE(geom, pde, bcs, num_domain=1000, num_boundary=100, anchors=None)

    trainer = Trainer(data_pde, inverse_vars)
    trainer.compile(lr=args.lr)

    print(f"Training for {args.epochs} iterations...")
    trainer.train(epochs=args.epochs)

    # 6. Evaluate
    print("Evaluating results...")
    analyzer = ResultsAnalyzer(trainer.model, inverse_vars, true_params)
    metrics = analyzer.calculate_metrics()

    # Convert dimensionless results back to physical
    print("\n=== Results ===")
    print(f"Estimated Pitch: {metrics['pitch_pred'] * L_char * 1000:.4f} mm")
    print(f"Estimated Depth: {metrics['depth_pred'] * L_char * 1000:.4f} mm")

    if true_params["pitch"] > 0:
        print(f"True Pitch:      {true_params['pitch'] * 1000:.4f} mm")
        print(f"True Depth:      {true_params['depth'] * 1000:.4f} mm")
        print(f"Pitch Error:     {metrics['pitch_error']:.2%}")
        print(f"Depth Error:     {metrics['depth_error']:.2%}")


if __name__ == "__main__":
    main()
