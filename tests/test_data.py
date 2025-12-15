import pytest
import os
import numpy as np
from src.data import DataLoader, NonDimensionalizer

# --- DataLoader Tests ---

def test_dataloader_creation():
    """
    Tests that the DataLoader can be instantiated.
    """
    file_path = "tests/mock_data.npz"
    assert os.path.exists(file_path)
    loader = DataLoader(file_path)
    assert loader is not None

def test_dataloader_loads_mock_data():
    """
    Tests that the DataLoader can successfully load data from a mock .npz file.
    """
    file_path = "tests/mock_data.npz"
    loader = DataLoader(file_path)
    
    data = loader.get_data()
    
    assert data is not None
    assert "x" in data
    assert "y" in data
    assert np.array_equal(data["x"], np.array([1, 2, 3]))

# --- NonDimensionalizer Tests ---

def test_nondimensionalizer_creation():
    """
    Tests that the NonDimensionalizer can be instantiated.
    """
    physical_constants = {
        'L_char': 0.02, # Characteristic length
        'T_char': 1e-6, # Characteristic time
    }
    transformer = NonDimensionalizer(physical_constants)
    assert transformer is not None

def test_nondimensionalizer_scales_correctly():
    """
    Tests that the NonDimensionalizer correctly scales a value.
    """
    physical_constants = {
        'L_char': 0.02,
        'T_char': 1e-6,
    }
    transformer = NonDimensionalizer(physical_constants)
    
    # Mock data to transform
    data_to_transform = {'x': 0.01, 't': 2e-6} # e.g., a length and time value
    
    dimensionless_data = transformer.transform_data(data_to_transform)
    
    assert 'x' in dimensionless_data
    assert 't' in dimensionless_data
    # Expected: 0.01 / 0.02 = 0.5
    assert dimensionless_data['x'] == pytest.approx(0.5)
    # Expected: 2e-6 / 1e-6 = 2.0
    assert dimensionless_data['t'] == pytest.approx(2.0)

def test_get_scales():
    """
    Tests that the get_scales method returns the correct scales.
    """
    physical_constants = {
        'L_char': 0.02,
        'T_char': 1e-6,
    }
    transformer = NonDimensionalizer(physical_constants)
    scales = transformer.get_scales()
    assert scales['L_char'] == 0.02

# --- Integration Tests ---

def test_integration_loads_and_transforms_real_data():
    """
    Tests the integration of DataLoader and NonDimensionalizer using a real data file.
    """
    file_path = "PINN_data/p1250_d100.npz"
    loader = DataLoader(file_path)

    # These would be derived from the problem, but are hardcoded for the test
    physical_constants = {
        'L_char': 0.04,  # y_length from PINN_FDTD3.py
        'T_char': 1e-5,
    }

    processed_data = loader.get_processed_data(physical_constants)
    raw_data = loader.get_data()

    # Check that the processed data is different from the raw data
    assert processed_data is not None
    assert 'x' in processed_data
    assert processed_data['x'][0] != raw_data['x'][0]
    
    # Check that the scaling is correct
    assert processed_data['x'][0] == pytest.approx(raw_data['x'][0] / physical_constants['L_char'])
