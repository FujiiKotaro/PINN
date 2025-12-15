import numpy as np
from typing import Dict

class DataLoader:
    """
    Loads data from .npz files.
    """
    def __init__(self, file_path: str):
        """
        Initializes the DataLoader with the path to a data file.

        Args:
            file_path: The path to the .npz file.
        """
        self.file_path = file_path
        self._data = None

    def get_data(self):
        """
        Loads data from the .npz file.
        
        For now, it returns the raw loaded numpy data container.
        """
        if self._data is None:
            self._data = np.load(self.file_path, allow_pickle=True)
        return self._data

    def get_physical_constants(self) -> Dict:
        """
        Returns physical constants derived from the data or hardcoded defaults.
        """
        # Constants from PINN_FDTD3.py
        E = 206 * 1e9
        rho = 7840
        V = 0.27
        
        c11 = E * (1 - V) / ((1 + V) * (1 - 2 * V))
        c13 = E * V / ((1 + V) * (1 - 2 * V))
        c55 = (c11 - c13) / 2
        
        data = self.get_data()
        
        # Check if x is available to determine L_char
        # Data might be flattened arrays, so we look at max/min
        if 'x' in data and data['x'].size > 0:
            x_max = np.max(data['x'])
        else:
            x_max = 0.02 # Default from FDTD script
            
        return {
            'rho': rho,
            'c11': c11,
            'c13': c13,
            'c55': c55,
            'x_max': x_max,
            'L_char': x_max, 
            'T_char': 1e-5, 
        }

    def get_true_params(self) -> Dict:
        """
        Returns the ground truth parameters from the data file.
        """
        data = self.get_data()
        return {
            'pitch': float(data['p']) if 'p' in data else 0.0,
            'depth': float(data['d']) if 'd' in data else 0.0,
            'width': float(data['w']) if 'w' in data else 0.0
        }

    def get_processed_data(self, physical_constants: Dict):
        """
        Loads and then non-dimensionalizes the data.

        Args:
            physical_constants: A dictionary of characteristic scales for the NonDimensionalizer.

        Returns:
            A dictionary of non-dimensionalized data.
        """
        raw_data = self.get_data()
        transformer = NonDimensionalizer(physical_constants)
        
        # This is a simplified transformation for now.
        # It assumes the raw_data is a dictionary-like object.
        data_to_transform = {
            'x': raw_data['x'],
            'y': raw_data['y'],
            't': raw_data['t'],
        }

        return transformer.transform_data(data_to_transform)

class NonDimensionalizer:
    """
    Converts physical data and parameters into a dimensionless system.
    """
    def __init__(self, physical_constants: Dict):
        """
        Initializes the transformer with characteristic scales.

        Args:
            physical_constants: A dictionary containing characteristic
                                values, e.g., {'L_char': 0.02, 'T_char': 1e-6}.
        """
        self.scales = physical_constants

    def transform_data(self, data: Dict) -> Dict:
        """
        Applies non-dimensionalization to the input data.
        
        Assumes keys in data correspond to dimensions in scales 
        (e.g., data['x'] uses 'L_char', data['t'] uses 'T_char').
        """
        dimensionless_data = {}
        if 'x' in data:
            dimensionless_data['x'] = data['x'] / self.scales['L_char']
        if 'y' in data:
            dimensionless_data['y'] = data['y'] / self.scales['L_char']
        if 't' in data:
            dimensionless_data['t'] = data['t'] / self.scales['T_char']
        # Add other transformations as needed
        return dimensionless_data

    def get_scales(self) -> Dict:
        """Returns the characteristic scales used for transformation."""
        return self.scales

