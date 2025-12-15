import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to sys.path so we can import main if it's in root
sys.path.append(os.getcwd())

from main import main

class TestMainIntegration(unittest.TestCase):
    @patch('sys.argv', ['main.py', '--data', 'dummy.npz', '--epochs', '10'])
    @patch('main.DataLoader')
    @patch('main.PINNModelBuilder')
    @patch('main.InverseProblemSolver')
    @patch('main.Trainer')
    @patch('main.ResultsAnalyzer')
    def test_main_pipeline(self, MockAnalyzer, MockTrainer, MockSolver, MockBuilder, MockLoader):
        """
        Test that main function orchestrates the components correctly.
        """
        # Setup mocks
        mock_loader_instance = MockLoader.return_value
        mock_loader_instance.get_physical_constants.return_value = {
            'L_char': 1.0, 
            'T_char': 1.0,
            'rho': 1.0,
            'c11': 1.0,
            'c13': 1.0,
            'c55': 1.0,
            'x_max': 1.0,
            'width': 0.1
        }
        mock_loader_instance.get_data.return_value = {'reflection': []} # minimal data
        mock_loader_instance.get_true_params.return_value = {'pitch': 1.0, 'depth': 0.1}
        
        mock_builder_instance = MockBuilder.return_value
        
        mock_solver_instance = MockSolver.return_value
        mock_solver_instance.get_inverse_variables.return_value = []
        mock_solver_instance.create_data_object.return_value = MagicMock()
        
        mock_trainer_instance = MockTrainer.return_value
        
        mock_analyzer_instance = MockAnalyzer.return_value
        mock_analyzer_instance.calculate_metrics.return_value = {
            'pitch_pred': 1.0, 
            'depth_pred': 0.1, 
            'pitch_error': 0.0, 
            'depth_error': 0.0
        }
        
        # Run main
        # We need to ensure main doesn't crash if it expects command line args or specific files.
        # Main likely sets up hardcoded paths or reads args. 
        # For this test, we assume main uses default or we patch sys.argv if needed.
        # We'll see how main is implemented.
        
        main()
        
        # Verify call order
        MockLoader.assert_called()
        # MockNonDim.assert_called() # Removed
        MockBuilder.assert_called()
        MockSolver.assert_called()
        # MockTrainer.assert_called() # Check if Trainer is called
        # Note: In main.py, trainer is instantiated.
        
        # Verify trainer methods
        mock_trainer_instance.compile.assert_called()
        mock_trainer_instance.train.assert_called()
        MockAnalyzer.assert_called()
        mock_analyzer_instance.calculate_metrics.assert_called()

if __name__ == '__main__':
    unittest.main()
