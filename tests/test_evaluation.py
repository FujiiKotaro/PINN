import unittest
from unittest.mock import MagicMock
import numpy as np
from src.evaluation import ResultsAnalyzer

class TestResultsAnalyzer(unittest.TestCase):
    def test_parameter_evaluation(self):
        """
        Test that ResultsAnalyzer correctly extracts parameters and calculates errors.
        """
        # Mock dde.Model and Variables
        mock_model = MagicMock()
        
        # Mock variables with current value
        # In deepxde, variables are usually tensors or stored in the model. 
        # But here we probably pass the variables list to the analyzer or extract them.
        # The design says "takes the trained model, inspects the final values of the pitch and depth variables".
        # We need to know how to get them. 
        # Assuming we pass the variable objects (which have .numpy() or similar, or we access their backend value)
        
        # Let's assume ResultsAnalyzer takes a list of variables or a dict.
        # For simple testing, let's pass the VALUES directly or mock the extraction.
        # If the class is responsible for extraction, we need to mock dde.Variable.
        
        # Let's assume the constructor takes the model and the inverse_variables list.
        
        mock_pitch = MagicMock()
        mock_pitch.name = "pitch" # verify naming if needed
        # Mocking the value extraction. DeepXDE variables are often tf.Variable or torch.Tensor.
        # We usually access them via model.sess.run(var) in TF or var.item() in Torch.
        # Let's assume we can pass the *values* or the Analyzer knows how to extract.
        # To make it backend agnostic, maybe we pass the variables and the analyzer calls `backend.to_numpy(var)`.
        
        # For simplicity, let's assume the Analyzer takes a dictionary of {name: value} or we pass the variable objects and mock `item()`.
        mock_pitch.detach.return_value.cpu.return_value.item.return_value = 1.5 # Predicted pitch
        
        mock_depth = MagicMock()
        mock_depth.detach.return_value.cpu.return_value.item.return_value = 0.2 # Predicted depth
        
        inverse_vars = [mock_pitch, mock_depth]
        
        true_params = {'pitch': 1.5, 'depth': 0.2} # Perfect match
        
        analyzer = ResultsAnalyzer(mock_model, inverse_vars, true_params)
        
        metrics = analyzer.calculate_metrics()
        
        self.assertAlmostEqual(metrics['pitch_error'], 0.0)
        self.assertAlmostEqual(metrics['depth_error'], 0.0)
        
        # Test with error
        mock_pitch.detach.return_value.cpu.return_value.item.return_value = 1.6 # Error
        analyzer = ResultsAnalyzer(mock_model, inverse_vars, true_params)
        metrics = analyzer.calculate_metrics()
        self.assertNotEqual(metrics['pitch_error'], 0.0)

    def test_r2_score(self):
        # Test the r2 calculation logic
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        r2 = ResultsAnalyzer.calculate_r2(y_true, y_pred)
        self.assertEqual(r2, 1.0)
        
        y_pred_bad = [10.0, 20.0, 30.0]
        r2_bad = ResultsAnalyzer.calculate_r2(y_true, y_pred_bad)
        self.assertLess(r2_bad, 0)

if __name__ == '__main__':
    unittest.main()
