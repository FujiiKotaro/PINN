from typing import Dict, List, Any
import numpy as np

class ResultsAnalyzer:
    """
    Analyzes the results of the training.
    """
    def __init__(self, model: Any, inverse_vars: List[Any], true_params: Dict[str, float]):
        """
        Args:
            model: The trained DeepXDE model.
            inverse_vars: List of inverse variables (assumed order: pitch, depth).
            true_params: Dictionary of ground truth values {'pitch': ..., 'depth': ...}.
        """
        self.model = model
        self.inverse_vars = inverse_vars
        self.true_params = true_params

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculates the error between estimated and true parameters.
        
        Returns:
            Dictionary containing predicted values and relative errors.
        """
        est_values = []
        for var in self.inverse_vars:
            try:
                # PyTorch
                val = var.detach().cpu().item()
            except AttributeError:
                try:
                    # PyTorch without cpu() (if cpu) or other tensors
                    val = var.detach().item()
                except AttributeError:
                    # TensorFlow / NumPy / Others
                    try:
                        val = var.numpy()
                    except:
                        # Raw float?
                        val = float(var)
            est_values.append(val)
        
        # Assuming order is pitch, depth as per design/tasks
        est_pitch = est_values[0] if len(est_values) > 0 else 0.0
        est_depth = est_values[1] if len(est_values) > 1 else 0.0
        
        true_pitch = self.true_params.get('pitch', 1.0) # Avoid div by zero default
        true_depth = self.true_params.get('depth', 1.0)
        
        pitch_error = abs(est_pitch - true_pitch) / true_pitch if true_pitch != 0 else abs(est_pitch)
        depth_error = abs(est_depth - true_depth) / true_depth if true_depth != 0 else abs(est_depth)
        
        return {
            'pitch_pred': est_pitch,
            'depth_pred': est_depth,
            'pitch_error': pitch_error,
            'depth_error': depth_error
        }

    @staticmethod
    def calculate_r2(y_true: List[float], y_pred: List[float]) -> float:
        """
        Calculates the R2 score.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        return 1 - (ss_res / ss_tot)
