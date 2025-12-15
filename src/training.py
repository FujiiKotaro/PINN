import deepxde as dde
from typing import List

class Trainer:
    """
    Compiles and runs the training process.
    """
    def __init__(self, data_obj: dde.data.TimePDE, inverse_vars: List[dde.Variable], layer_size: List[int] = None):
        """
        Initializes the Trainer.

        Args:
            data_obj: The DeepXDE data object (e.g., TimePDE).
            inverse_vars: List of variables to be inverted (e.g., pitch, depth).
            layer_size: Network architecture [input_dim, hidden1, ..., output_dim]. 
                        Defaults to [3, 20, 20, 20, 2].
        """
        if layer_size is None:
            # Default architecture: 3 inputs (x,y,t), 2 outputs (u,v), 3 hidden layers of 20
            layer_size = [3] + [20] * 3 + [2]
            
        self.net = dde.maps.FNN(layer_size, "tanh", "Glorot normal")
        self.model = dde.Model(data_obj, self.net)
        self.inverse_vars = inverse_vars

    def compile(self, optimizer: str = "adam", lr: float = 0.001):
        """
        Compiles the model with the specified optimizer and learning rate.
        Sets external_trainable_variables for inverse problems.
        """
        self.model.compile(optimizer, lr=lr, external_trainable_variables=self.inverse_vars)

    def train(self, epochs: int):
        """
        Runs the training loop.

        Args:
            epochs: Number of iterations to train.
        """
        variable_callback = dde.callbacks.VariableValue(
            self.inverse_vars, period=100, filename="variables.dat"
        )
        
        self.model.train(iterations=epochs, callbacks=[variable_callback])
