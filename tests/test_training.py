import unittest
from unittest.mock import MagicMock, patch
import deepxde as dde
from src.training import Trainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.mock_data = MagicMock(spec=dde.data.TimePDE)
        self.mock_variable = MagicMock(spec=dde.Variable)
        self.mock_variable.dtype = "float32" # Mock dtype attribute usually required

    def test_trainer_initialization_and_compile(self):
        """
        Test that Trainer initializes correctly and compiles the model.
        """
        # We need to mock dde.Model because it does heavy lifting in __init__ usually, 
        # or we check if Trainer.compile calls model.compile.
        
        with patch('src.training.dde.Model') as MockModel, \
             patch('src.training.dde.maps.FNN') as MockFNN:
            
            trainer = Trainer(self.mock_data, [self.mock_variable])
            
            # Verify FNN was created (assuming Trainer creates a default network)
            # The exact architecture might be a parameter or hardcoded for now based on the spec/design
            # Design says "Defines the dde.nn.FNN network"
            MockFNN.assert_called()
            
            # Verify Model was created with the data and net
            MockModel.assert_called()
            
            # Verify compile was called with 'adam' and external_trainable_variables
            trainer.compile()
            trainer.model.compile.assert_called()
            
            # Check arguments of compile
            call_args = trainer.model.compile.call_args
            self.assertEqual(call_args[0][0], "adam") # optimizer
            self.assertIn("external_trainable_variables", call_args[1])
            self.assertEqual(call_args[1]["external_trainable_variables"], [self.mock_variable])

    def test_train_execution(self):
        """
        Test that train method executes training loop with callbacks.
        """
        with patch('src.training.dde.Model') as MockModel, \
             patch('src.training.dde.maps.FNN') as MockFNN, \
             patch('src.training.dde.callbacks.VariableValue') as MockVariableValue:
            
            trainer = Trainer(self.mock_data, [self.mock_variable])
            trainer.model.train = MagicMock()
            
            epochs = 1000
            trainer.train(epochs)
            
            # Verify VariableValue callback was created
            MockVariableValue.assert_called()
            
            # Verify train was called with epochs and callbacks
            trainer.model.train.assert_called()
            call_args = trainer.model.train.call_args
            self.assertEqual(call_args[1]['iterations'], epochs) # iterations is standard in dde
            self.assertIn('callbacks', call_args[1])
            # Check if our mock callback is in the list
            self.assertTrue(any(isinstance(cb, MagicMock) for cb in call_args[1]['callbacks']))

if __name__ == '__main__':
    unittest.main()
