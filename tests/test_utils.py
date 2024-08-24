import unittest
import os
import pickle
from prophet import Prophet
from neuralprophet import NeuralProphet
from changepoint_detection.utils import load_model, save_model


class TestUtilsFunctions(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "checkpoint"
        self.prophet_model_path = os.path.join(self.checkpoint_dir, "test_prophet_model.pkl")
        self.neuralprophet_model_path = os.path.join(self.checkpoint_dir, "test_neuralprophet_model.pkl")

    def test_save_and_load_prophet_model(self):
        # Create and save a Prophet model
        model = Prophet()
        save_model(model, self.prophet_model_path)
        self.assertTrue(os.path.exists(self.prophet_model_path), "Prophet model was not saved correctly.")

        # Load the model
        loaded_model = load_model(self.prophet_model_path)
        self.assertIsInstance(loaded_model, Prophet, "Loaded model is not a Prophet instance.")

    def test_save_and_load_neuralprophet_model(self):
        # Create and save a NeuralProphet model
        model = NeuralProphet()
        save_model(model, self.neuralprophet_model_path)
        self.assertTrue(os.path.exists(self.neuralprophet_model_path), "NeuralProphet model was not saved correctly.")

        # Load the model
        loaded_model = load_model(self.neuralprophet_model_path)
        self.assertIsInstance(loaded_model, NeuralProphet, "Loaded model is not a NeuralProphet instance.")

    def tearDown(self):
        # Remove created files after tests
        if os.path.exists(self.prophet_model_path):
            os.remove(self.prophet_model_path)
        if os.path.exists(self.neuralprophet_model_path):
            os.remove(self.neuralprophet_model_path)


if __name__ == '__main__':
    unittest.main()
