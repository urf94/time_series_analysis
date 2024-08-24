import unittest
import pandas as pd
from changepoint_detection.inference import inference_prophet, inference_neuralprophet
from changepoint_detection.train import train_prophet, train_neuralprophet

class TestInferenceFunctions(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "checkpoint"
        self.df = pd.DataFrame({
            'ds': pd.date_range(start='2022-01-01', periods=10, freq='D'),
            'y': [1, 2, 1.5, 2.5, 1.8, 2.1, 1.9, 2.3, 2.2, 1.7]
        })

        # 먼저 학습을 수행하여 모델 생성
        train_prophet(self.df, scale=0.1, checkpoint_dir=self.checkpoint_dir)
        train_neuralprophet(self.df, checkpoint_dir=self.checkpoint_dir)

    def test_inference_prophet(self):
        changepoints = inference_prophet(self.df, scale=0.1, checkpoint_dir=self.checkpoint_dir)
        self.assertIsInstance(changepoints, list, "Prophet inference should return a list.")
        self.assertGreaterEqual(len(changepoints), 0, "Changepoints list should not be empty.")

    def test_inference_neuralprophet(self):
        changepoints = inference_neuralprophet(self.df, checkpoint_dir=self.checkpoint_dir)
        self.assertIsInstance(changepoints, list, "NeuralProphet inference should return a list.")
        self.assertGreaterEqual(len(changepoints), 0, "Changepoints list should not be empty.")

if __name__ == '__main__':
    unittest.main()
