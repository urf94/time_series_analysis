import unittest
import os
import pandas as pd
from changepoint_detection.inference import inference_prophet, inference_neuralprophet


class TestInferenceFunctions(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "checkpoint"
        self.df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2022-01-01", periods=10, freq="D"),
                "y": [1, 2, 1.5, 2.5, 1.8, 2.1, 1.9, 2.3, 2.2, 1.7],
            }
        )

        # Prophet 모델이 존재하는지 확인
        prophet_model_path = os.path.join(self.checkpoint_dir, "prophet_scale_0.1.pkl")
        if not os.path.exists(prophet_model_path):
            self.skipTest(f"Prophet model not found at {prophet_model_path}")

        # NeuralProphet 모델이 존재하는지 확인
        neuralprophet_model_path = os.path.join(
            self.checkpoint_dir, "neuralprophet.pkl"
        )
        if not os.path.exists(neuralprophet_model_path):
            self.skipTest(
                f"NeuralProphet model not found at {neuralprophet_model_path}"
            )

    def test_inference_prophet(self):
        try:
            changepoints = inference_prophet(
                self.df, scale=0.1, checkpoint_dir=self.checkpoint_dir
            )
            self.assertIsInstance(
                changepoints, list, "Prophet inference should return a list."
            )
            self.assertGreaterEqual(
                len(changepoints), 0, "Changepoints list should not be empty."
            )
        except Exception as e:
            self.fail(f"Prophet inference failed with exception: {e}")

    def test_inference_neuralprophet(self):
        try:
            changepoints = inference_neuralprophet(
                self.df, checkpoint_dir=self.checkpoint_dir
            )
            self.assertIsInstance(
                changepoints, list, "NeuralProphet inference should return a list."
            )
            self.assertGreaterEqual(
                len(changepoints), 0, "Changepoints list should not be empty."
            )
        except RuntimeError as e:
            self.fail(f"NeuralProphet inference failed with RuntimeError: {e}")
        except Exception as e:
            self.fail(f"NeuralProphet inference failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
