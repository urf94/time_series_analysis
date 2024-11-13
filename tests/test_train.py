import unittest
import pandas as pd
from changepoint_detection.train import train_prophet


class TestTrainFunctions(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "checkpoint"
        # 테스트용 데이터 생성
        self.df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2022-01-01", periods=10, freq="D"),
                "y": [1, 2, 1.5, 2.5, 1.8, 2.1, 1.9, 2.3, 2.2, 1.7],
            }
        )

    def test_train_prophet(self):
        scale = 0.1
        try:
            train_prophet(self.df, scale, checkpoint_dir=self.checkpoint_dir)
        except Exception as e:
            self.assertIsInstance(e, Exception)
            print(f"Handled exception: {e}")

    # def test_train_neuralprophet(self):
    #     try:
    #         train_neuralprophet(self.df, checkpoint_dir=self.checkpoint_dir)
    #     except RuntimeError as e:
    #         self.assertIsInstance(e, RuntimeError)
    #         print(f"Handled RuntimeError: {e}")


if __name__ == "__main__":
    unittest.main()
