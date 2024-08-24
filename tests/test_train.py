import unittest
import os
import pandas as pd
from changepoint_detection.train import train_prophet, train_neuralprophet


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
        train_prophet(self.df, scale, checkpoint_dir=self.checkpoint_dir)
        model_path = os.path.join(self.checkpoint_dir, f"prophet_scale_{scale}.pkl")
        self.assertTrue(
            os.path.exists(model_path),
            f"{model_path} not found after training Prophet.",
        )

    def test_train_neuralprophet(self):
        train_neuralprophet(self.df, checkpoint_dir=self.checkpoint_dir)
        model_path = os.path.join(self.checkpoint_dir, "neuralprophet.pkl")
        self.assertTrue(
            os.path.exists(model_path),
            f"{model_path} not found after training NeuralProphet.",
        )

    def tearDown(self):
        # 테스트 후 체크포인트 파일 삭제
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith(".pkl"):
                os.remove(os.path.join(self.checkpoint_dir, file))


if __name__ == "__main__":
    unittest.main()
