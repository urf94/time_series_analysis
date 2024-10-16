import unittest
import numpy as np
import pandas as pd
import datetime
from changepoint_detection import voting, proba, proba_w_post, change_point_with_proba


class TestEnsembleFunction(unittest.TestCase):

    def setUp(self):
        # 테스트용 가상 데이터프레임 생성
        first_half = [i + (i % 10) for i in range(60)]
        second_half = first_half[39::-1]  # 첫 번째 절반의 대칭
        y_values = first_half + second_half

        self.df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2022-01-01", periods=100, freq="D"),
                "y": y_values,
            }
        )

    def test_proba_output_type(self):
        # proba 함수 호출 및 반환 값이 dict인지 확인
        result = proba(self.df)
        self.assertIsInstance(
            result, dict, "The output of proba should be a dictionary."
        )
        self.assertIn("n", result, "The output dictionary should contain the key 'n'.")
        self.assertIn("k", result, "The output dictionary should contain the key 'k'.")
        self.assertIn(
            "datetime",
            result,
            "The output dictionary should contain the key 'datetime'.",
        )

    def test_proba_not_empty(self):
        # proba 함수 호출 및 반환 값이 None이 아닌지 확인
        result = proba(self.df)
        self.assertIsNotNone(result, "The output of proba should not be None.")
        self.assertGreater(
            len(result), 0, "The output of proba should not be an empty dictionary."
        )

    def test_proba_valid_keys(self):
        # proba 함수 호출 및 반환된 딕셔너리가 예상된 키를 포함하고 있는지 확인
        result = proba(self.df)
        self.assertIn("n", result, "The output dictionary should contain the key 'n'.")
        self.assertIn("k", result, "The output dictionary should contain the key 'k'.")
        self.assertIsInstance(
            result["n"], int, "The value of 'n' should be an integer."
        )
        self.assertIsInstance(result["k"], float, "The value of 'k' should be a float.")
        self.assertIsInstance(
            result["datetime"],
            datetime.date,
            "The value of 'datetime' should be a datetime.date.",
        )

    def test_proba_with_invalid_data(self):
        # 너무 짧은 데이터에 대해 proba 함수가 None을 반환하는지 확인
        short_df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2022-01-01", periods=5, freq="D"),
                "y": [1, 2, 3, 4, 5],
            }
        )
        self.assertIsNone(
            proba(short_df), "The output should be None for too short data."
        )

    def test_proba_with_threshold(self):
        # 일정 threshold를 설정하고 proba 함수가 올바르게 작동하는지 확인
        result = proba(self.df, th=1.5)
        self.assertIsInstance(
            result, dict, "The output of proba should be a dictionary."
        )
        self.assertGreaterEqual(
            result["n"], 0, "The value of 'n' should be non-negative."
        )
        self.assertGreaterEqual(
            result["k"],
            -100.0,
            "The value of 'k' should be greater than or equal to -100.",
        )
        self.assertLessEqual(
            result["k"], 100.0, "The value of 'k' should be less than or equal to 100."
        )
        self.assertLessEqual(
            result["datetime"],
            datetime.date.today(),
            "The value of 'datetime' should be less than or equal to today.",
        )

        pre_changepoint = datetime.date(2022, 2, 25)
        result2 = proba_w_post(self.df, th=1.5, pre_changepoint=pre_changepoint)
        self.assertIsNone(result2)

    # ------------------- 새로운 테스트 케이스 시작 -------------------

    def test_change_point_with_proba_output_type(self):
        # change_point_with_proba 함수 호출 및 반환 값이 dict인지 확인
        result = change_point_with_proba(self.df)
        self.assertIsInstance(
            result,
            dict,
            "The output of change_point_with_proba should be a dictionary.",
        )
        self.assertIn("n", result, "The output dictionary should contain the key 'n'.")
        self.assertIn(
            "k1", result, "The output dictionary should contain the key 'k1'."
        )
        self.assertIn(
            "k2", result, "The output dictionary should contain the key 'k2'."
        )
        self.assertIn(
            "delta", result, "The output dictionary should contain the key 'delta'."
        )
        self.assertIn("p", result, "The output dictionary should contain the key 'p'.")

    def test_change_point_with_proba_not_empty(self):
        # change_point_with_proba 함수 호출 및 반환 값이 None이 아닌지 확인
        result = change_point_with_proba(self.df)
        self.assertIsNotNone(
            result, "The output of change_point_with_proba should not be None."
        )
        self.assertGreater(
            len(result),
            0,
            "The output of change_point_with_proba should not be an empty dictionary.",
        )

    def test_change_point_with_proba_valid_keys_and_types(self):
        # change_point_with_proba 함수 호출 및 반환된 딕셔너리가 예상된 키를 포함하고 있는지 확인
        result = change_point_with_proba(self.df)
        if result:
            self.assertIn(
                "n", result, "The output dictionary should contain the key 'n'."
            )
            self.assertIn(
                "k1", result, "The output dictionary should contain the key 'k1'."
            )
            self.assertIn(
                "k2", result, "The output dictionary should contain the key 'k2'."
            )
            self.assertIn(
                "delta", result, "The output dictionary should contain the key 'delta'."
            )
            self.assertIn(
                "p", result, "The output dictionary should contain the key 'p'."
            )
            self.assertIsInstance(
                result["n"], int, "The value of 'n' should be an integer."
            )
            self.assertIsInstance(
                result["k1"], float, "The value of 'k1' should be a float."
            )
            self.assertIsInstance(
                result["k2"], float, "The value of 'k2' should be a float."
            )
            self.assertTrue(
                isinstance(result["delta"], float) or result["delta"] is None,
                "The value of 'delta' should be a float or None.",
            )
            self.assertIsInstance(
                result["p"], float, "The value of 'p' should be a float."
            )

    def test_change_point_with_proba_with_invalid_data(self):
        # 너무 짧은 데이터에 대해 change_point_with_proba 함수가 None을 반환하는지 확인
        short_df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2022-01-01", periods=5, freq="D"),
                "y": [1, 2, 3, 4, 5],
            }
        )
        self.assertIsNone(
            change_point_with_proba(short_df),
            "The output should be None for too short data.",
        )

    def test_change_point_with_proba_with_threshold(self):
        # 일정 threshold를 설정하고 change_point_with_proba 함수가 올바르게 작동하는지 확인
        result = change_point_with_proba(self.df, th=10)
        # 임계값을 높게 설정했기 때문에 결과가 None일 가능성이 큼
        self.assertIsNone(
            result,
            "The output should be None when the threshold is set high and no changepoint meets it.",
        )

    def test_change_point_with_proba_correctness(self):
        # 알려진 변화점을 가진 데이터에서 change_point_with_proba 함수가 이를 정확히 감지하는지 확인
        # 변경점이 있는 데이터 생성
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        y = np.concatenate(
            [np.arange(100) * 1, np.arange(100) * 3 + 100]
        )  # 기울기 1에서 3으로 증가
        test_df = pd.DataFrame({"ds": dates, "y": y})

        result = change_point_with_proba(
            test_df, scales=[0.01, 0.05, 0.1], norm_method="z-score", th=2
        )

        self.assertIsNotNone(
            result,
            "The output of change_point_with_proba should not be None for data with a change point.",
        )
        # 변화점이 약 100번째 날짜인지 확인 (실제 Prophet 감지 위치는 다를 수 있음)
        expected_cp = dates[100].date()

        # 기울기와 n_days가 예상 범위에 있는지 확인
        self.assertTrue(
            90 <= result["n"] <= 110,
            "The value of 'n' should be exactly 100 for this synthetic data.",
        )
        self.assertAlmostEqual(
            result["k1"],
            1.0,
            places=1,
            msg="The value of 'k1' should be approximately 1.0.",
        )
        self.assertAlmostEqual(
            result["k2"],
            3.0,
            places=1,
            msg="The value of 'k2' should be approximately 2.0.",
        )
        d = result["delta"]
        self.assertGreaterEqual(
            d, 190, "The 'delta' should be greater than 200 - alpha."
        )
        self.assertLessEqual(d, 210, "The 'delta' should be less than 200 + alpha.")
        self.assertIsInstance(result["p"], float, "The value of 'p' should be a float.")

    def test_change_point_with_proba_delta_calculation_opposite_signs(self):
        # 기울기가 변화점 이전에는 양수, 이후에는 음수인 데이터 생성
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        y_pre = np.arange(100) * 1  # 기울기 1
        y_post = np.arange(100) * (-1) + 100  # 기울기 -1
        y = np.concatenate([y_pre, y_post])
        test_df = pd.DataFrame({"ds": dates, "y": y})

        result = change_point_with_proba(
            test_df, scales=[0.01, 0.05, 0.1], norm_method="z-score", th=2
        )

        self.assertIsNotNone(
            result,
            "The output of change_point_with_proba should not be None for data with a change point.",
        )
        self.assertIn(
            "delta", result, "The output dictionary should contain the key 'delta'."
        )
        # delta는 양수에서 음수로 변화했으므로 None이어야 함
        self.assertIsNone(
            result["delta"],
            "The value of 'delta' should be None when k1 and k2 have opposite signs.",
        )


if __name__ == "__main__":
    unittest.main()
