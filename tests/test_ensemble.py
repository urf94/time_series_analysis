import unittest
import pandas as pd
import numpy as np
import datetime
from changepoint_detection import proba, proba_w_post, change_point_with_proba

from samples.taylormade import ds as tm_ds, y as tm_y
from samples.women09 import ds as wm_ds, y as wm_y

class TestEnsembleFunctionDeprecated(unittest.TestCase):

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


class TestChangePointDetectionFunctions(unittest.TestCase):

    def setUp(self):
        # 테스트용 'ds'와 'y' 데이터

        self.ds = wm_ds
        self.y = wm_y
        # self.ds = tm_ds
        # self.y = tm_y

        self.df = pd.DataFrame({"ds": self.ds, "y": self.y})

    # ------------------- 기존 proba 함수 테스트 -------------------
    # DELETED

    # ------------------- 기존 change_point_with_proba 함수 테스트 -------------------

    def test_change_point_with_proba_output_type(self):
        """change_point_with_proba 함수 호출 및 반환 값이 dict인지 확인"""
        result = change_point_with_proba(self.df)
        self.assertIsInstance(
            result,
            dict,
            "The output of change_point_with_proba should be a dictionary.",
        )
        self.assertIn("n", result, "The output dictionary should contain the key 'n'.")
        self.assertIn(
            "datetime",
            result,
            "The output dictionary should contain the key 'datetime'.",
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
        self.assertIn("p", result, "The output dictionary should contain the key 'p'.")

    def test_change_point_with_proba_not_empty(self):
        """change_point_with_proba 함수 호출 및 반환 값이 None이 아닌지 확인"""
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
        """change_point_with_proba 함수 호출 및 반환된 딕셔너리가 예상된 키를 포함하고 있는지 확인"""
        result = change_point_with_proba(self.df)
        if result:
            self.assertIn(
                "n", result, "The output dictionary should contain the key 'n'."
            )
            self.assertIn(
                "datetime",
                result,
                "The output dictionary should contain the key 'datetime'.",
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
                result["datetime"], str, "The value of 'datetime' should be a string."
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
        """너무 짧은 데이터에 대해 change_point_with_proba 함수가 None을 반환하는지 확인"""
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
        """일정 threshold를 설정하고 change_point_with_proba 함수가 올바르게 작동하는지 확인"""
        result = change_point_with_proba(self.df, th=10)
        # 임계값을 높게 설정했기 때문에 결과가 None일 가능성이 큼
        self.assertIsNone(
            result,
            "The output should be None when the threshold is set high and no changepoint meets it.",
        )

    def test_change_point_with_proba_correctness(self):
        """알려진 변화점을 가진 데이터에서 change_point_with_proba 함수가 이를 정확히 감지하는지 확인"""
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
        expected_cp = dates[103].strftime("%Y-%m-%d")
        self.assertEqual(
            result["datetime"],
            expected_cp,
            f"The detected changepoint should be {expected_cp}.",
        )
        # 기울기와 n_days가 예상 범위에 있는지 확인
        self.assertTrue(
            90 <= result["n"] <= 110,
            "The value of 'n' should be around 100 for this synthetic data.",
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
            msg="The value of 'k2' should be approximately 3.0.",
        )
        # self.assertGreaterEqual(
        #     result["delta"], 190, "The 'delta' should be greater than 190."
        # )
        # self.assertLessEqual(
        #     result["delta"], 210, "The 'delta' should be less than 210."
        # )
        self.assertGreater(
            result["delta"], 280, "The 'delta' should be more than 280."
        )
        self.assertIsInstance(result["p"], float, "The value of 'p' should be a float.")

    def test_change_point_with_proba_delta_calculation_opposite_signs(self):
        """기울기가 변화점 이전에는 양수, 이후에는 음수인 데이터 생성"""
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

    # ------------------- 추가적인 예외 및 기능 테스트 -------------------

    def test_insufficient_data(self):
        """입력 데이터의 고유 날짜 수가 90 미만일 때 None을 반환하는지 테스트"""
        # 고유한 'ds'가 2개인 데이터프레임
        ds_short = [
            "2024-07-23",
            "2024-07-24",
        ] * 45  # 총 90 데이터 포인트지만 고유 날짜는 2개
        y_short = [100] * 90
        sample_df_short = pd.DataFrame({"ds": ds_short, "y": y_short})
        result = change_point_with_proba(sample_df_short)
        self.assertIsNone(
            result, "Result should be None for insufficient unique data points."
        )

    def test_sparse_data(self):
        """sparsity=True일 때 y=0이 45개 초과하면 None을 반환하는지 테스트"""
        # y=0인 데이터 포인트가 46개인 데이터프레임
        ds_sparse = self.ds.copy()
        y_sparse = [0] * 50 + self.y.copy()[50:]
        sample_df_sparse = pd.DataFrame({"ds": ds_sparse, "y": y_sparse})
        result = change_point_with_proba(sample_df_sparse, sparsity=True)
        self.assertIsNone(
            result, "Result should be None for sparse data when sparsity=True."
        )

    def test_zero_y_values(self):
        """y=0 값이 있는 데이터에서 함수가 올바르게 처리하는지 테스트"""
        # 일부 y 값이 0인 데이터프레임
        ds_zero_y = self.ds.copy() + ["2024-10-21", "2024-10-22", "2024-10-23"]
        y_zero_y = self.y + [0, 0, 0]
        sample_df_zero_y = pd.DataFrame({"ds": ds_zero_y, "y": y_zero_y})
        result = change_point_with_proba(sample_df_zero_y)
        self.assertTrue(
            isinstance(result, dict) or result is None,
            "Result should be a dict or None when some y=0.",
        )

    def test_invalid_ds_column(self):
        """'ds' 컬럼이 없는 데이터프레임을 입력했을 때 ValueError가 발생하는지 테스트"""
        sample_df_missing_ds = pd.DataFrame({"y": self.y})
        with self.assertRaises(ValueError):
            change_point_with_proba(sample_df_missing_ds)

    def test_invalid_y_column(self):
        """'y' 컬럼이 없는 데이터프레임을 입력했을 때 ValueError가 발생하는지 테스트"""
        sample_df_missing_y = pd.DataFrame({"ds": self.ds})
        with self.assertRaises(ValueError):
            change_point_with_proba(sample_df_missing_y)

    def test_invalid_norm_method(self):
        """지원되지 않는 norm_method를 입력했을 때 NotImplementedError가 발생하는지 테스트"""
        sample_df_str = pd.DataFrame({"ds": self.ds, "y": self.y})
        with self.assertRaises(NotImplementedError):
            change_point_with_proba(sample_df_str, norm_method="unsupported_method")

    def test_sparsity_false(self):
        """sparsity=False일 때 함수가 정상적으로 동작하는지 테스트"""
        # y=0인 데이터 포인트가 46개인 데이터프레임
        ds_sparse = self.ds.copy()
        y_sparse = [0] * 50 + self.y.copy()[50:]
        sample_df_sparse = pd.DataFrame({"ds": ds_sparse, "y": y_sparse})
        result = change_point_with_proba(sample_df_sparse, sparsity=False)
        self.assertTrue(
            isinstance(result, dict) or result is None,
            "Result should be a dict or None when sparsity=False.",
        )

    def test_change_point_correctness(self):
        """명확한 변화점이 있는 데이터에서 함수가 이를 정확히 감지하는지 테스트"""
        # 명확한 변화점이 있는 데이터 생성
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        y = np.concatenate(
            [np.arange(100) * 1, np.arange(100) * 3 + 100]
        )  # 기울기 1에서 3으로 증가
        test_df = pd.DataFrame({"ds": dates, "y": y})

        result = change_point_with_proba(
            test_df, norm_method="z-score", th=2
        )

        self.assertIsNotNone(
            result,
            "The output of change_point_with_proba should not be None for data with a change point.",
        )
        # 변화점이 약 100번째 날짜인지 확인
        expected_cp = dates[103].strftime("%Y-%m-%d")
        self.assertEqual(
            result["datetime"],
            expected_cp,
            f"The detected changepoint should be {expected_cp}.",
        )
        # 기울기와 n_days가 예상 범위에 있는지 확인
        self.assertTrue(
            90 <= result["n"] <= 110,
            "The value of 'n' should be around 100 for this synthetic data.",
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
            msg="The value of 'k2' should be approximately 3.0.",
        )
        self.assertGreater(
            result["delta"], 280, "The 'delta' should be more than 280."
        )
        # self.assertGreaterEqual(
        #     result["delta"], 190, "The 'delta' should be greater than 190."
        # )
        # self.assertLessEqual(
        #     result["delta"], 210, "The 'delta' should be less than 210."
        # )
        self.assertIsInstance(result["p"], float, "The value of 'p' should be a float.")

    def test_change_point_with_proba_delta_calculation_opposite_signs(self):
        """기울기가 변화점 이전에는 양수, 이후에는 음수인 데이터 생성"""
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

    def test_change_point_with_proba_ds_various_types(self):
        """change_point_with_proba 함수가 다양한 ds 타입(str, datetime64[ns], datetime64[ms], pandas.Timestamp)에 대해 정상 동작하는지 확인"""
        # 1. 문자열 (str)
        sample_df_str = pd.DataFrame({"ds": self.ds, "y": self.y})
        result_str = change_point_with_proba(sample_df_str)
        self.assertTrue(
            isinstance(result_str, dict) or result_str is None,
            "Result should be a dict or None for 'ds' as string.",
        )
        if result_str:
            expected_keys = {"n", "datetime", "k1", "k2", "delta", "p"}
            self.assertTrue(
                expected_keys.issubset(result_str.keys()),
                "Result dict missing expected keys for 'ds' as string.",
            )
            self.assertIsInstance(result_str["n"], int, "'n' should be an integer.")
            self.assertIsInstance(
                result_str["datetime"], str, "'datetime' should be a string."
            )
            self.assertIsInstance(result_str["k1"], float, "'k1' should be a float.")
            self.assertIsInstance(result_str["k2"], float, "'k2' should be a float.")
            self.assertTrue(
                isinstance(result_str["delta"], float) or result_str["delta"] is None,
                "'delta' should be a float or None.",
            )
            self.assertIsInstance(result_str["p"], float, "'p' should be a float.")

        # 2. datetime64[ns]
        sample_df_datetime_ns = pd.DataFrame(
            {"ds": pd.to_datetime(self.ds, format="%Y-%m-%d"), "y": self.y}
        )
        result_datetime_ns = change_point_with_proba(sample_df_datetime_ns)
        self.assertTrue(
            isinstance(result_datetime_ns, dict) or result_datetime_ns is None,
            "Result should be a dict or None for 'ds' as datetime64[ns].",
        )
        if result_datetime_ns:
            expected_keys = {"n", "datetime", "k1", "k2", "delta", "p"}
            self.assertTrue(
                expected_keys.issubset(result_datetime_ns.keys()),
                "Result dict missing expected keys for 'ds' as datetime64[ns].",
            )
            self.assertIsInstance(
                result_datetime_ns["n"], int, "'n' should be an integer."
            )
            self.assertIsInstance(
                result_datetime_ns["datetime"], str, "'datetime' should be a string."
            )
            self.assertIsInstance(
                result_datetime_ns["k1"], float, "'k1' should be a float."
            )
            self.assertIsInstance(
                result_datetime_ns["k2"], float, "'k2' should be a float."
            )
            self.assertTrue(
                isinstance(result_datetime_ns["delta"], float)
                or result_datetime_ns["delta"] is None,
                "'delta' should be a float or None.",
            )
            self.assertIsInstance(
                result_datetime_ns["p"], float, "'p' should be a float."
            )

        # 3. datetime64[ms]
        sample_df_datetime_ms = pd.DataFrame(
            {
                "ds": pd.to_datetime(self.ds, format="%Y-%m-%d").astype(
                    "datetime64[ms]"
                ),
                "y": self.y,
            }
        )
        result_datetime_ms = change_point_with_proba(sample_df_datetime_ms)
        self.assertTrue(
            isinstance(result_datetime_ms, dict) or result_datetime_ms is None,
            "Result should be a dict or None for 'ds' as datetime64[ms].",
        )
        if result_datetime_ms:
            expected_keys = {"n", "datetime", "k1", "k2", "delta", "p"}
            self.assertTrue(
                expected_keys.issubset(result_datetime_ms.keys()),
                "Result dict missing expected keys for 'ds' as datetime64[ms].",
            )
            self.assertIsInstance(
                result_datetime_ms["n"], int, "'n' should be an integer."
            )
            self.assertIsInstance(
                result_datetime_ms["datetime"], str, "'datetime' should be a string."
            )
            self.assertIsInstance(
                result_datetime_ms["k1"], float, "'k1' should be a float."
            )
            self.assertIsInstance(
                result_datetime_ms["k2"], float, "'k2' should be a float."
            )
            self.assertTrue(
                isinstance(result_datetime_ms["delta"], float)
                or result_datetime_ms["delta"] is None,
                "'delta' should be a float or None.",
            )
            self.assertIsInstance(
                result_datetime_ms["p"], float, "'p' should be a float."
            )

        # 4. pandas.Timestamp 객체
        sample_df_timestamp = pd.DataFrame(
            {"ds": [pd.Timestamp(x) for x in self.ds], "y": self.y}
        )
        result_timestamp = change_point_with_proba(sample_df_timestamp)
        self.assertTrue(
            isinstance(result_timestamp, dict) or result_timestamp is None,
            "Result should be a dict or None for 'ds' as pandas.Timestamp.",
        )
        if result_timestamp:
            expected_keys = {"n", "datetime", "k1", "k2", "delta", "p"}
            self.assertTrue(
                expected_keys.issubset(result_timestamp.keys()),
                "Result dict missing expected keys for 'ds' as pandas.Timestamp.",
            )
            self.assertIsInstance(
                result_timestamp["n"], int, "'n' should be an integer."
            )
            self.assertIsInstance(
                result_timestamp["datetime"], str, "'datetime' should be a string."
            )
            self.assertIsInstance(
                result_timestamp["k1"], float, "'k1' should be a float."
            )
            self.assertIsInstance(
                result_timestamp["k2"], float, "'k2' should be a float."
            )
            self.assertTrue(
                isinstance(result_timestamp["delta"], float)
                or result_timestamp["delta"] is None,
                "'delta' should be a float or None.",
            )
            self.assertIsInstance(
                result_timestamp["p"], float, "'p' should be a float."
            )

    if __name__ == "__main__":
        unittest.main()
