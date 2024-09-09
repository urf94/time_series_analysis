import unittest
import pandas as pd
import datetime
from changepoint_detection import voting, proba, NoChangePointDetectedError

class TestEnsembleFunction(unittest.TestCase):

    def setUp(self):
        # 테스트용 가상 데이터프레임 생성
        first_half = [i + (i % 10) for i in range(60)]
        second_half = first_half[39::-1]  # 첫 번째 절반의 대칭
        y_values = first_half + second_half

        self.df = pd.DataFrame({
            'ds': pd.date_range(start='2022-01-01', periods=100, freq='D'),
            'y': y_values
        })

    def test_proba_output_type(self):
        # proba 함수 호출 및 반환 값이 dict인지 확인
        result = proba(self.df)
        self.assertIsInstance(result, dict, "The output of proba should be a dictionary.")
        self.assertIn("n", result, "The output dictionary should contain the key 'n'.")
        self.assertIn("k", result, "The output dictionary should contain the key 'k'.")
        self.assertIn("datetime", result, "The output dictionary should contain the key 'datetime'.")

    def test_proba_not_empty(self):
        # proba 함수 호출 및 반환 값이 None이 아닌지 확인
        result = proba(self.df)
        self.assertIsNotNone(result, "The output of proba should not be None.")
        self.assertGreater(len(result), 0, "The output of proba should not be an empty dictionary.")

    def test_proba_valid_keys(self):
        # proba 함수 호출 및 반환된 딕셔너리가 예상된 키를 포함하고 있는지 확인
        result = proba(self.df)
        self.assertIn("n", result, "The output dictionary should contain the key 'n'.")
        self.assertIn("k", result, "The output dictionary should contain the key 'k'.")
        self.assertIsInstance(result["n"], int, "The value of 'n' should be an integer.")
        self.assertIsInstance(result["k"], float, "The value of 'k' should be a float.")
        self.assertIsInstance(result["datetime"], datetime.date, "The value of 'datetime' should be a datime.date.")

    def test_proba_with_invalid_data(self):
        # 너무 짧은 데이터에 대해 NoChangePointDetectedError가 발생하는지 확인
        short_df = pd.DataFrame({
            'ds': pd.date_range(start='2022-01-01', periods=5, freq='D'),
            'y': [1, 2, 3, 4, 5]
        })
        self.assertIsNone(proba(short_df))


    def test_proba_with_threshold(self):
        # 일정 threshold를 설정하고 proba 함수가 올바르게 작동하는지 확인
        result = proba(self.df, th=1.5)
        self.assertIsInstance(result, dict, "The output of proba should be a dictionary.")
        self.assertGreaterEqual(result["n"], 0, "The value of 'n' should be non-negative.")
        self.assertGreaterEqual(result["k"], -100.0, "The value of 'k' should be greater than or equal to -100.")
        self.assertLessEqual(result["k"], 100.0, "The value of 'k' should be less than or equal to 100.")
        self.assertLessEqual(result["datetime"], datetime.date.today(), "The value of 'datetime' should be less than or equal to today.")

if __name__ == '__main__':
    unittest.main()
