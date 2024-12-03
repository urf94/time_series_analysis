from typing import Sequence
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


def detect_outliers_with_gmm(time_series: Sequence, n_lags: int, percentile: float = 5) -> np.ndarray:
    """
    Gaussian Mixture Model (GMM)을 사용하여 시계열 데이터의 이상치 여부를 반환하는 함수.
    :param time_series: 시계열 데이터 (list, numpy array, pandas Series)
    :param n_lags: 사용할 lag 개수
    :param percentile: 이상치 기준 확률 밀도의 백분위 값
    :return: 입력 데이터와 같은 길이의 배열 (1: 정상, -1: 이상치)
    """
    def create_lagged_features(arr: Sequence, n_lags: int):
        df = pd.DataFrame({'x_t': arr})
        for lag in range(1, n_lags + 1):
            df[f'x_t-{lag}'] = df['x_t'].shift(lag)
        return df.dropna()

    # Lag 데이터 생성
    df = create_lagged_features(time_series, n_lags)

    # GMM 모델 학습
    gmm = GaussianMixture(n_components=2, random_state=42)
    df['probabilities'] = gmm.fit(df).score_samples(df)

    # 이상치 여부 결정 (백분위 기준)
    threshold = np.percentile(df['probabilities'], percentile)
    df['anomaly_score'] = np.where(df['probabilities'] < threshold, -1, 1)

    # 원본 시계열 데이터 길이에 맞춰 결과 생성
    outliers = np.ones(len(time_series))  # 초기값: 정상 데이터 (1)
    outliers[df.index] = df['anomaly_score']

    return outliers
