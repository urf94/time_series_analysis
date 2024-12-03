from typing import Sequence
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def detect_outliers_with_dbscan(time_series: Sequence, n_lags: int, eps: float = 10, min_samples: int = 5) -> np.ndarray:
    """
    DBSCAN을 사용하여 시계열 데이터의 이상치 여부를 반환하는 함수.
    :param time_series: 시계열 데이터 (list, numpy array, pandas Series)
    :param n_lags: 사용할 lag 개수
    :param eps: DBSCAN의 거리 기준 (eps)
    :param min_samples: 클러스터 형성에 필요한 최소 샘플 수
    :return: 입력 데이터와 같은 길이의 배열 (1: 정상, -1: 이상치)
    """
    def create_lagged_features(arr: Sequence, n_lags: int):
        df = pd.DataFrame({'x_t': arr})
        for lag in range(1, n_lags + 1):
            df[f'x_t-{lag}'] = df['x_t'].shift(lag)
        return df.dropna()

    # Lag 데이터 생성
    df = create_lagged_features(time_series, n_lags)

    # DBSCAN 모델 학습
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster'] = dbscan.fit_predict(df)

    # 이상치 여부 결정
    outliers = np.ones(len(time_series))  # 초기값: 정상 데이터 (1)
    outliers[df.index] = np.where(df['cluster'] == -1, -1, 1)

    return outliers
