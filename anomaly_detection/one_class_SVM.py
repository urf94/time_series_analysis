from typing import Sequence
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM

def detect_outliers_with_svm(time_series: Sequence, n_lags: int, gamma: float = 0.1, nu: float = 0.05) -> np.ndarray:
    """
    One-Class SVM을 사용하여 시계열 데이터의 이상치 여부를 반환하는 함수.
    :param time_series: 시계열 데이터 (list, numpy array, pandas Series)
    :param n_lags: 사용할 lag 개수
    :param gamma: SVM의 커널 계수
    :param nu: 이상치 비율 상한
    :return: 입력 데이터와 같은 길이의 배열 (1: 정상, -1: 이상치)
    """
    def create_lagged_features(arr: Sequence, n_lags: int):
        df = pd.DataFrame({'x_t': arr})
        for lag in range(1, n_lags + 1):
            df[f'x_t-{lag}'] = df['x_t'].shift(lag)
        return df.dropna()

    # Lag 데이터 생성
    df = create_lagged_features(time_series, n_lags)

    # SVM 모델 학습
    svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
    df['anomaly_score'] = svm.fit_predict(df)

    # 이상치 여부 결정
    outliers = np.ones(len(time_series))  # 초기값: 정상 데이터 (1)
    outliers[df.index] = df['anomaly_score']

    return outliers
