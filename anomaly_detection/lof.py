import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from typing import Sequence

# # 1. 시계열 데이터 생성
# np.random.seed(42)
# time_series = np.random.normal(loc=100, scale=10, size=100)
# time_series[30] = 150  # 이상치 추가
# time_series[75] = 50   # 이상치 추가


def detect_outliers_with_isolation_forest(time_series: Sequence, n_lags: int) -> np.ndarray:
    """
    Isolation Forest를 사용하여 시계열 데이터의 이상치 여부를 반환하는 함수.
    :param time_series: 시계열 데이터 (list, numpy array, pandas Series)
    :param n_lags: 사용할 lag 개수
    :return: 입력 데이터와 같은 길이의 배열 (1: 정상, -1: 이상치)
    """
    def create_lagged_features(arr: Sequence, n_lags: int):
        df = pd.DataFrame({'x_t': arr})
        for lag in range(1, n_lags + 1):
            df[f'x_t-{lag}'] = df['x_t'].shift(lag)
        return df.dropna()

    # Lag 데이터 생성
    df = create_lagged_features(time_series, n_lags)

    # Isolation Forest 모델 학습
    iso_forest = IsolationForest(random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(df)

    # 원본 시계열 데이터 길이에 맞춰 결과 생성
    outliers = np.ones(len(time_series))  # 초기값: 정상 데이터 (1)
    outliers[df.index] = df['anomaly_score']

    return outliers



if __name__ == "__main__":
    # 1. 시계열 데이터 생성
    np.random.seed(42)
    time_series = np.random.normal(loc=100, scale=10, size=100)
    time_series[30] = 150  # 이상치 추가
    time_series[75] = 50  # 이상치 추가

    # 2. 함수 실행
    n_lags = 3  # 사용할 lag 개수
    outlier_arr = detect_outliers_with_lags(time_series, n_lags)

    # 3. 결과 확인
    print("Outlier Array:")
    print(outlier_arr)

    # 4. 시각화 (이상치 표시)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(time_series)), time_series, label="Time Series", color='blue')
    plt.scatter(np.where(outlier_arr == -1), time_series[outlier_arr == -1],
                color='red', label="Outliers")
    plt.title("Time Series with Detected Outliers")
    plt.legend()
    plt.show()

    x1 = time_series[:-1]
    x2 = time_series[1:]

    plt.figure()
    for x1_, x2_ in zip(x1, x2):
        plt.scatter(x1_, x2_)
    plt.show()
