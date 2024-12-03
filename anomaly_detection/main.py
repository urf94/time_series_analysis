from typing import Sequence
import pandas as pd
import numpy as np
from .lof import detect_outliers_with_isolation_forest
from .dbscan import detect_outliers_with_dbscan
from .one_class_SVM import detect_outliers_with_svm
from .gaussian_mixture import detect_outliers_with_gmm
from .visualization import visualize_outlier_detection_results



if __name__ == "__main__":
    # 1. 시계열 데이터 생성
    np.random.seed(42)
    time_series = np.random.normal(loc=100, scale=10, size=100)
    time_series[30] = 150  # 이상치 추가
    time_series[75] = 50  # 이상치 추가

    n_lags = 3  # 사용할 lag 개수
    # DBSCAN
    outliers_dbscan = detect_outliers_with_dbscan(time_series, n_lags)
    # One-Class SVM
    outliers_svm = detect_outliers_with_svm(time_series, n_lags)
    # Isolation Forest
    outliers_if = detect_outliers_with_isolation_forest(time_series, n_lags)
    # GMM
    outliers_gmm = detect_outliers_with_gmm(time_series, n_lags)

    # 탐지 결과 딕셔너리
    outliers_dict = {
        "DBSCAN": outliers_dbscan,
        "One-Class SVM": outliers_svm,
        "Isolation Forest": outliers_if,
        "GMM": outliers_gmm
    }

    # 시각화
    visualize_outlier_detection_results(time_series, outliers_dict)

    # # 3. 결과 확인
    # print("Outlier Array:")
    # print(outlier_arr)
    #
    # # 4. 시각화 (이상치 표시)
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(len(time_series)), time_series, label="Time Series", color='blue')
    # plt.scatter(np.where(outlier_arr == -1), time_series[outlier_arr == -1],
    #             color='red', label="Outliers")
    # plt.title("Time Series with Detected Outliers")
    # plt.legend()
    # plt.show()
    #
    # x1 = time_series[:-1]
    # x2 = time_series[1:]
    #
    # plt.figure()
    # for x1_, x2_ in zip(x1, x2):
    #     plt.scatter(x1_, x2_)
    # plt.show()


