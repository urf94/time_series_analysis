import numpy as np
import matplotlib.pyplot as plt


def visualize_outlier_detection_with_offset(time_series, outliers_dict, offset: float = 1, title="Outlier Detection with Offset"):
    """
    여러 방법으로 이상치 탐지 결과를 수직 오프셋 방식으로 시각화.
    :param time_series: 원본 시계열 데이터 (numpy array or list)
    :param outliers_dict: 이상치 탐지 결과 딕셔너리 (방법 이름: 이상치 여부 배열)
    :param offset: 이상치 탐지 결과 시각화 용 오프셋 (수직)
    :param title: 그래프 제목
    """
    plt.figure(figsize=(12, 8))

    # 원본 시계열 데이터 플롯
    plt.plot(range(len(time_series)), time_series, label="Original Time Series", color='blue', linewidth=1)

    # 각 방법별 이상치 표시
    colors = ['red', 'green', 'purple', 'orange']
    for i, (method, outliers) in enumerate(outliers_dict.items()):
        anomaly_indices = np.where(outliers == -1)[0]  # 이상치 인덱스
        plt.scatter(anomaly_indices, time_series[anomaly_indices] + (i * offset),
                    color=colors[i], label=f"{method} Outliers", s=50, alpha=0.7)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value (with offset)")
    plt.legend()
    plt.grid()
    plt.show()

