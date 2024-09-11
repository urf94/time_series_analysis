import datetime
from typing import Union
import numpy as np
import pandas as pd
from prophet import Prophet
from collections import Counter


class NoChangePointDetectedError(Exception):
    def __str__(self):
        return "No changepoint detected in the provided data."


def softmax(x):
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def z_score_normalization(x):
    mean_proba = np.mean(x)
    std_proba = np.std(x)
    z_scores = (x - mean_proba) / std_proba
    return z_scores


def voting(df: pd.DataFrame):
    # changepoint_prior_scale 값들을 정의 (default: 0.05)
    scales = [0.005 * (5*i) for i in range(1, 9)]    # 0.005, 0.025, 0.05, ..., 0.2
    changepoints_list = []

    # 각 scale에 대해 Prophet 모델을 학습하고 changepoints를 추출
    for scale in scales:
        model = Prophet(changepoint_prior_scale=scale)
        model.fit(df)
        forecast = model.predict(model.make_future_dataframe(periods=0))

        # 모델에서 changepoints 추출
        changepoints = model.changepoints
        changepoints_list.extend(changepoints)

    # 각 changepoint에 대해 voting
    changepoint_counter = Counter(changepoints_list)
    most_common_changepoint, _ = changepoint_counter.most_common(1)[0]
    return most_common_changepoint


def proba(df, norm_method: str = "z-score", th: float = 2) -> Union[None, dict]:
    # changepoint_prior_scale 값들을 정의
    scales = [0.005 * (5 * i) for i in range(1, 9)]  # 0.005, 0.025, 0.05, ..., 0.2
    n_scales = len(scales)
    datetime_proba = {}
    all_trends = []

    # 각 scale에 대해 Prophet 모델을 학습하고 changepoints 및 proba 추출
    for scale in scales:
        model = Prophet(changepoint_prior_scale=scale, changepoint_range=1.0)
        model.fit(df)

        # 모델에서 changepoints 추출
        changepoints = model.changepoints

        # changepoints의 확률(proba) 계산
        proba = np.abs(np.nanmean(model.params['delta'], axis=0))
        if norm_method == "z-score":
            proba = z_score_normalization(proba)
        elif norm_method == "softmax":
            proba = softmax(proba)
        proba /= n_scales

        # changepoints와 proba를 datetime_proba에 누적
        for i, changepoint in enumerate(changepoints):
            if changepoint in datetime_proba:
                datetime_proba[changepoint] += proba[i]
            else:
                datetime_proba[changepoint] = proba[i]

        # 각 모델의 trend 예측 결과 저장
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        all_trends.append(forecast['trend'])

    # 여러 모델로부터의 trend 평균 계산
    avg_trend = pd.concat(all_trends, axis=1).mean(axis=1)

    # 일정 임계값(threshold)을 넘는 changepoint들만 반환
    results = [changepoint for changepoint, proba in datetime_proba.items() if proba >= th]

    if not results:
        return None

    max_changepoint = max(results)

    # get partial data after changepoint
    changepoint_index = forecast[forecast['ds'] == max_changepoint].index[0]
    post_changepoint_trend = avg_trend.iloc[changepoint_index + 1:]  # changepoint 이후의 trend

    # calculate min, max trend value
    max_trend_after = post_changepoint_trend.max()  # 이후 데이터의 최대 trend 값
    min_trend_after = post_changepoint_trend.min()  # 이후 데이터의 최소 trend 값

    # trend value at changepoint
    changepoint_trend = avg_trend.iloc[changepoint_index]

    # K: delta trend
    if max_trend_after - changepoint_trend >= changepoint_trend - min_trend_after:
        k = (max_trend_after - changepoint_trend) / changepoint_trend * 100
    else:
        k = (min_trend_after - changepoint_trend) / changepoint_trend * 100

    # N: timestamp interval (chagepoint ~ last)
    last_datetime = df['ds'].max()
    n = (last_datetime - max_changepoint).days

    if n < len(df) * 0.1:
        return None

    # 결과 반환
    return {"n": n, "k": round(k, 2), 'datetime': max_changepoint.date()}


def proba_w_post(df, norm_method: str = "z-score", th: float = 2, **kwargs):
    pre_datetime = kwargs.get('pre_changepoint')
    inference = proba(df, norm_method, th)
    if inference:
        return None if inference['datetime'] == pre_datetime else inference
    else:
        return None

