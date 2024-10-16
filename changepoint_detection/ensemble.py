import warnings
from typing import Union, Optional
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
    scales = [0.005 * (5 * i) for i in range(1, 9)]  # 0.005, 0.025, 0.05, ..., 0.2
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
    warnings.warn("'proba' function will be deprecated in future", DeprecationWarning)

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
        proba = np.abs(np.nanmean(model.params["delta"], axis=0))
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
        all_trends.append(forecast["trend"])

    # 여러 모델로부터의 trend 평균 계산
    avg_trend = pd.concat(all_trends, axis=1).mean(axis=1)

    # 일정 임계값(threshold)을 넘는 changepoint들만 반환
    results = [
        changepoint for changepoint, proba in datetime_proba.items() if proba >= th
    ]

    if not results:
        return None

    max_changepoint = max(results)

    # get partial data after changepoint
    changepoint_index = forecast[forecast["ds"] == max_changepoint].index[0]
    post_changepoint_trend = avg_trend.iloc[changepoint_index + 1 :]

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
    last_datetime = df["ds"].max()
    n = (last_datetime - max_changepoint).days

    if n < len(df) * 0.1:
        return None

    # 결과 반환
    return {"n": n, "k": round(k, 2), "datetime": max_changepoint.date()}


def change_point_with_proba(
    df: pd.DataFrame,
    scales: Optional[list] = None,
    norm_method: str = "z-score",
    th: float = 2,
    random_seed: int = 9999,
) -> Union[None, dict]:
    """
    시계열 데이터에서 변화점을 추출하고 각 변화점의 확률을 계산합니다.

    Parameters:
    - df: 시계열 데이터프레임 (ds, y 컬럼 포함)
    - scales: changepoint_prior_scale 값들의 리스트
    - norm_method: 정규화 방법 ('z-score' 또는 'softmax')
    - th: 변화점 확률 임계값 (기본값: 2)
    - random_seed: 랜덤 시드

    Returns:
    - 변화점 정보가 담긴 딕셔너리 또는 None
    """
    np.random.seed(random_seed)

    if scales is None:
        scales = [0.005 * (5 * i) for i in range(1, 9)]  # 0.005, 0.025, 0.05, ..., 0.2
    n_scales = len(scales)
    pseudo_cps = {}
    all_trends = []

    # 각 scale에 대해 Prophet 모델을 학습하고 changepoints 및 proba 추출
    for scale in scales:
        model = Prophet(
            changepoint_prior_scale=scale,
            changepoint_range=1.0,
            daily_seasonality=False,
            yearly_seasonality=False,
        )
        model.fit(df)

        # 각 모델의 trend 예측 결과 저장
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        all_trends.append(forecast["trend"])

        # 모델에서 잠재 changepoints 추출
        changepoints = model.changepoints

        # changepoints의 Z-score 계산
        proba = np.abs(np.nanmean(model.params["delta"], axis=0))
        if norm_method == "z-score":
            proba = z_score_normalization(proba)
        elif norm_method == "softmax":
            proba = softmax(proba)
        else:
            raise NotImplementedError("Unsupported normalization method.")

        proba = np.where(proba < th, 0, proba)

        # changepoints와 proba를 datetime_proba에 누적
        for i, changepoint in enumerate(changepoints):
            if changepoint in pseudo_cps:
                pseudo_cps[changepoint] += proba[i] / n_scales
            else:
                pseudo_cps[changepoint] = proba[i] / n_scales

    # 평균 trend 계산
    avg_trend = pd.concat(all_trends, axis=1).mean(axis=1)
    # changepoint thresholding
    pseudo_cps = {cp: prob for cp, prob in pseudo_cps.items() if prob >= th}

    # 임계값 이상의 changepoint가 없는 경우 return None
    if not pseudo_cps:
        return None

    # 확률 기준으로 가장 높은 확률을 가진 changepoint 선택
    highest_proba_changepoint = max(pseudo_cps, key=pseudo_cps.get)
    highest_proba = pseudo_cps[highest_proba_changepoint]

    i_changepoint = df["ds"].tolist().index(highest_proba_changepoint)

    # N: timestamp interval (changepoint ~ last)
    last_datetime = df["ds"].max()
    n_days = (last_datetime - highest_proba_changepoint).days
    if (n_days < len(df) * 0.1) or (n_days > len(df) * 0.9):
        return None

    # changepoint 이전, 이후 trend
    pre_changepoint_trend = avg_trend.iloc[i_changepoint - 1 : i_changepoint]
    post_changepoint_trend = avg_trend.iloc[i_changepoint + 1 :]

    # trend 값의 최소 및 최대 계산
    max_trend_after = post_changepoint_trend.max()  # 이후 데이터의 최대 trend 값
    min_trend_after = post_changepoint_trend.min()  # 이후 데이터의 최소 trend 값

    # trend 값 at changepoint
    changepoint_trend = avg_trend.iloc[i_changepoint]

    # K: delta trend 계산
    if max_trend_after - changepoint_trend >= changepoint_trend - min_trend_after:
        k = (max_trend_after - changepoint_trend) / changepoint_trend * 100
    else:
        k = (min_trend_after - changepoint_trend) / changepoint_trend * 100

    # N: timestamp interval (changepoint ~ last)
    last_datetime = df["ds"].max()
    n_days = (last_datetime - highest_proba_changepoint).days

    if n_days < len(df) * 0.1:
        return None

    # 결과 반환
    return {
        "n": n_days,
        "k": round(k, 2),
        "datetime": highest_proba_changepoint.date(),
        "proba": round(highest_proba, 4),
    }


def proba_w_post(
    df,
    scales=None,
    method="auto",
    norm_method: str = "z-score",
    th: float = 2,
    **kwargs,
):
    pre_datetime = kwargs.get("pre_changepoint")
    if method == "past":
        inference = proba(df, norm_method, th)
    elif method == "auto":
        inference = change_point_with_proba(
            df, scales=scales, norm_method=norm_method, th=th, **kwargs
        )
    else:
        raise ValueError(f"Invalid input argument of method, '{method}'.")
    if inference:
        return None if inference["datetime"] == pre_datetime else inference
    else:
        return None
