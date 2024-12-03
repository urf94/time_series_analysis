import warnings
import datetime
from datetime import datetime
import calendar
from collections import defaultdict
from typing import Union, Optional, Sequence
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


def proba_depreceted(
    df, norm_method: str = "z-score", th: float = 2
) -> Union[None, dict]:
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


class Ensembler:
    def __init__(self,
                 cp_priors: Sequence = tuple(0.005 * (5 * i) for i in range(1, 9)),
                 cp_proba_norm: str = "z-score",
                 cp_threshold: float = 2,
                 smooth_kernel: tuple = (1/6, 2/3, 1/6),
                 confidence_interval: float = 0.95,
                 sparsity_check: bool = False,
                 debug: bool = False
                 ) -> None:

        self.today = datetime.now()
        _, last_day_of_month = calendar.monthrange(self.today.year, self.today.month)
        last_date_of_month = datetime(self.today.year, self.today.month, last_day_of_month)
        self.forecast_days = (last_date_of_month - self.today).days + 1  # 오늘 포함하려면 +1

        self.cp_priors = cp_priors
        self.cp_proba_norm = cp_proba_norm
        self.cp_threshold = cp_threshold
        self.smooth_kernel = smooth_kernel
        self.confidence_interval = confidence_interval
        self.sparsity_check = sparsity_check
        self.debug = debug

        self.norm_func = z_score_normalization if cp_proba_norm == "z-score" else softmax

    def sanity_check(self, df) -> bool:
        try:
            df["ds"] = pd.to_datetime(df["ds"]).dt.strftime("%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Error converting 'ds' column to 'yyyy-mm-dd' format: {e}")

        try:
            df["y"] = df["y"].fillna(0)
        except Exception as e:
            raise ValueError(f"Error occurs during fill NA value with 0: {e}")

        if len(df["ds"].unique()) < 90:
            if self.debug:
                print(f"Input data needs more than 90 data points.")
            return False

        if empty_point := sum(df["y"] == 0) > 45 and self.sparsity_check:
            if self.debug:
                print(f"Input data is too sparse. {empty_point} points is empty.")
            return False
        return True

    def ensembles(self, df, method='soft'):
        n_scales = len(self.cp_priors)
        pseudo_cps = defaultdict(float)

        # 각 scale에 대해 Prophet 모델을 학습하고 changepoints 및 proba 추출
        for scale in self.cp_priors:
            model = Prophet(
                changepoint_prior_scale=scale,
                changepoint_range=1.0,
                daily_seasonality=False,
                yearly_seasonality=False,
            )
            model.fit(df)

            proba = self.norm_func(np.abs(np.nanmean(model.params["delta"], axis=0)))
            proba = np.where(proba < self.cp_threshold, 0, proba)
            if self.debug:
                print(f"Filtered proba with scale {scale} : {proba}")

            # changepoints와 proba를 pseudo_cps에 누적
            for i, changepoint in enumerate(model.changepoints):
                changepoint = str(changepoint.date())
                pseudo_cps[changepoint] += proba[i] / n_scales
        return self.postprocessing(df, pseudo_cps)

    def postprocessing(self, df, cp_candidates: dict):
        # changepoint thresholding
        if self.debug:
            print(f"Pseudo CPs before thresholding: {cp_candidates}")

        pseudo_cps = {cp: prob for cp, prob in cp_candidates.items() if prob >= self.cp_threshold}
        if self.debug:
            print(f"Pseudo CPs after thresholding: {pseudo_cps}")

        if not pseudo_cps:
            if self.debug:
                print("No changepoint meets the threshold.")
            return None

        max_proba_cp = max(pseudo_cps, key=pseudo_cps.get)
        max_proba = pseudo_cps[max_proba_cp]

        # N: timestamp interval (changepoint ~ last)
        last_datetime_str = df["ds"].max()
        last_datetime = datetime.datetime.strptime(last_datetime_str, "%Y-%m-%d")
        changepoint_datetime = datetime.datetime.strptime(max_proba_cp, "%Y-%m-%d")
        n_days = (last_datetime - changepoint_datetime).days

        if (n_days < len(df) * 0.1) or (n_days > len(df) * 0.9):
            if self.debug:
                print(f"Changepoint {max_proba_cp} is too close to the end. n_days: {n_days}")
            return None

        try:
            i_changepoint = df["ds"].tolist().index(max_proba_cp)
        except ValueError:
            # Change point not found in 'ds' column
            if self.debug:
                print(f"Cannot find changepoint {max_proba_cp} in 'ds' column.")
            return None

        if i_changepoint < 2 or len(df) - i_changepoint < 2:
            if self.debug:
                print("Insufficient data points before or after changepoint.")
            return None

        return max_proba_cp, max_proba, i_changepoint, n_days

    def detect_outliers(self, df, forecast):
        """
        이상치 탐지: 2가지 방식의 이상치를 추출하여 병합
        Args:
            df (pd.DataFrame): 입력 데이터프레임 (ds, y)
            forecast (pd.DataFrame): Prophet 예측 결과
        Returns:
            outliers (list): 이상치로 판별된 날짜 리스트
        """
        # Prophet 신뢰구간 추출
        yhat = forecast["yhat"]
        yhat_lower_95 = forecast["yhat_lower"]
        yhat_upper_95 = forecast["yhat_upper"]
        yhat_lower_99 = forecast["yhat_lower"] - 2 * (yhat - yhat_lower_95)
        yhat_upper_99 = forecast["yhat_upper"] + 2 * (yhat_upper_95 - yhat)

        # 커널 컨볼루션 적용
        smoothed_y = np.convolve(df["y"], self.smooth_kernel, mode="same")

        # 이상치 판별
        outliers_95 = df["ds"][
            (smoothed_y < yhat_lower_95) | (smoothed_y > yhat_upper_95)
            ].tolist()
        outliers_99 = df["ds"][
            (df["y"] < yhat_lower_99) | (df["y"] > yhat_upper_99)
            ].tolist()

        # 두 방식에서 탐지된 이상치 병합
        return list(set(outliers_95 + outliers_99))

    def forecast_month_performance(self, df, forecast):
        """
        현재 월의 마지막 날까지 성과 예측
        Args:
            df (pd.DataFrame): 입력 데이터프레임 (ds, y)
            forecast (pd.DataFrame): Prophet 예측 결과
        Returns:
            actual_sum (float): 해당 월의 실제 y의 합계
            predicted_sum (float): 해당 월의 예측 yhat의 합계
        """
        current_month = pd.to_datetime(df["ds"]).dt.month.max()
        current_year = pd.to_datetime(df["ds"]).dt.year.max()

        # 현재 월 필터링
        mask = (pd.to_datetime(forecast["ds"]).dt.month == current_month) & \
               (pd.to_datetime(forecast["ds"]).dt.year == current_year)
        monthly_forecast = forecast[mask]
        monthly_actual = df[df["ds"].isin(monthly_forecast["ds"])]

        # 실제값과 예측값의 합계 계산
        actual_sum = monthly_actual["y"].sum()
        predicted_sum = monthly_forecast["yhat"].sum()

        return actual_sum, predicted_sum

    def __call__(self, df: pd.DataFrame) -> Union[None, dict]:
        sanity = self.sanity_check(df)
        if not sanity:
            return None

        ensemble_result = self.ensembles(df)
        if ensemble_result is None:
            return None
        max_proba_cp, max_proba, i_cp, n_days = ensemble_result

        fin_model = Prophet(
            changepoint_prior_scale=1,  # 필요에 따라 조정
            changepoint_range=1.0,
            interval_width=self.confidence_interval,
            daily_seasonality=False,
            yearly_seasonality=False,
            changepoints=[max_proba_cp],
        )
        fin_model.fit(df)
        fin_future = fin_model.make_future_dataframe(periods=self.forecast_days)
        fin_forecast = fin_model.predict(fin_future)

        outliers = self.detect_outliers(df, fin_forecast)
        actual_sum, predicted_sum = self.forecast_month_performance(df, fin_forecast)

        # 기존 결과 추출
        df_trend = pd.DataFrame({"ds": df["ds"], "trend": fin_forecast["trend"]})
        pre_trend = df_trend.iloc[:i_cp]["trend"].tolist()
        post_trend = df_trend.iloc[i_cp:]["trend"].tolist()
        post_trend = [t for t in post_trend if not pd.isna(t)]

        k1 = (pre_trend[-1] - pre_trend[0]) / (len(pre_trend) - 1)
        k2 = (post_trend[-1] - post_trend[0]) / (len(post_trend) - 1)

        delta = round(100 * k2 / k1, 2) if k1 * k2 > 0 else None

        # 결과 반환
        return {
            "n": n_days,
            "datetime": max_proba_cp,
            "k1": round(k1, 2),
            "k2": round(k2, 2),
            "delta": delta,
            "p": round(max_proba, 2),
            "trend": fin_forecast["trend"].tolist(),
            "outliers": outliers,
            "performance": {
                "actual_sum": actual_sum,
                "predicted_sum": predicted_sum,
            },
        }


def change_point_with_proba(
    df: pd.DataFrame,
    scales: Optional[list] = None,
    norm_method: str = "z-score",
    th: float = 2,
    sparsity: bool = False,
    debug: bool = False,
) -> Union[None, dict]:
    """
    시계열 데이터에서 변화점을 추출하고 각 변화점의 확률을 계산합니다.
    가장 높은 확률의 변화점만을 반영한 트렌드의 기울기를 계산합니다.

    Parameters:
    - df: 시계열 데이터프레임 (ds, y 컬럼 포함)
    - scales: changepoint_prior_scale 값들의 리스트
    - norm_method: 정규화 방법 ('z-score' 또는 'softmax')
    - th: 변화점 확률 임계값 (기본값: 2)
    - debug: 디버그 출력 여부 (기본값: False)

    Returns:
    - dict: 변화점 정보가 담긴 딕셔너리
        {
            "n": int,          # 변화점과 마지막 날짜 간의 일수 차이
            "datetime": str,   # 변화점 날짜 (yyyy-mm-dd)
            "k1": float,       # 변화점 이전 추세의 기울기
            "k2": float,       # 변화점 이후 추세의 기울기
            "delta": float,    # 추세 기울기의 변화율 (k2 - k1) * 100
            "p": float         # 변화점의 확률
        }
        또는 None
    """

    # 입력 데이터프레임 복사 및 'ds' 컬럼 변환
    df = df.copy()
    try:
        df["ds"] = pd.to_datetime(df["ds"]).dt.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Error converting 'ds' column to 'yyyy-mm-dd' format: {e}")

    try:
        df["y"] = df["y"].fillna(0)
    except Exception as e:
        raise ValueError(f"Error occurs during fill NA value with 0: {e}")

    if len(df["ds"].unique()) < 90:
        if debug:
            print(f"Input data needs more than 90 data points.")
        return None

    if empty_point := sum(df["y"] == 0) > 45 and sparsity:
        if debug:
            print(f"Input data is too sparse. {empty_point} points is empty.")
        return None

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

        # 모델에서 잠재 changepoints 추출 및 문자열로 변환
        changepoints = model.changepoints

        # changepoints의 Z-score 계산
        proba = np.abs(np.nanmean(model.params["delta"], axis=0))
        if norm_method == "z-score":
            proba = z_score_normalization(proba)
        elif norm_method == "softmax":
            proba = softmax(proba)
        else:
            raise NotImplementedError("Unsupported normalization method.")

        # 임계값 필터링
        proba = np.where(proba < th, 0, proba)
        if debug:
            print(f"Filtered proba with scale {scale} : {proba}")

        # changepoints와 proba를 pseudo_cps에 누적
        for i, changepoint in enumerate(changepoints):
            changepoint = str(changepoint.date())
            if changepoint in pseudo_cps:
                pseudo_cps[changepoint] += proba[i] / n_scales
            else:
                pseudo_cps[changepoint] = proba[i] / n_scales

    # 평균 trend 계산
    avg_trend = pd.concat(all_trends, axis=1).mean(axis=1)

    # changepoint thresholding
    if debug:
        print(f"Pseudo CPs before thresholding: {pseudo_cps}")
    pseudo_cps = {cp: prob for cp, prob in pseudo_cps.items() if prob >= th}

    if debug:
        print(f"Pseudo CPs after thresholding: {pseudo_cps}")

    # 임계값 이상의 changepoint가 없는 경우 return None
    if not pseudo_cps:
        if debug:
            print("No changepoint meets the threshold.")
        return None

    # 확률 기준으로 가장 높은 확률을 가진 changepoint 선택
    highest_proba_changepoint = max(pseudo_cps, key=pseudo_cps.get)
    highest_proba = pseudo_cps[highest_proba_changepoint]

    # Find the index of the changepoint
    try:
        i_changepoint = df["ds"].tolist().index(highest_proba_changepoint)
    except ValueError:
        # Change point not found in 'ds' column
        if debug:
            print(
                f"Cannot find changepoint {highest_proba_changepoint} in 'ds' column."
            )
        return None

    # N: timestamp interval (changepoint ~ last)
    last_datetime_str = df["ds"].max()
    last_datetime = datetime.datetime.strptime(last_datetime_str, "%Y-%m-%d")
    changepoint_datetime = datetime.datetime.strptime(
        highest_proba_changepoint, "%Y-%m-%d"
    )
    n_days = (last_datetime - changepoint_datetime).days

    if (n_days < len(df) * 0.1) or (n_days > len(df) * 0.9):
        if debug:
            print(
                f"Changepoint {highest_proba_changepoint} is too close to the end. n_days: {n_days}"
            )
        return None

    # Prophet 모델을 재구성하여 선택된 changepoint만 포함
    model_single_cp = Prophet(
        changepoint_prior_scale=1,  # 필요에 따라 조정
        changepoint_range=1.0,
        daily_seasonality=False,
        yearly_seasonality=False,
        changepoints=[highest_proba_changepoint],
    )
    model_single_cp.fit(df)
    future_single_cp = model_single_cp.make_future_dataframe(periods=0)
    forecast_single_cp = model_single_cp.predict(future_single_cp)
    trend_single_cp = forecast_single_cp["trend"]

    # 추세 시리즈를 DataFrame에 추가
    df_trend = pd.DataFrame({"ds": df["ds"], "trend": trend_single_cp})

    # 변화점 전후의 트렌드 추출
    pre_trend = df_trend.iloc[:i_changepoint]["trend"].tolist()
    post_trend = df_trend.iloc[i_changepoint:]["trend"].tolist()
    post_trend = [t for t in post_trend if not pd.isna(t)]

    # 기울기 계산 (max - min) / (len(df) - 1)
    if len(pre_trend) < 2 or len(post_trend) < 2:
        if debug:
            print("Insufficient data points before or after changepoint.")
        return None
    k1 = (pre_trend[-1] - pre_trend[0]) / (len(pre_trend) - 1)
    k2 = (post_trend[-1] - post_trend[0]) / (len(post_trend) - 1)

    # delta 계산: 추세 기울기의 변화율
    # delta = round(100 * (k2 - k1), 2) if k1 * k2 > 0 else None
    delta = round(100 * k2 / k1, 2) if k1 * k2 > 0 else None

    # 결과 반환
    return {
        "n": n_days,
        "datetime": highest_proba_changepoint,
        "k1": round(k1, 2),
        "k2": round(k2, 2),
        "delta": delta,
        "p": round(highest_proba, 2),
        "trend": trend_single_cp.tolist(),
    }


def proba_w_post(
    df,
    scales=None,
    method="past",
    norm_method: str = "z-score",
    th: float = 2,
    **kwargs,
):
    pre_datetime = kwargs.get("pre_changepoint")
    if method == "past":
        inference = proba_depreceted(df, norm_method, th)
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
