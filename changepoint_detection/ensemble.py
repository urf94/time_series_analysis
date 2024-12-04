import warnings
import datetime
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


class Ensembler:
    """
    Ensembler: Prophet 기반으로 시계열 데이터를 분석하고 변화점(changepoints) 및 이상치(outliers)를 탐지하며,
    미래 성과 예측을 수행하는 클래스.

    Attributes:
        cp_priors (Sequence): Changepoint prior scale 값들의 리스트. 기본값은 [0.025, 0.05, ..., 0.2].
        cp_proba_norm (str): 변화점 확률 계산에 사용할 정규화 함수. "z-score" 또는 "softmax".
        cp_threshold (float): 변화점 확률에 대한 임계값. 이 값보다 낮은 확률은 무시됨.
        smooth_kernel (tuple): 이상치 탐지를 위한 커널 컨볼루션의 가중치.
        confidence_intervals (tuple): 이상치 탐지 시 사용할 Prophet 신뢰구간 (기본: 95%, 99%).
        sparsity_check (bool): 데이터가 희소한지 확인하는 옵션. True일 경우 희소 데이터 검증 수행.
        debug (bool): 디버그 모드. True일 경우 중간 과정 출력.
    """
    def __init__(self,
                 cp_priors: Sequence = tuple(0.005 * (5 * i) for i in range(1, 9)),
                 cp_proba_norm: str = "z-score",
                 cp_threshold: float = 2,
                 smooth_kernel: tuple = (1/6, 2/3, 1/6),
                 confidence_intervals: tuple = (0.95, 0.99),
                 sparsity_check: bool = False,
                 debug: bool = False
                 ) -> None:
        """
        Initializes the Ensembler class.

        Args:
            cp_priors (Sequence): Changepoint prior scale 값들의 리스트.
            cp_proba_norm (str): 변화점 확률 계산에 사용할 정규화 함수.
            cp_threshold (float): 변화점 확률에 대한 임계값.
            smooth_kernel (tuple): 이상치 탐지를 위한 커널 가중치.
            confidence_intervals (tuple): Prophet 신뢰구간의 리스트.
            sparsity_check (bool): 데이터 희소성 검사 여부.
            debug (bool): 디버그 모드 활성화 여부.
        """

        self.today = datetime.datetime.now()
        _, last_day_of_month = calendar.monthrange(self.today.year, self.today.month)
        last_date_of_month = datetime.datetime(self.today.year, self.today.month, last_day_of_month)
        self.forecast_days = (last_date_of_month - self.today).days + 1  # 오늘 포함하려면 +1

        self.cp_priors = cp_priors
        self.cp_proba_norm = cp_proba_norm
        self.cp_threshold = cp_threshold
        self.smooth_kernel = smooth_kernel
        self.confidence_intervals = confidence_intervals
        self.sparsity_check = sparsity_check
        self.debug = debug

        self.norm_func = z_score_normalization if cp_proba_norm == "z-score" else softmax

    def sanity_check(self, df) -> bool:
        """
        데이터 유효성 검사를 수행.

        Args:
            df (pd.DataFrame): 입력 데이터프레임 (ds: 날짜, y: 값).

        Returns:
            bool: 데이터가 유효하면 True, 그렇지 않으면 False.
        """
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
        """
        여러 Prophet 모델을 학습하여 변화점 확률 계산 및 병합.

        Args:
            df (pd.DataFrame): 입력 데이터프레임 (ds: 날짜, y: 값).
            method (str): 확률 병합 방식 (현재 미사용).

        Returns:
            dict: 후처리된 변화점 데이터.
        """
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
        """
        변화점 후보를 필터링 및 후처리.

        Args:
            df (pd.DataFrame): 입력 데이터프레임.
            cp_candidates (dict): 변화점 후보 (changepoint와 확률의 딕셔너리).

        Returns:
            tuple: 가장 높은 확률의 변화점 정보 (날짜, 확률, 인덱스, 변화 이후 일수).
        """
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

    def detect_outliers(self, df):
        """
        Prophet 신뢰구간과 커널 컨볼루션을 기반으로 이상치를 탐지.

        Args:
            df (pd.DataFrame): 입력 데이터프레임 (ds: 날짜, y: 값).

        Returns:
            set: 이상치로 탐지된 날짜들의 집합.
        """

        conf_ = {}
        for interval in self.confidence_intervals:
            outlier_model = Prophet(
                changepoint_prior_scale=1,  # 필요에 따라 조정
                changepoint_range=1.0,
                interval_width=interval,
                daily_seasonality=False,
                yearly_seasonality=False,
            )
            outlier_model.fit(df)
            future_ = outlier_model.make_future_dataframe(periods=0)
            forecast_ = outlier_model.predict(future_)
            conf_[interval] = (forecast_["yhat_lower"], forecast_["yhat_upper"])

        # 커널 컨볼루션 적용
        smoothed_y = np.convolve(df["y"], self.smooth_kernel, mode="same")

        # 이상치 판별
        i_out95 = (smoothed_y < conf_[0.95][0]) | (smoothed_y > conf_[0.95][1])
        i_out99 = (df["y"] < conf_[0.99][0]) | (df["y"] > conf_[0.99][1])
        out95 = set(df["ds"][i_out95])
        out99 = set(df["ds"][i_out99])
        out = out95.union(out99)
        # 두 방식에서 탐지된 이상치 병합
        return out

    def forecast_month_performance(self, df, forecast):
        """
        현재 월의 마지막 날까지 성과 예측.

        Args:
            df (pd.DataFrame): 입력 데이터프레임 (ds: 날짜, y: 값).
            forecast (pd.DataFrame): Prophet 예측 결과.

        Returns:
            dict: 실제 값 및 예측 값의 합계와 신뢰구간.
        """
        current_month = pd.to_datetime(df["ds"]).dt.month.max()
        current_year = pd.to_datetime(df["ds"]).dt.year.max()

        # 현재 월 필터링
        mask = (pd.to_datetime(forecast["ds"]).dt.month == current_month) & \
               (pd.to_datetime(forecast["ds"]).dt.year == current_year)
        monthly_forecast = forecast[mask]
        monthly_actual = df[df["ds"].isin(monthly_forecast["ds"])]

        # 실제값과 예측값의 합계 계산
        actual = monthly_actual["y"].sum()
        predict = actual + monthly_forecast["yhat"].sum()
        predict_lower = actual + monthly_forecast["yhat_lower"].sum()
        predict_upper = actual + monthly_forecast["yhat_upper"].sum()

        return {'actual': actual, 'predict': predict, 'predict_lower': predict_lower, 'predict_upper': predict_upper}

    def __call__(self, df: pd.DataFrame) -> Union[None, dict]:
        """
        Ensembler 클래스를 호출하여 전체 프로세스를 실행.

        Args:
            df (pd.DataFrame): 입력 데이터프레임 (ds: 날짜, y: 값).

        Returns:
            dict: 분석 결과. 포함되는 항목:
                - n (int): 변화 이후 일수.
                - datetime (str): 변화점 날짜.
                - k1 (float): 변화 이전의 기울기.
                - k2 (float): 변화 이후의 기울기.
                - delta (float): 기울기 변화율.
                - p (float): 변화점 확률.
                - trend (list): 예측된 추세.
                - outliers (set): 이상치 날짜 집합.
                - forecast (dict): 월별 성과 예측 결과 (실제 값, 예측 값, 신뢰구간).
        """
        sanity = self.sanity_check(df)
        if not sanity:
            return None

        ensemble_result = self.ensembles(df)
        if ensemble_result is None:
            return None
        max_proba_cp, max_proba, i_cp, n_days = ensemble_result
        outliers = self.detect_outliers(df)
        #
        fin_model = Prophet(
            changepoint_prior_scale=1,  # 필요에 따라 조정
            changepoint_range=1.0,
            interval_width=0.80,
            daily_seasonality=False,
            yearly_seasonality=False,
            changepoints=[max_proba_cp],
        )
        fin_model.fit(df)
        fin_future = fin_model.make_future_dataframe(periods=self.forecast_days)
        fin_forecast = fin_model.predict(fin_future)

        forecast_dict = self.forecast_month_performance(df, fin_forecast)

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
            "forecast": forecast_dict,
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
