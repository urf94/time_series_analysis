import os
import pickle
from .utils import load_model
import pandas as pd


def inference_prophet(df, scale, checkpoint_dir="checkpoint"):
    model_filename = f"prophet_scale_{scale}.pkl"
    model_path = os.path.join(checkpoint_dir, model_filename)

    # 모델 로드
    model = load_model(model_path)

    # 추론 (changepoint 확인)
    changepoints = model.changepoints

    return changepoints.tolist()


def inference_neuralprophet(df, checkpoint_dir="checkpoint"):
    model_filename = "neuralprophet.pkl"
    model_path = os.path.join(checkpoint_dir, model_filename)

    # 모델 로드
    model = load_model(model_path)

    # NeuralProphet의 예측 수행
    forecast = model.predict(df)

    # Changepoint를 추출하는 방식으로 트렌드의 변화를 확인
    if "trend_change" in forecast.columns:
        changepoints = forecast.loc[forecast["trend_change"].notnull(), "ds"].tolist()
    else:
        changepoints = []

    return changepoints
