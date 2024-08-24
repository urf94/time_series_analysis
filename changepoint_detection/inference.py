import os
import pickle
from .utils import load_model


def inference_prophet(df, scale, checkpoint_dir="checkpoint"):
    model_filename = f"prophet_scale_{scale}.pkl"
    model_path = os.path.join(checkpoint_dir, model_filename)

    # 모델 로드
    model = load_model(model_path)

    # 추론 (changepoint 확인)
    changepoints = model.predict(df[["ds"]])["changepoint"]

    return changepoints.dropna().tolist()


def inference_neuralprophet(df, checkpoint_dir="checkpoint"):
    model_filename = "neuralprophet.pkl"
    model_path = os.path.join(checkpoint_dir, model_filename)

    # 모델 로드
    model = load_model(model_path)

    # 추론 (changepoint 확인)
    changepoints = model.model.find_changepoints(df["y"].values)

    return changepoints.tolist()
