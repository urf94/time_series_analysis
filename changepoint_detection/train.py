import os
from prophet import Prophet
from .utils import load_model, save_model


def train_prophet(df, scale, checkpoint_dir="checkpoint"):
    model_filename = f"prophet_scale_{scale}.pkl"
    model_path = os.path.join(checkpoint_dir, model_filename)

    # 모델 로드 (없으면 새로운 모델 생성)
    if os.path.exists(model_path):
        print(f"Model {model_filename} already exists. Skipping training.")
        return  # 이미 존재하는 경우 학습을 건너뜁니다.

    model = Prophet(changepoint_prior_scale=scale)

    # 모델 학습
    model.fit(df[["ds", "y"]])

    # 모델 저장
    save_model(model, model_path)
    print(f"Prophet model with scale {scale} saved to {model_path}")
