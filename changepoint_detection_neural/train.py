import os

from neuralprophet import NeuralProphet
from .utils import load_model, save_model


def train_neuralprophet(df, checkpoint_dir="checkpoint"):
    model_filename = "neuralprophet.pkl"
    model_path = os.path.join(checkpoint_dir, model_filename)

    # 모델 로드 (없으면 새로운 모델 생성)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_model(model_path)  # 이미 저장된 모델 로드
    else:
        print(f"No existing model found at {model_path}. Creating a new model.")
        model = NeuralProphet()

    # 추가 학습 (새 데이터를 사용해서 학습)
    model.fit(df, freq="D")

    # 학습 후 모델 저장
    save_model(model, model_path)
    print(f"Model saved after additional training to {model_path}")
