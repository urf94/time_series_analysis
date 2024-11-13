import os

from .utils import load_model

def inference_prophet(df, scale, checkpoint_dir=None):
    # 패키지 내 checkpoint 폴더를 기본값으로 설정
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoint")
    model_filename = f"prophet_scale_{scale}.pkl"
    model_path = os.path.join(checkpoint_dir, model_filename)

    # 모델 로드
    model = load_model(model_path)

    # 모델에서 감지된 changepoints 추출
    changepoints = model.changepoints

    return changepoints.tolist()
