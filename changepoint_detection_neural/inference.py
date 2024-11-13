import os

from pytorch_lightning import Trainer

from .utils import load_model


def inference_neuralprophet(df, checkpoint_dir=None):
    # 패키지 내 checkpoint 폴더를 절대 경로로 설정
    if checkpoint_dir is None:
        checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoint"))

    # 경로가 존재하는지 확인
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    model_filename = "neuralprophet.pkl"
    model_path = os.path.join(checkpoint_dir, model_filename)

    # 모델 로드
    model = load_model(model_path)

    # 새로운 Trainer 객체를 생성하여 default_root_dir 설정
    trainer = Trainer(default_root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs")),
                      logger=False)  # 로깅 비활성화

    # 모델에 새로운 Trainer 설정
    model.trainer = trainer

    # 추론 수행
    forecast = model.predict(df)

    return forecast['trend'].tolist()
