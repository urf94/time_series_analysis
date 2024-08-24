import os
import pandas as pd
from changepoint_detection.train import train_prophet, train_neuralprophet

# 체크포인트 디렉토리 생성
checkpoint_dir = "checkpoint"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Dummy 데이터 생성 (일일 데이터를 기반으로)
data = {
    'ds': pd.date_range(start='2022-01-01', periods=100, freq='D'),
    'y': [i + (i % 10) * 0.5 for i in range(100)]  # 간단한 변동성 있는 데이터
}
df = pd.DataFrame(data)

# Prophet 모델을 다양한 changepoint_prior_scale로 학습 및 저장
for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    train_prophet(df, scale, checkpoint_dir=checkpoint_dir)

# NeuralProphet 모델 학습 및 저장
train_neuralprophet(df, checkpoint_dir=checkpoint_dir)

print("Models have been trained and saved to the checkpoint directory.")
