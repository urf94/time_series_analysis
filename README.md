# Time Series Analysis

Time Series Analysis 패키지는 시계열 데이터의 다양한 분석 기능을 제공합니다.

- [x] **Changepoint Detection**: 시계열 데이터의 변화점을 탐지합니다.
- [x] **Anomaly Detection**: 시계열 데이터에서 이상치를 탐지합니다.
- [x] **Forecasting**: 미래의 값을 예측합니다.
- [ ] **Trend Classification**: 데이터의 트렌드를 분류 (향후 지원 예정).

## 설치
```bash
pip install https://github.com/urf94/time_series_analysis/releases/download/v0.2.3/time_series_analysis-0.2.3-py3-none-any.whl
pip install git+https://github.com/urf94/time_series_analysis.git@v0.2.3
```

## 입출력

### 1. change_point_proba

**INPUT 파라미터**

- `df (pd.DataFrame)`: 시계열 데이터프레임. 날짜 열(`ds`)과 값 열(`y`)을 포함해야 합니다.
- `scales (Optional[list])`: changepoint_prior_scale 값들의 리스트. Prophet 모델 학습 시 사용되며, 기본값은 [0.005, 0.025, 0.05, ..., 0.2]입니다.
- `norm_method (str)`: 정규화 방법. `z-score` 또는 `softmax` 중 선택 가능. (기본값: `z-score`).
- `th (float)`: 변화점 확률 임계값. 이 값 이상인 변화점만 결과에 포함됩니다. (기본값: `2`).

**OUTPUT 파라미터**
- `n`: 마지막 데이터 시점 기준으로 변화점이 발생한 날짜까지의 일수. 
  - 예: `n=7` → 마지막 날짜로부터 7일 전에 발생.
- `k1`: 변화점 이전 데이터의 증감률. 단위는 `[지표 / 일수]`.
- `k2`: 변화점 이후 데이터의 증감률. 단위는 `[지표 / 일수]`.
- `delta`: 변화점 이전과 이후의 증감률 변화량. 단위는 `%p`.
  - `delta는` `k1과` `k2의` 부호가 같을 때만 계산되며, 다르면 `None`.
- `p`: 모델이 추정한 변화점 가능도.
  - `norm_method="softmax"`인 경우 확률 값.
  - `norm_method="z-score"`인 경우 편차 값.

### 2. Ensembler
**INPUT 파라미터**
- `cp_priors (Sequence)`: Prophet 모델 학습 시 사용할 `changepoint_prior_scale` 값들의 리스트. 작을수록 모델이 변화점을 덜 감지하며, 클수록 더 많은 변화점을 감지. *기본값: `[0.025, 0.05, ..., 0.2]`*
- `cp_proba_norm (str)`: 변화점 확률을 계산할 때 사용할 정규화 함수. z-score 또는 softmax 중 선택 가능. *기본값: `z-score`*
- `cp_threshold (float)`: 변화점 확률의 임계값. 이 값보다 낮은 확률은 무시. *기본값: `2`*
- `smooth_kernel (tuple)`: 이상치 탐지를 위한 커널 컨볼루션의 가중치. *기본값: `(1/6, 2/3, 1/6)`*
- `confidence_intervals (tuple)`: Prophet 이상치 탐지 시 사용할 신뢰구간의 리스트. *기본값: `(0.95, 0.99)`*
- `sparsity_check (bool)`: 데이터가 희소한지 검사할지 여부. *기본값: `False`*
- `debug (bool)`: 디버그 모드 활성화 여부. 활성화 시 중간 과정 출력. *기본값: `False`*

**OUTPUT 파라미터**

Ensembler의 결과는 다음 항목을 포함하는 딕셔너리로 반환됩니다.

- `n`: 마지막 데이터 시점 기준으로 변화점이 발생한 날짜까지의 일수.
- `datetime`: 변화점 날짜.
- `k1`: 변화점 이전 데이터의 증감률. 단위는 `[지표 / 일수]`.
- `k2`: 변화점 이후 데이터의 증감률. 단위는 `[지표 / 일수]`.
- `delta`: 변화점 이전과 이후의 증감률 변화량. 단위는 `%p`.
- `p`: 모델이 추정한 변화점 가능도.
  - `cp_proba_norm="softmax"`인 경우 확률 값.
  - `cp_proba_norm="z-score"`인 경우 편차 값.
- `trend`: 예측된 전체 추세 값의 리스트.
- `outliers`: 이상치로 탐지된 날짜들의 집합.
- `forecast`: 월별 성과 예측 결과 (실제값, 예측값, 신뢰구간 포함).

## 사용법

### 1. Changepoint Detection
시계열 데이터에서 Prophet 모델 기반으로 변화점을 탐지하고, 해당 변화점의 확률을 계산합니다.

#### 사용 방법
- **구버전 (change_point_with_proba)**: 주어진 데이터에 대해 다양한 `changepoint_prior_scale` 값을 사용해 Prophet 모델을 학습하고 변화점을 탐지합니다.
- **신버전 (Ensembler)**: 여러 Prophet 모델을 조합하여 가장 가능성이 높은 변화점을 탐지합니다.

#### 예제 코드
```python
import pandas as pd
from changepoint_detection import change_point_with_proba, Ensembler

# 예제 데이터 생성
data = {
    "ds": pd.date_range(start="2024-01-01", periods=100),
    "y": [i + (i * 0.1) for i in range(100)]
}
df = pd.DataFrame(data)

# 구버전 사용법: change_point_with_proba
result_old = change_point_with_proba(
    df, 
    scales=[0.005, 0.01, 0.05], 
    norm_method="z-score", 
    th=2, 
    debug=True
)
if result_old:
    print("구버전 결과:")
    print(f"변경점 날짜: {result_old['datetime']}")
    print(f"기울기 변화율 (delta): {result_old['delta']}%")
else:
    print("구버전 결과: 변경점이 없습니다.")

# 신버전 사용법: Ensembler
ensembler = Ensembler(cp_priors=[0.005, 0.01, 0.05], cp_threshold=2, debug=True)
result_new = ensembler(df)
if result_new:
    print("\n신버전 결과:")
    print(f"변경점 날짜: {result_new['datetime']}")
    print(f"기울기 변화율 (delta): {result_new['delta']}%")
    print(f"월별 성과 예측: {result_new['forecast']}")
    print(f"이상치 목록: {result_new['outliers']}")
else:
    print("신버전 결과: 변경점이 없습니다.")
```

### 2. Anomaly Detection
Prophet 모델의 신뢰구간과 커널 컨볼루션을 활용하여 이상치를 탐지합니다.

#### 특징
- Prophet의 95% 및 99% 신뢰구간을 기준으로 이상치를 판별합니다.
- 커널 컨볼루션을 사용하여 이상치 탐지를 개선합니다.

#### 예제 코드
```python
from changepoint_detection import Ensembler

# Ensembler 사용법 (이상치 탐지 포함)
ensembler = Ensembler(debug=True)
result = ensembler(df)

if result:
    print(f"이상치 목록: {result['outliers']}")
else:
    print("이상치가 없습니다.")
```

### 3. Forecasting
Prophet 모델을 활용하여 현재 월의 마지막 날짜까지 예측합니다.

#### 특징
- 입력된 데이터로부터 미래 값을 예측합니다.
- 월별 실제값과 예측값(신뢰구간 포함)을 반환합니다.

#### 예제 코드
```python
from changepoint_detection import Ensembler

# Ensembler 사용법 (예측 포함)
ensembler = Ensembler(debug=True)
result = ensembler(df)

if result:
    print(f"월별 성과 예측: {result['forecast']}")
else:
    print("예측을 수행할 데이터가 충분하지 않습니다.")
```

## 결과 예시

### Ensembler 결과
```plain
변경점 날짜: 2024-02-15
기울기 변화율 (delta): 12.34%
월별 성과 예측: {'actual': 500, 'predict': 550, 'predict_lower': 530, 'predict_upper': 570}
이상치 목록: {'2024-01-15', '2024-02-05'}
```

### change_point_with_proba 결과
```plain
변경점 날짜: 2024-02-15
기울기 변화율 (delta): 11.95%
```

## Parameter
### Input
