# time_series_analysis

Time Series Analysis 패키지는 시계열 데이터의 다양한 분석 기능을 제공합니다.
- [x] changepoint detection
- [ ] anomaly detection
- [ ] forecast
- [ ] trend classification

### 설치

```commandline

pip install https://github.com/urf94/time_series_analysis/releases/download/v0.2.1/time_series_analysis-0.2.1-py3-none-any.whl

pip install git+https://github.com/urf94/time_series_analysis.git@v0.2.1

```


## 사용법

### changepoint_detection
subpackage `changepoint_detection` 내의 `proba_w_post` 함수를 통해 입력 data에서 changepoint를 감지할 수 있습니다.

- 입력 df에 대해 다양한 hyper parameter를 갖는 Prophet 모델을 학습합니다.
- 각 model마다 changepoint 후보군을 추론하고 확률을 계산합니다.
- 계산된 확률을 정규화 및 누적하여 최종 changepoint를 선택합니다.


#### 파라미터
- `df (pd.DataFrame)`: 시계열 데이터프레임 (ds, y 컬럼 포함)
- `method (str)`: ad ('past' 또는 'auto', 기본값: 'past')
- `scales (Optional[list])`: changepoint_prior_scale 값들의 리스트 (기본값: [0.005, 0.025, 0.05, ..., 0.2])
- `norm_method (str)`: 정규화 방법 ('z-score' 또는 'softmax', 기본값: 'z-score')
- `th (float)`: 변화점 확률 임계값 (기본값: 2)

#### 출력
사용한 `method`에 따라 다른 출력을 반환받습니다.
- `past` :
  - `n` : 마지막 데이터 시점 기준으로 changepoint가 발생한 날짜까지의 일수를 나타냅니다. 예를 들어, n=7이라면 changepoint가 마지막 날짜로부터 7일 전에 발생했다는 것을 의미합니다.
  - `k` : changepoint에서의 트렌드 값과 이후 데이터에서의 최대/최소 트렌드 값의 차이를 백분율로 나타냅니다. k 값이 양수이면 증가하는 트렌드, 음수이면 감소하는 트렌드를 나타냅니다. (예를 들어, k=3.5는 3.5%의 증가를, k=-2.0은 2.0%의 감소를 의미합니다.)
- `auto` :
  - `n` : 마지막 데이터 시점 기준으로 changepoint가 발생한 날짜까지의 일수를 나타냅니다. 예를 들어, n=7이라면 changepoint가 마지막 날짜로부터 7일 전에 발생했다는 것을 의미합니다.
  - `k1` : changepoint 이전 data에서의 증감률. 단위는 _**[지표 / 일수]**_ 입니다.
  - `k2` : changepoint 이후 data에서의 증감률. 단위는 _**[지표 / 일수]**_ 입니다.
  - `delta` : changepoint 이전과 이후의 증감률 변화량입니다 (즉, k1과 k2의 차이이며 단위는 ***%p***입니다). 이 값은 k1과 k2의 부호가 같을 때만 계산되며, 다른 경우에는 `None`입니다.
  - `p` : model이 추정한 changepoint일 가능도입니다. `norm_method`가 _**softmax**_ 인 경우는 확률을 의미하며, _**z-score**_ 인 경우는 편차를 의미합니다.

## Usage

### Python

- PySpark DataFrame에서 분석하려는 컬럼 `A`와 `datetime` 컬럼을 선택한 후 pandas DataFrame으로 변환합니다.
- 변환된 DataFrame을 `changepoint_detection.inference` 함수에 전달하고 변곡점을 반환받습니다.


```python
from pyspark.sql import SparkSession
import datetime
import pandas as pd
from changepoint_detection import change_point_with_proba, proba_w_post

# Spark 세션 초기화
spark = SparkSession.builder.appName("TimeSeriesAnalysis").getOrCreate()

# 'datetime'과 'A' 컬럼을 포함하는 예시 PySpark 데이터프레임
data = [
    ('2023-01-01', 10),
    ('2023-01-02', 15),
    ('2023-01-03', 12),
    # ... 더 많은 행들 ...
]
df = spark.createDataFrame(data, ["datetime", "A"])

# pandas 데이터프레임으로 변환하고 컬럼명 변경
pandas_df = df.select("datetime", "A").toPandas()
pandas_df = pandas_df.rename(columns={"datetime": "ds", "A": "y"})

# change_point_with_proba 함수 호출
result = change_point_with_proba(pandas_df) # Default: norm_method="z-score" / th=2

# 후처리: 이전 changepoint와 같은 경우 None 
pre_changepoint = datetime.date.today()
result = proba_w_post(pandas_df, pre_changepoint, method="auto") # Default: norm_method="z-score" / th=2

if result:
    print(f"변경점 확률: {result['p']}")
    print(f"변경점 이전 추세 기울기 (k1): {result['k1']}")
    print(f"변경점 이후 추세 기울기 (k2): {result['k2']}")
    print(f"Delta (기울기 변화율): {result['delta']}%")
    print(f"N (일수 차이): {result['n']}")
else:
    print("임계값을 만족하는 변경점이 없습니다.")

```

```commandline
변경점 확률: 2.0
변경점 이전 추세 기울기 (k1): 0.52
변경점 이후 추세 기울기 (k2): 0.63
Delta (기울기 변화율): 11.54%p
N (일수 차이): 179
```


### 주요 변경 사항
- `proba`(Deprecated) : `proba_w_post` 함수에서 method=`past`(기본값)일 때 실행할 수 있지만 `auto`를 권장합니다.
- `change_point_with_proba` : 이제 가장 최근 날짜의 change point 대신 가장 확률이 큰 change point를 반환합니다.
