# time_series_analysis

- [x] changepoint detection
- [ ] anomaly detection
- [ ] forecast
- [ ] trend classification

### 설치

```commandline

pip install https://github.com/urf94/time_series_analysis/releases/download/v0.1.8/time_series_analysis-0.1.8-py3-none-any.whl

pip install git+https://github.com/urf94/time_series_analysis.git@v0.1.8

```


## 사용법

### proba
subpackage `changepoint_detection` 내의 proba 함수를 통해 chagepoint를 감지하고 기준일 대비 기간 n과 증감폭 k를 반환받습니다.
- 입력 df에 대해 다양한 hyper parameter를 갖는 Prophet 모델을 학습합니다.
- 각 model마다 changepoint 후보군을 추론하고 확률을 계산합니다.
- 계산된 확률을 정규화 및 누적하여 최종 changepoint를 선택합니다.
- n: 마지막 데이터 시점 기준으로 changepoint가 발생한 날짜까지의 일수를 나타냅니다. 예를 들어, n=7이라면 changepoint가 마지막 날짜로부터 7일 전에 발생했다는 것을 의미합니다.
- k: changepoint에서의 트렌드 값과 이후 데이터에서의 최대/최소 트렌드 값의 차이를 백분율로 나타냅니다. k 값이 양수이면 증가하는 트렌드, 음수이면 감소하는 트렌드를 나타냅니다. 예를 들어, k=3.5는 3.5%의 증가를, k=-2.0은 2.0%의 감소를 의미합니다.


### Python

- PySpark DataFrame에서 분석하려는 컬럼 `A`와 `datetime` 컬럼을 선택한 후 pandas DataFrame으로 변환합니다.
- 변환된 DataFrame을 `changepoint_detection.inference` 함수에 전달하고 변곡점을 반환받습니다.


```python
from pyspark.sql import SparkSession
import pandas as pd
from changepoint_detection import proba

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

# proba 함수 호출
result = proba(pandas_df) # Defaut: norm_method="z-score" / th=2

# 분석 결과 출력
print(result)   # {"n": 8, "k": -15.4, "datetime": datetime.date(2022, 4, 10)}

```

### PySpark
위와 같은 동작을 Pandas DataFrame으로 변환 없이 PySpark에서 수행할 수 있습니다.


```python
from pyspark.sql import SparkSession
import pandas as pd
from changepoint_detection import proba

# Spark 세션 초기화
spark = SparkSession.builder.appName("TimeSeriesAnalysis").getOrCreate()

# 데이터프레임 예시
data = [
    ('2023-01-01', 10),
    ('2023-01-02', 15),
    ('2023-01-03', 12),
    # ... 더 많은 행들 ...
]
df = spark.createDataFrame(data, ["datetime", "A"])

# PySpark UDF 작성
def detect_changepoints(dates, values):
    try:
        # pandas 데이터프레임으로 변환
        pandas_df = pd.DataFrame({
            'ds': dates,
            'y': values
        })

        # proba 함수를 사용한 changepoint 감지
        result = proba(pandas_df, norm_method="z-score", th=2)
        
        if result:
            return result["n"], result["k"], result["datetime"]
        else:
            return None, None, None

# UDF를 사용하여 데이터프레임에 changepoints 추가
result_df = df.groupBy().applyInPandas(lambda pdf: pd.DataFrame({
    'datetime': pdf['datetime'],
    'changepoints': [detect_changepoints(pdf['datetime'].tolist(), pdf['A'].tolist())]
}), schema='datetime string, changepoints map<int, float, datetime.date>')

# 결과 출력
result_df.show(truncate=False)

```

주요 변경 사항
- `inference_prophet`와 `inference_neuralprophet` 대신 `proba` 함수 사용
- ~~예외 처리 추가. changepoint 감지에 실패한 경우 대응~~