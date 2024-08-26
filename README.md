# time_series_analysis

- [x] changepoint detection
- [ ] anomaly detection
- [ ] forecast
- [ ] trend classification

### 설치

```commandline
pip install https://github.com/urf94/time_series_analysis/releases/download/v0.1.5/time_series_analysis-0.1.5-py3-none-any.whl
```


## 사용법

### Python

- PySpark DataFrame에서 분석하려는 컬럼 `A`와 `datetime` 컬럼을 선택한 후 pandas DataFrame으로 변환합니다.
- 변환된 DataFrame을 `changepoint_detection.inference` 함수에 전달하고 변곡점을 반환받습니다.


```python
from pyspark.sql import SparkSession
import pandas as pd
from time_series_analysis.changepoint_detection import inference_prophet, inference_neuralprophet

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

# Inference 함수 호출 (Prophet 사용 예시)
changepoints = inference_prophet(pandas_df, scale=0.1, checkpoint_dir="checkpoint")

# 감지된 changepoint 출력
print(changepoints)

# Inference 함수 호출 (Neural Prophet 사용 예시)
changepoints = inference_neuralprophet(pandas_df, checkpoint_dir="checkpoint")

# 감지된 changepoint 출력
print(changepoints)
```

### PySpark UDF
위와 같은 동작을 PySpark UDF를 통해 수행할 수 있습니다. 


```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import pandas as pd
from time_series_analysis.changepoint_detection import inference_prophet

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
    # pandas 데이터프레임으로 변환
    pandas_df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    # Prophet을 사용한 changepoint 감지
    changepoints = inference_prophet(pandas_df, scale=0.1, checkpoint_dir="checkpoint")
    
    # changepoints를 리스트로 반환
    return changepoints

# UDF 등록 (ArrayType(StringType())은 changepoints를 리스트로 반환할 것을 가정)
changepoint_udf = udf(detect_changepoints, ArrayType(StringType()))

# UDF를 사용하여 데이터프레임에 changepoints 추가
result_df = df.groupBy().applyInPandas(lambda pdf: pd.DataFrame({
    'datetime': pdf['datetime'],
    'changepoints': [detect_changepoints(pdf['datetime'].tolist(), pdf['A'].tolist())]
}), schema='datetime string, changepoints array<string>')

# 결과 출력
result_df.show(truncate=False)

```
