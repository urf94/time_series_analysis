import os
import pickle
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def load_model(model_path, model_class=None, **kwargs):
    """Load model from the given path. If not found, create a new model instance."""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
    else:
        if model_class:
            model = model_class(**kwargs)
            print(f"New model created for {model_class.__name__} with parameters {kwargs}")
        else:
            raise ValueError("Model not found and no model_class provided for creation.")
    return model


def save_model(model, model_path):
    """Save the model to the given path."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def vis(df, prophet_df, forecast_days):
    fig, ax = plt.subplots(figsize=(32, 8))

    # 원본 데이터
    ax.plot(df['order_date'], df['total_price'], label='Original Sales', color='blue')

    # 예측된 값과 신뢰 구간
    ax.plot(prophet_df['ds'].iloc[:-forecast_days + 1], prophet_df['yhat'].iloc[:-forecast_days + 1],
            label='Fitted Sales', color='red')
    # ax.plot(view_forecast['ds'].iloc[:-forecast_days+1], view_forecast["trend"].iloc[:-forecast_days+1], label='Trend Line')
    ax.plot(prophet_df['ds'].iloc[-forecast_days:], prophet_df['yhat'].iloc[-forecast_days:], linestyle='dashed',
            label='Predicted Sales', color='red')
    ax.fill_between(prophet_df['ds'], prophet_df['yhat_lower'], prophet_df['yhat_upper'], color='pink',
                    alpha=0.5)

    # 이상치 식별 (신뢰 구간 밖의 값들)
    outliers = df[(df['total_price'] < prophet_df['yhat_lower'].iloc[:-forecast_days]) | (
                df['total_price'] > prophet_df['yhat_upper'].iloc[:-forecast_days])]
    ax.scatter(outliers['order_date'], outliers['total_price'], color='black', label='Outliers', s=50, zorder=5)

    # 그래프 설정
    ax.set_title('Sales Data with Prophet Prediction and Outliers')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()

    plt.show()
