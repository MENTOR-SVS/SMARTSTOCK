# M2.py - Forecasting with Prophet, LSTM, ARIMA (product + category)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
import warnings, os

warnings.filterwarnings("ignore")

# -------------------------
# 1. Load cleaned sales data
# -------------------------
df = pd.read_csv("cleaned_retail_sales.csv")
df['date'] = pd.to_datetime(df['date'])

os.makedirs("data", exist_ok=True)

# -------------------------
# 2. Helper Functions (LSTM)
# -------------------------
def train_lstm(series, n_lags=7, epochs=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - n_lags):
        X.append(scaled[i:i+n_lags, 0]); y.append(scaled[i+n_lags, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, scaler

def forecast_lstm(model, scaler, series, steps=30, n_lags=7):
    data = scaler.transform(series.values.reshape(-1, 1)).flatten().tolist()
    preds = []
    for _ in range(steps):
        x_input = np.array(data[-n_lags:]).reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)
        data.append(yhat[0][0]); preds.append(yhat[0][0])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

# -------------------------
# 3. Forecasting per product
# -------------------------
forecast_list = []

for product in df['product_id'].unique():
    product_df = df[df['product_id'] == product][['date','units_sold','category']]

    category = product_df['category'].iloc[0]
    sales_series = product_df.set_index('date')['units_sold']

    # Prophet
    prophet_df = product_df.rename(columns={'date':'ds','units_sold':'y'})
    model_p = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model_p.fit(prophet_df)
    future = model_p.make_future_dataframe(periods=30)
    forecast_p = model_p.predict(future)
    yhat_p = forecast_p['yhat'][-30:]

    # LSTM
    train_series = sales_series.iloc[:int(len(sales_series)*0.8)]
    lstm_model, scaler = train_lstm(train_series)
    yhat_l = forecast_lstm(lstm_model, scaler, sales_series, steps=30)

    # ARIMA
    try:
        model_a = ARIMA(sales_series, order=(5,1,0))
        model_fit = model_a.fit()
        yhat_a = model_fit.forecast(steps=30)
    except:
        yhat_a = pd.Series([sales_series.mean()] * 30)

    # Evaluation (use last 30 days if available)
    actual = sales_series[-30:] if len(sales_series) >= 30 else sales_series
    def mape(y_true,y_pred): return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mae_p, rmse_p = mean_absolute_error(actual,yhat_p[:len(actual)]), np.sqrt(mean_squared_error(actual,yhat_p[:len(actual)]))
    mae_l, rmse_l = mean_absolute_error(actual,yhat_l[:len(actual)]), np.sqrt(mean_squared_error(actual,yhat_l[:len(actual)]))
    mae_a, rmse_a = mean_absolute_error(actual,yhat_a[:len(actual)]), np.sqrt(mean_squared_error(actual,yhat_a[:len(actual)]))

    print(f"{product} ({category}) → Prophet RMSE={rmse_p:.2f}, LSTM RMSE={rmse_l:.2f}, ARIMA RMSE={rmse_a:.2f}")

    # Pick best
    best_forecast = yhat_p
    best_rmse = rmse_p
    if rmse_l < best_rmse:
        best_forecast = yhat_l
        best_rmse = rmse_l
    if rmse_a < best_rmse:
        best_forecast = yhat_a
        best_rmse = rmse_a

    forecast_dates = pd.date_range(start=product_df['date'].max()+pd.Timedelta(days=1), periods=30)
    temp = pd.DataFrame({"date":forecast_dates,
                         "forecast_best":best_forecast,
                         "product_id":product,
                         "category":category})
    forecast_list.append(temp)

forecast_all = pd.concat(forecast_list)
forecast_all.to_csv("data/forecast_results.csv", index=False)

print("\n✅ Forecast results saved in data/forecast_results.csv")
