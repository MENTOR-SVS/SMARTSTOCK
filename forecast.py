import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle, os, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

df = pd.read_csv("cleaned_retail_sales.csv")
df.columns = df.columns.str.strip()

date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
product_col = next((c for c in df.columns if "product" in c.lower()), None)
sales_col = "units_sold"

if date_col is None or product_col is None or sales_col not in df.columns:
    raise KeyError(f"Missing needed column: date: {date_col}, product: {product_col}, sales: {sales_col}")

df[date_col] = pd.to_datetime(df[date_col])
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def train_lstm(series, n_lags=7, epochs=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - n_lags):
        X.append(scaled[i:i+n_lags, 0])
        y.append(scaled[i+n_lags, 0])
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
        data.append(yhat[0][0])
        preds.append(yhat[0][0])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

forecast_list = []
all_products = df[product_col].unique()[:5]

target_product_for_graph = "P001"

for product_name in all_products:
    print(f"🔄 Training Prophet & LSTM for {product_name}...")
    product_df = df[df[product_col] == product_name][[date_col, sales_col]]


    prophet_df = product_df.rename(columns={date_col: 'ds', sales_col: 'y'})
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model_prophet.fit(prophet_df)
    future = model_prophet.make_future_dataframe(periods=30)
    forecast_p = model_prophet.predict(future)
    yhat_prophet = forecast_p['yhat'][-30:]


    sales_series = product_df.set_index(date_col)[sales_col]
    train_size = int(len(sales_series) * 0.8)
    train_series = sales_series.iloc[:train_size]
    lstm_model, scaler = train_lstm(train_series)
    yhat_lstm = forecast_lstm(lstm_model, scaler, sales_series, steps=30)

    actual = sales_series[-30:] if len(sales_series) >= 30 else sales_series
    mae_prophet = mean_absolute_error(actual, yhat_prophet[:len(actual)])
    rmse_prophet = np.sqrt(mean_squared_error(actual, yhat_prophet[:len(actual)]))
    mae_lstm = mean_absolute_error(actual, yhat_lstm[:len(actual)])
    rmse_lstm = np.sqrt(mean_squared_error(actual, yhat_lstm[:len(actual)]))

    print(f"Prophet MAE: {mae_prophet:.2f} RMSE: {rmse_prophet:.2f}")
    print(f"LSTM MAE: {mae_lstm:.2f} RMSE: {rmse_lstm:.2f}")

    if rmse_lstm < rmse_prophet:
        print("✅ LSTM better, saving LSTM predictions.")
        best_forecast = yhat_lstm
        with open(f"models/lstm_model_{product_name}.pkl", "wb") as f:
            pickle.dump((lstm_model.to_json(), scaler.get_params()), f)
        best_model_name = "LSTM"
    else:
        print("✅ Prophet better, saving Prophet predictions.")
        best_forecast = yhat_prophet
        with open(f"models/prophet_model_{product_name}.pkl", "wb") as f:
            pickle.dump(model_prophet, f)
        best_model_name = "Prophet"

    forecast_dates = pd.date_range(start=product_df[date_col].max() + pd.Timedelta(days=1), periods=30)
    temp = pd.DataFrame({
        'date': forecast_dates,
        'forecast_best': best_forecast,
        product_col: product_name
    })
    forecast_list.append(temp)


    if product_name == target_product_for_graph:
        plt.figure(figsize=(10, 5))
        plt.plot(actual.index, actual.values, label='Actual', marker='o')
        plt.plot(forecast_dates, best_forecast, label=f'Predicted by {best_model_name}', marker='x')
        plt.title(f"Actual vs Predicted Sales for Product {product_name}")
        plt.xlabel('Date')
        plt.ylabel(sales_col)
        plt.legend()
        plt.grid(True)
        plt.show()

forecast_all = pd.concat(forecast_list)
forecast_all.to_csv("data/forecast_results.csv", index=False)

print("\n✅ Forecast results saved with Prophet & LSTM comparison in data/forecast_results.csv")
