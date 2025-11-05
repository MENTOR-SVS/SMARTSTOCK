# milestone2_forecasting_seasonality_aware_fast.py
# --------------------------------------------------------
# Forecasting comparison: Prophet vs LSTM for all products
# Seasonality-aware model selection + unified forecast output
# --------------------------------------------------------

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------
# Helper Function: SMAPE
# --------------------------------------------------------
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + 1e-10))

# --------------------------------------------------------
# Load data
# --------------------------------------------------------
file_path = "data/sales_data_cleaned.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

df = pd.read_csv(file_path)
date_col = next((col for col in df.columns if "date" in col.lower() or "time" in col.lower()), None)
if not date_col:
    raise ValueError("❌ No date column found in dataset.")
df[date_col] = pd.to_datetime(df[date_col])

os.makedirs("data/forecasts", exist_ok=True)

summary = []
combined_forecasts = []  # 🟢 New: to collect all final forecasts

# --------------------------------------------------------
# Loop through all products
# --------------------------------------------------------
for product_name in df['Product ID'].unique():
    print(f"\n================ Processing Product: {product_name} ================\n")

    product_df = df[df['Product ID'] == product_name][[date_col, 'Units Sold', 'Promotion']].copy()
    product_df = product_df.sort_values(by=date_col)

    if len(product_df) < 20:
        print(f"⚠️ Skipping {product_name}: Not enough data points ({len(product_df)})")
        continue

    prophet_df = product_df.rename(columns={date_col: 'ds', 'Units Sold': 'y', 'Promotion': 'promotion'})
    sales_series = prophet_df.set_index('ds')['y']

    # --------------------------------------------------------
    # Seasonality detection
    # --------------------------------------------------------
    if len(sales_series) >= 28:
        weekly_pattern = sales_series.groupby(sales_series.index.weekday).mean().std()
        monthly_pattern = sales_series.groupby(sales_series.index.month).mean().std()
        seasonality_strength = weekly_pattern + monthly_pattern
    else:
        seasonality_strength = 0

    seasonality_threshold = 5

    # --------------------------------------------------------
    # Prophet
    # --------------------------------------------------------
    try:
        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model_prophet.add_regressor('promotion')
        model_prophet.fit(prophet_df)

        future = model_prophet.make_future_dataframe(periods=30)
        future['promotion'] = 0
        forecast_prophet = model_prophet.predict(future)

        actual = sales_series.tail(30)
        predicted_prophet = forecast_prophet.set_index('ds')['yhat'].tail(len(actual))
        mae_prophet = mean_absolute_error(actual, predicted_prophet)
        rmse_prophet = np.sqrt(mean_squared_error(actual, predicted_prophet))
        acc_prophet = 100 - smape(actual.values, predicted_prophet.values)

    except Exception as e:
        print(f"❌ Prophet failed for {product_name}: {e}")
        mae_prophet = rmse_prophet = acc_prophet = np.nan

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------
    try:
        scaler = MinMaxScaler()
        scaled_sales = scaler.fit_transform(sales_series.values.reshape(-1, 1))

        look_back = 10
        X, y_data = [], []
        for i in range(len(scaled_sales) - look_back):
            X.append(scaled_sales[i:i+look_back])
            y_data.append(scaled_sales[i+look_back])
        X, y_data = np.array(X), np.array(y_data)

        if len(X) < 20:
            print(f"⚠️ Skipping LSTM for {product_name}: Not enough samples ({len(X)})")
            mae_lstm = rmse_lstm = acc_lstm = np.nan
        else:
            split = int(len(X)*0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y_data[:split], y_data[split:]

            model_lstm = Sequential([
                LSTM(50, activation='relu', input_shape=(look_back, 1)),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

            y_pred = model_lstm.predict(X_test)
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            mae_lstm = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
            rmse_lstm = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
            acc_lstm = 100 - smape(y_test_rescaled.flatten(), y_pred_rescaled.flatten())

    except Exception as e:
        print(f"❌ LSTM failed for {product_name}: {e}")
        mae_lstm = rmse_lstm = acc_lstm = np.nan

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------
    if np.isnan(mae_prophet) and not np.isnan(mae_lstm):
        best_model = "LSTM"
    elif np.isnan(mae_lstm) and not np.isnan(mae_prophet):
        best_model = "Prophet"
    elif seasonality_strength >= seasonality_threshold:
        best_model = "Prophet"
    else:
        best_model = "Prophet" if mae_prophet <= mae_lstm else "LSTM"

    summary.append({
        "Product": product_name,
        "Best_Model": best_model,
        "Seasonality_Strength": round(seasonality_strength, 2),
        "Prophet_MAE": round(mae_prophet, 2) if not np.isnan(mae_prophet) else None,
        "LSTM_MAE": round(mae_lstm, 2) if not np.isnan(mae_lstm) else None
    })

    # --------------------------------------------------------
    # Save product forecast + add to combined list
    # --------------------------------------------------------
    final_forecast = forecast_prophet[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'forecast'})
    final_forecast['Product'] = product_name
    final_forecast['Model'] = best_model

    final_forecast.to_csv(f"data/forecasts/forecast_{product_name}.csv", index=False)
    combined_forecasts.append(final_forecast)

    print(f"📊 Best model for {product_name}: {best_model}")

# --------------------------------------------------------
# Save combined forecast + summary
# --------------------------------------------------------
combined_df = pd.concat(combined_forecasts, ignore_index=True)
combined_df.to_csv("data/forecast_results.csv", index=False)
summary_df = pd.DataFrame(summary)
summary_df.to_csv("data/forecasts/summary_forecasts.csv", index=False)

print("\n✅ Combined forecast saved to: data/forecast_results.csv")
print("✅ Summary saved to: data/forecasts/summary_forecasts.csv")
