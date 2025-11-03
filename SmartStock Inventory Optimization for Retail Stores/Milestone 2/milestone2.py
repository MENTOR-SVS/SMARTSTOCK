import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# 1. Load Cleaned Data
df = pd.read_csv("processed_retail_store_inventory.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Create directories to save models, plots, and data
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 2. Helper Functions for LSTM
def train_lstm(series, n_lags=7, epochs=10):
    """Prepares data and trains a simple LSTM model."""
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

def forecast_lstm(model, scaler, history, steps, n_lags=7):
    """Forecasts future values using a trained LSTM model."""
    predictions = []
    last_data = history[-n_lags:]
    input_data = scaler.transform(last_data.values.reshape(-1, 1))

    for _ in range(steps):
        x_input = input_data.reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0][0])
        input_data = np.append(input_data[1:], yhat)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# 3. Forecasting Loop
forecast_list = []
evaluation_results = []

products_to_forecast = df['Product_ID'].unique()

for product in products_to_forecast:
    print(f"--- Forecasting for Product: {product} ---")
    product_df = df[df['Product_ID'] == product][['Date', 'Units_Sold']].sort_values('Date')
    
    
    # Group by date and sum units sold to handle duplicates, then set index
    product_df = product_df.groupby('Date')['Units_Sold'].sum().reset_index().set_index('Date')
    
    # Use daily data by resampling and filling missing values
    product_df = product_df.asfreq('D').fillna(method='ffill')

    # Split data into 80% train and 20% test
    train_size = int(len(product_df) * 0.8)
    train, test = product_df[0:train_size], product_df[train_size:]

    # Prophet Model
    print("Training Prophet model...")
    prophet_train_df = train.reset_index().rename(columns={'Date': 'ds', 'Units_Sold': 'y'})
    model_p = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model_p.fit(prophet_train_df)
    future = model_p.make_future_dataframe(periods=len(test))
    forecast_p = model_p.predict(future)
    yhat_p = forecast_p['yhat'][-len(test):]

    # ARIMA Model
    print("Training ARIMA model...")
    model_a = ARIMA(train['Units_Sold'], order=(5,1,0))
    model_a_fit = model_a.fit()
    yhat_a = model_a_fit.forecast(steps=len(test))

    # LSTM Model
    print("Training LSTM model...")
    n_lags = 7
    lstm_model, scaler = train_lstm(train['Units_Sold'], n_lags=n_lags, epochs=50)
    yhat_l = forecast_lstm(lstm_model, scaler, train['Units_Sold'], steps=len(test), n_lags=n_lags)

    # Evaluation
    mae_p = mean_absolute_error(test['Units_Sold'], yhat_p)
    rmse_p = np.sqrt(mean_squared_error(test['Units_Sold'], yhat_p))
    
    mae_a = mean_absolute_error(test['Units_Sold'], yhat_a)
    rmse_a = np.sqrt(mean_squared_error(test['Units_Sold'], yhat_a))
    
    mae_l = mean_absolute_error(test['Units_Sold'], yhat_l)
    rmse_l = np.sqrt(mean_squared_error(test['Units_Sold'], yhat_l))
    
    evaluation_results.extend([
        {'Product_ID': product, 'Model': 'Prophet', 'MAE': mae_p, 'RMSE': rmse_p},
        {'Product_ID': product, 'Model': 'ARIMA', 'MAE': mae_a, 'RMSE': rmse_a},
        {'Product_ID': product, 'Model': 'LSTM', 'MAE': mae_l, 'RMSE': rmse_l}
    ])

    print(f"Prophet -> MAE: {mae_p:.2f}, RMSE: {rmse_p:.2f}")
    print(f"ARIMA   -> MAE: {mae_a:.2f}, RMSE: {rmse_a:.2f}")
    print(f"LSTM    -> MAE: {mae_l:.2f}, RMSE: {rmse_l:.2f}")

    # Compare and Save Best Model
    models = {'Prophet': (model_p, yhat_p, rmse_p), 'ARIMA': (model_a_fit, yhat_a, rmse_a), 'LSTM': (lstm_model, yhat_l, rmse_l)}
    best_model_name = min(models, key=lambda k: models[k][2])
    best_model_obj, best_forecast, _ = models[best_model_name]
    
    print(f"Best model for {product}: {best_model_name}")

    # Save the best model
    model_filename = f"models/best_model_{product}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model_obj, f)
    
    # Store forecast results
    temp_forecast = pd.DataFrame({
        'Date': test.index,
        'Product_ID': product,
        'Actual_Sales': test['Units_Sold'].values,
        'Forecasted_Sales': best_forecast,
        'Model': best_model_name
    })
    forecast_list.append(temp_forecast)
    
    # Generate and Save Plots
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Units_Sold'], label='Train')
    plt.plot(test.index, test['Units_Sold'], label='Actual')
    plt.plot(test.index, best_forecast, label='Forecast')
    plt.title(f'Sales Forecast for {product} (Best Model: {best_model_name})')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/forecast_vs_actual_{product}.png")
    plt.close()

    if best_model_name == 'Prophet':
        fig = model_p.plot_components(forecast_p)
        plt.savefig(f"plots/prophet_components_{product}.png")
        plt.close(fig)

# 4. Finalize and Save Results
final_forecast_df = pd.concat(forecast_list)
final_forecast_df.to_csv("data/forecast_results.csv", index=False)

evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df.to_csv("data/evaluation_metrics.csv", index=False)

print("\n✅ Forecasting Milestone Completed!")
print("- Best models saved in the 'models' directory.")
print("- Forecast plots saved in the 'plots' directory.")
print("- Forecast results saved to 'data/forecast_results.csv'.")
print("- Evaluation metrics saved to 'data/evaluation_metrics.csv'.")

print("\n--- Model Evaluation Metrics ---")
print(evaluation_df.to_string())