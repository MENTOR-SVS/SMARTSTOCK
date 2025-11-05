# Milestone 2: Forecasting with Prophet, ARIMA & LSTM 

import os, warnings, logging
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['LOGLEVEL'] = 'ERROR'

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("pystan").setLevel(logging.ERROR)
logging.getLogger("fbprophet").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = "./output/cleaned_retail_inventory_dataset.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("⚠️ Run Milestone 1 first! Cleaned dataset not found.")

os.makedirs("data", exist_ok=True)
os.makedirs("forecast_plots", exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Aggregate monthly sales per product to ensure regular frequency
df = df.groupby(['item_name', pd.Grouper(key='transaction_date', freq='M')])['sales'].sum().reset_index()


# Metrics
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

def wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    denom = np.sum(y_true)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / denom

# LSTM 
def create_lstm_dataset(X, y, n_lags=3):
    Xs, ys = [], []
    for i in range(len(X) - n_lags):
        Xs.append(X[i:i+n_lags])
        ys.append(y[i+n_lags])
    return np.array(Xs), np.array(ys)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(series, target_col=0, n_lags=3, epochs=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    X, y = create_lstm_dataset(scaled, scaled[:, target_col], n_lags=n_lags)
    if len(X) < 10:  # not enough samples for LSTM
        return None, None
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    model = build_lstm((X.shape[1], X.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=8, verbose=0, callbacks=[es])
    return model, scaler

def forecast_lstm(model, scaler, series, target_col=0, steps=3, n_lags=3):
    data_scaled = scaler.transform(series)
    seq = data_scaled[-n_lags:]
    preds = []
    for _ in range(steps):
        x_input = seq.reshape((1, n_lags, data_scaled.shape[1]))
        yhat = model.predict(x_input, verbose=0)
        next_row = seq[-1].copy()
        next_row[target_col] = yhat[0][0]
        seq = np.vstack([seq[1:], next_row])
        preds.append(yhat[0][0])
    dummy = np.zeros((steps, data_scaled.shape[1]))
    dummy[:, target_col] = preds
    return scaler.inverse_transform(dummy)[:, target_col]

# Forecasting loop
forecast_list = []
eval_summary = []

all_products = df["item_name"].unique()

for product in all_products:
    print("\n" + "="*70)
    print(f" Training Prophet, ARIMA & LSTM for {product}...")
    print("="*70)

    product_df = df[df['item_name'] == product].copy().sort_values("transaction_date")
    if len(product_df) < 12:
        print(f" Skipping {product}: need >=12 monthly observations (have {len(product_df)})")
        continue

   
    train = product_df.iloc[:-3].reset_index(drop=True)
    test = product_df.iloc[-3:].reset_index(drop=True)
    actual = test['sales'].values


    try:
        prophet_train = train.rename(columns={'transaction_date':'ds','sales':'y'})
        model_p = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
     
        model_p.fit(prophet_train[['ds','y']])
        future = model_p.make_future_dataframe(periods=3, freq='M')
        forecast_p = model_p.predict(future)
        yhat_p = forecast_p['yhat'][-3:].values
    except Exception as e:
        yhat_p = np.repeat(train['sales'].mean(), 3)

    
    try:
        s = train.set_index('transaction_date')['sales'].asfreq('M')
        s = s.fillna(method='ffill')
        model_a = sm.tsa.ARIMA(s, order=(1,1,1))
        res_a = model_a.fit()
        yhat_a = res_a.forecast(steps=3)
      
        yhat_a = np.array(yhat_a).reshape(-1)[:3]
    except Exception as e:
        yhat_a = np.repeat(train['sales'].mean(), 3)


    series = product_df[['sales']].values
    lstm_model, scaler = train_lstm(series, target_col=0, n_lags=3, epochs=30)
    if lstm_model is not None:
        try:
            yhat_lstm = forecast_lstm(lstm_model, scaler, series, target_col=0, steps=3, n_lags=3)
        except Exception:
            yhat_lstm = np.repeat(train['sales'].mean(), 3)
    else:
        yhat_lstm = np.repeat(train['sales'].mean(), 3)

    models = {
        "Prophet": np.array(yhat_p, dtype=float),
        "ARIMA": np.array(yhat_a, dtype=float),
        "LSTM": np.array(yhat_lstm, dtype=float)
    }


    metrics = {}
    for mname, preds in models.items():
       
        if len(preds) < len(actual):
            preds = np.pad(preds, (0, len(actual)-len(preds)), 'edge')
        elif len(preds) > len(actual):
            preds = preds[:len(actual)]

        mae = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mape_val = mape(actual, preds)
        wmape_val = wmape(actual, preds)
        metrics[mname] = (mae, rmse, mape_val, wmape_val)

        eval_summary.append([product, mname, mae, rmse, mape_val, wmape_val])

    # determine best by WMAPE (lower is better)
    best_model = min(metrics.keys(), key=lambda k: metrics[k][3] if not np.isnan(metrics[k][3]) else np.inf)
    best_preds = models[best_model]
    # ensure same length
    n = min(len(test), len(best_preds))
    best_preds = best_preds[:n]

    print(f"\n Evaluation for {product} (last {n} months test):")
    for mname in ["Prophet","ARIMA","LSTM"]:
        mae, rmse, mape_val, wmape_val = metrics[mname]
        mape_str = f"{mape_val:.2f}%" if not np.isnan(mape_val) else "nan"
        wmape_str = f"{wmape_val:.2f}%" if not np.isnan(wmape_val) else "nan"
        print(f"{mname:<7} -> MAE: {mae:10.2f}   RMSE: {rmse:10.2f}   MAPE: {mape_str:8}   WMAPE: {wmape_str:8}")
    print(f"\n Best Model for {product}: {best_model}")

    n = min(len(test), len(best_preds))
    df_fore = pd.DataFrame({
        "date": pd.to_datetime(test['transaction_date'].values)[:n],
        "forecast": best_preds[:n],
        "actual": test['sales'].values[:n],
        "product": product,
        "best_model": best_model
    })
    forecast_list.append(df_fore)


    try:
        plt.figure(figsize=(10,5))
        plt.plot(train['transaction_date'], train['sales'], label="Train")
        plt.plot(test['transaction_date'], test['sales'], label="Actual", color="black")
        plt.plot(test['transaction_date'], models["Prophet"][:len(test)], "--", label="Prophet")
        plt.plot(test['transaction_date'], models["ARIMA"][:len(test)], "--", label="ARIMA")
        plt.plot(test['transaction_date'], models["LSTM"][:len(test)], "--", label="LSTM")
        plt.plot(test['transaction_date'][:n], best_preds, "o-", label=f"Best: {best_model}", linewidth=3)
        plt.title(f"Monthly Forecast for {product}")
        plt.xlabel("Date")
        plt.ylabel("Sales (Units)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        safe_name = "".join(c if c.isalnum() else "_" for c in product)[:200]
        plt.savefig(f"forecast_plots/forecast_{safe_name}.png", dpi=300)
        plt.close()
    except Exception:
        pass  

#  Save Results
if forecast_list:
    forecast_all = pd.concat(forecast_list, ignore_index=True)
    forecast_all.to_csv("data/forecast_results.csv", index=False)

    summary_df = pd.DataFrame(eval_summary, columns=["Product","Model","MAE","RMSE","MAPE","WMAPE"])

   
    product_sales = df.groupby("item_name")['sales'].sum().reset_index().sort_values("sales", ascending=False)
    cutoff = max(1, int(len(product_sales) * 0.2))
    top_products = product_sales.head(cutoff)['item_name'].tolist()

    filtered_summary = summary_df[summary_df['Product'].isin(top_products)].reset_index(drop=True)
    filtered_summary.to_csv("data/forecast_evaluation_summary.csv", index=False)

    print("\n Forecasts saved to data/forecast_results.csv")
    print(" Evaluation summary (Top products) saved to data/forecast_evaluation_summary.csv")
    print("\nSample (Top products):")
    print(filtered_summary.head())
else:
    print("\n No forecasts generated.")
