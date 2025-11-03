import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a5c 0%, #ffedcc 100%);
            border-top-right-radius: 16px;
            border-bottom-right-radius: 16px;
            box-shadow: 2px 0 8px #cfe2f3;
        }
        section[data-testid="stSidebar"] .sidebar-content {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        .stSlider, .stNumberInput, .stFileUploader {
            background: #fff7e6;
            border-radius: 14px !important;
            margin-bottom: 16px;
        }
        .stSlider .rc-slider-track {
            background: #ea7600 !important; 
        }
        .stSlider .rc-slider-handle {
            border-color: #ea7600 !important;
        }
        .stSelectbox label, .stSlider label, .stNumberInput label, .stFileUploader label {
            font-weight: bold;
            color: #ea7600; 
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")

if not os.path.exists("data/forecast_results.csv"):
    st.error("⚠ Run forecasting first!")
    st.stop()

df = pd.read_csv("data/forecast_results.csv")
df.columns = df.columns.str.strip()

required_cols = ["product_id", "date", "forecast_best"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors='coerce')

st.title("📦 Milestone 4: Streamlit Dashboard & Reporting")

# ---- Sidebar Layout ----
st.sidebar.title("🔧 Settings")
st.sidebar.markdown("---")
lead = st.sidebar.slider("⏳ Lead Time", 1, 30, 7)
oc = st.sidebar.slider("💸 Ordering Cost", 10, 200, 50)
hc = st.sidebar.slider("📦 Holding Cost", 1, 20, 2)
z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}[st.sidebar.selectbox("✅ Service Level", ["90%", "95%", "99%"], index=1)]
st.sidebar.markdown("---")
st.sidebar.markdown("⬆ *Upload New Sales Data*")
upl = st.sidebar.file_uploader("Choose CSV File", type="csv")
if upl:
    new = pd.read_csv(upl)
    new.columns = new.columns.str.strip()
    st.sidebar.success("File uploaded ✅")
    st.sidebar.info("Re-run forecasting.py manually to refresh predictions.")

tab1, tab2, tab3, tab4 = st.tabs(["Forecasts", "Inventory", "Stock Alerts", "Reports"])

import matplotlib.dates as mdates

with tab1:
    prod = st.selectbox("Select Product", df["product_id"].unique())
    sub = df[df["product_id"] == prod]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sub["date"], sub["forecast_best"], label="Forecast", marker='o', color="#FF5733", linewidth=2)
    ax.set_title(prod, fontsize=14, color="#1560bd", fontweight='bold')
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Forecast Value", fontsize=12)
    ax.grid(True, linestyle="--", color="#8bc8fd")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate(rotation=45)  
    plt.tight_layout()
    st.pyplot(fig)

# ---- Inventory Tab ----
with tab2:
    plan = []
    for p in df["product_id"].unique():
        d = df[df["product_id"] == p]
        avg = d["forecast_best"].mean() / 30
        dem = d["forecast_best"].sum()
        std = d["forecast_best"].std()
        eoq = np.sqrt((2 * dem * oc) / hc)
        ss = z * std * np.sqrt(lead)
        rop = (avg * lead) + ss
        plan.append({
            "Product": p,
            "AvgDailySales": avg,
            "TotalDemand": dem,
            "EOQ": eoq,
            "SafetyStock": ss,
            "ReorderPoint": rop
        })
    inv = pd.DataFrame(plan)
    st.dataframe(inv)

# ---- Alerts Tab ----
with tab3:
    inv["CurrentStock"] = np.random.randint(10, 100, len(inv))
    inv["Action"] = np.where(inv["CurrentStock"] < inv["ReorderPoint"], "Reorder 🚨", "OK ✅")
    st.dataframe(inv[["Product", "CurrentStock", "ReorderPoint", "Action"]])
    st.bar_chart(inv.set_index("Product")[["CurrentStock", "ReorderPoint"]])

# ---- Reports Tab ----
with tab4:
    st.download_button("📥 Download Report", inv.to_csv(index=False), "daily_reorder_report.csv")