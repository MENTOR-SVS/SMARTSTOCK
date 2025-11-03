

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os


st.set_page_config(page_title="📊 Smart Inventory Dashboard", layout="wide")
st.title("📦 Milestone 4: Smart Inventory Optimization & Reporting")
data_path_csv = "data/forecast_results.csv"
data_path_xls = "retail_store_inventory.xls"

if os.path.exists(data_path_csv):
    df = pd.read_csv(data_path_csv)
    st.success("✅ Loaded forecast_results.csv")
elif os.path.exists(data_path_xls):
    df = pd.read_excel(data_path_xls)
    st.warning("⚠️ Using retail_store_inventory.xls (no forecasts found).")
else:
    st.error("❌ No dataset found! Please upload or place data in project folder.")
    st.stop()


forecast_col = None
for col in df.columns:
    if "forecast" in col.lower():
        forecast_col = col
        break
if forecast_col is None:
    for col in df.columns:
        if "unit" in col.lower() or "sales" in col.lower():
            forecast_col = col
            break

if forecast_col is None:
    st.error("⚠️ Could not detect a forecast or sales column.")
    st.stop()

df["forecast_best"] = df[forecast_col]


if "Product ID" not in df.columns:
    st.error("❌ Missing 'Product ID' column in dataset.")
    st.stop()

if "date" not in df.columns:
    df["date"] = pd.date_range(start="2025-01-01", periods=len(df))


st.sidebar.header("⚙️ Inventory Parameters")
lead = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
oc = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
hc = st.sidebar.slider("Holding Cost ($/unit)", 1, 20, 2)
service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = service_levels[st.sidebar.selectbox("Service Level", list(service_levels.keys()), index=1)]


tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecasts", "📦 Inventory Plan", "⚠️ Stock Alerts", "📊 Reports"])

with tab1:
    st.subheader("📉 Forecasts / Sales Overview")
    prod = st.selectbox("Select Product", df["Product ID"].unique())
    sub = df[df["Product ID"] == prod]

    plt.figure(figsize=(8, 4))
    plt.plot(sub["date"], sub["forecast_best"], label="Forecast / Sales", color="steelblue")
    plt.title(f"Forecast for Product {prod}")
    plt.xlabel("Date")
    plt.ylabel("Forecasted Demand")
    plt.legend()
    st.pyplot(plt.gcf())


with tab2:
    st.subheader("📊 Inventory Optimization Plan")
    plan = []
    for p in df["Product ID"].unique():
        d = df[df["Product ID"] == p]
        avg = d["forecast_best"].mean() / 30
        dem = d["forecast_best"].sum()
        std = d["forecast_best"].std()

        eoq = np.sqrt((2 * dem * oc) / hc)
        ss = z * std * np.sqrt(lead)
        rop = (avg * lead) + ss

        plan.append({
            "Product ID": p,
            "AvgDailySales": round(avg, 2),
            "TotalDemand": round(dem, 2),
            "EOQ": round(eoq, 2),
            "SafetyStock": round(ss, 2),
            "ReorderPoint": round(rop, 2)
        })

    inv = pd.DataFrame(plan)
    st.dataframe(inv.style.background_gradient(cmap="YlGnBu"))

    
    os.makedirs("data", exist_ok=True)
    inv.to_csv("data/inventory_plan.csv", index=False)
    st.success("✅ Inventory plan saved to data/inventory_plan.csv")


with tab3:
    st.subheader("⚠️ Low Stock & Reorder Alerts")
    inv["CurrentStock"] = np.random.randint(10, 120, len(inv))
    inv["Status"] = np.where(inv["CurrentStock"] < inv["ReorderPoint"], "🚨 Reorder Needed", "✅ OK")

    st.dataframe(inv[["Product ID", "CurrentStock", "ReorderPoint", "Status"]])

    
    alert_path = "data/stock_alerts.csv"
    inv[["Product ID", "CurrentStock", "ReorderPoint", "Status"]].to_csv(alert_path, index=False)
    st.success(f"🚨 Stock alerts saved to {alert_path}")

    st.bar_chart(inv.set_index("Product ID")[["CurrentStock", "ReorderPoint"]])


with tab4:
    st.subheader("📄 Export & Reporting")
    st.download_button("📥 Download Inventory Report (CSV)", inv.to_csv(index=False), "inventory_report.csv")
    st.success("✅ Inventory plan successfully generated!")

    
    upl = st.file_uploader("📤 Upload New Sales Data (CSV)", type="csv")
    if upl:
        new = pd.read_csv(upl)
        new.to_csv("data/new_sales_data.csv", index=False)
        st.write("✅ New file uploaded and saved as data/new_sales_data.csv")
        st.info("ℹ️ Please rerun forecasting.py to update predictions.")
