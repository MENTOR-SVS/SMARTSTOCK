# dashboard.py - Smart Inventory Dashboard (Interactive Version with Plotly)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import os

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Smart Inventory Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- LOAD DATA ----------------------
if not os.path.exists("data/forecast_results.csv"):
    st.error("⚠ Run forecasting first! File 'data/forecast_results.csv' not found.")
    st.stop()

df = pd.read_csv("data/forecast_results.csv")
df.columns = df.columns.str.strip()  # clean any extra spaces

# Validate columns
required_cols = ["Product", "date", "forecast"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# ---------------------- PAGE TITLE ----------------------
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>📦 Smart Inventory Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

# ---------------------- SIDEBAR SETTINGS ----------------------
st.sidebar.markdown("### ⚙️ Configuration Settings")
lead = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
oc = st.sidebar.slider("Ordering Cost", 10, 200, 50)
hc = st.sidebar.slider("Holding Cost", 1, 20, 2)
service_level = st.sidebar.selectbox("Service Level", ["90%", "95%", "99%"], index=1)
z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}[service_level]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 Upload New Sales Data")
upl = st.sidebar.file_uploader("Upload CSV", type="csv")
if upl:
    new = pd.read_csv(upl)
    new.columns = new.columns.str.strip()
    st.sidebar.success("✅ File uploaded successfully!")
    st.sidebar.info("Re-run forecasting.py manually to refresh predictions.")

# ---------------------- TABS ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecasts", "📦 Inventory", "🚨 Stock Alerts", "📄 Reports"])

# ---------------------- TAB 1: FORECAST VISUALIZATION ----------------------
with tab1:
    st.subheader("📈 Forecast Trend Visualization")
    prod = st.selectbox("Select Product", df["Product"].unique())
    sub = df[df["Product"] == prod]

    fig = px.line(
        sub,
        x="date",
        y="forecast",
        title=f"Forecast Trend for {prod}",
        labels={"date": "Date", "forecast": "Forecasted Demand"},
        line_shape="spline",
        template="plotly_white"
    )
    fig.update_traces(line=dict(color="#0073e6", width=2))
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20, color="#2E86C1"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0")
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- TAB 2: INVENTORY PLANNING ----------------------
with tab2:
    st.subheader("📦 Inventory Planning & EOQ Calculation")

    plan = []
    for p in df["Product"].unique():
        d = df[df["Product"] == p]
        avg = d["forecast"].mean() / 30
        dem = d["forecast"].sum()
        std = d["forecast"].std()
        eoq = np.sqrt((2 * dem * oc) / hc)
        ss = z * std * np.sqrt(lead)
        rop = (avg * lead) + ss
        plan.append({
            "Product": p,
            "AvgDailySales": round(avg, 2),
            "TotalDemand": round(dem, 2),
            "EOQ": round(eoq, 2),
            "SafetyStock": round(ss, 2),
            "ReorderPoint": round(rop, 2)
        })

    inv = pd.DataFrame(plan)
    st.dataframe(inv, use_container_width=True)

# ---------------------- TAB 3: STOCK ALERTS ----------------------
with tab3:
    st.subheader("🚨 Real-Time Stock Alerts")

    inv["CurrentStock"] = np.random.randint(10, 100, len(inv))
    inv["Action"] = np.where(inv["CurrentStock"] < inv["ReorderPoint"], "Reorder 🚨", "OK ✅")

    st.dataframe(inv[["Product", "CurrentStock", "ReorderPoint", "Action"]], use_container_width=True)

    fig_bar = px.bar(
        inv,
        x="Product",
        y=["CurrentStock", "ReorderPoint"],
        barmode="group",
        title="Stock vs Reorder Point",
        template="plotly_white",
        labels={"value": "Quantity", "variable": "Metric"}
    )
    fig_bar.update_layout(title_x=0.5, title_font=dict(size=18))
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------- TAB 4: REPORT DOWNLOAD ----------------------
with tab4:
    st.subheader("📄 Reports & Downloads")
    st.download_button(
        label="📥 Download Inventory Report",
        data=inv.to_csv(index=False),
        file_name="daily_reorder_report.csv",
        mime="text/csv"
    )
    st.success("✅ Report ready for download!")

# ---------------------- FOOTER ----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'> Smart Inventory Management</p>",
    unsafe_allow_html=True
)
