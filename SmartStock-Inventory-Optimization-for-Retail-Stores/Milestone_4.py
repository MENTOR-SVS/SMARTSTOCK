import pandas as pd, numpy as np, streamlit as st, matplotlib.pyplot as plt, matplotlib.dates as mdates, os

st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")

if not os.path.exists("data/forecast_results.csv"):
    st.error("⚠️ Run forecasting first!"); st.stop()
df = pd.read_csv("data/forecast_results.csv")
df["date"] = pd.to_datetime(df["date"])  # Ensure proper datetime format

st.title("📦 Milestone 4: Streamlit Dashboard & Reporting")

lead = st.sidebar.slider("Lead Time", 1, 30, 7)
oc = st.sidebar.slider("Ordering Cost", 10, 200, 50)
hc = st.sidebar.slider("Holding Cost", 1, 20, 2)
z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}[st.sidebar.selectbox("Service Level", ["90%", "95%", "99%"], 1)]

tab1, tab2, tab3, tab4 = st.tabs(["Forecasts", "Inventory", "Stock Alerts", "Reports"])
with tab1:
    prod = st.selectbox("Select Product", df["Product ID"].unique())
    sub = df[df["Product ID"] == prod]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sub["date"], sub["forecast_best"], label="Forecast", marker="o", linewidth=2)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.set_title(f"Forecast for {prod}", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Forecasted Demand", fontsize=12)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
with tab2:
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
            "Product": p,
            "AvgDailySales": round(avg, 2),
            "TotalDemand": round(dem, 2),
            "EOQ": round(eoq, 2),
            "SafetyStock": round(ss, 2),
            "ReorderPoint": round(rop, 2)
        })
    inv = pd.DataFrame(plan)
    inv["Value"] = inv["TotalDemand"] * hc
    inv = inv.sort_values(by="Value", ascending=False)
    inv["Cumulative%"] = inv["Value"].cumsum() / inv["Value"].sum() * 100
    inv["ABC_Category"] = inv["Cumulative%"].apply(lambda x: "A" if x <= 20 else "B" if x <= 50 else "C")

    st.dataframe(inv)
    selected = inv[inv["Product"] == prod].iloc[0]
    st.metric("EOQ", f"{selected['EOQ']:.2f}")
    st.metric("Reorder Point", f"{selected['ReorderPoint']:.2f}")
    st.metric("Safety Stock", f"{selected['SafetyStock']:.2f}")

    weeks = np.arange(1, 9)
    inv_level = np.linspace(100, 30, 8)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(weeks, inv_level, label="Inventory Level", linewidth=2)
    ax2.axhline(y=selected["ReorderPoint"], color="orange", linestyle="--", label="ROP")
    ax2.axhline(y=selected["SafetyStock"], color="red", linestyle=":", label="Safety Stock")
    ax2.set_title(f"Inventory Level for {prod}", fontsize=14)
    ax2.set_xlabel("Week", fontsize=12)
    ax2.set_ylabel("Units", fontsize=12)
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
with tab3:
    inv["CurrentStock"] = np.random.randint(10, 100, len(inv))
    inv["Action"] = np.where(inv["CurrentStock"] < inv["ReorderPoint"], "Reorder 🚨", "OK ✅")
    st.dataframe(inv[["Product", "CurrentStock", "ReorderPoint", "Action"]])
    st.bar_chart(inv.set_index("Product")[["CurrentStock", "ReorderPoint"]])
with tab4:
    st.download_button("📥 Download Report", inv.to_csv(index=False), "daily_reorder_report.csv")
upl = st.sidebar.file_uploader("Upload New Sales Data", type="csv")
if upl:
    new = pd.read_csv(upl)
    st.sidebar.success("File uploaded ✅")
    st.sidebar.info("Re-run forecasting.py manually to refresh predictions.")