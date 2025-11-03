# M4.py - Smart Inventory Dashboard 

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

#  STREAMLIT CONFIG
st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")

# THEME & STYLES
st.markdown(
    """
<style>
html, body, [class*="css"] {
  background-color: #0A0E23;
  color: #EAF4FF;
  font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
  color: #4FC3F7;
  font-weight: 700;
}
hr {
  border: 0.5px solid rgba(79,195,247,0.22);
  margin: 12px 0;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F1B3A, #0A1224);
  color: #EAF4FF;
  border-right: 1px solid #18243f;
}
section.main > div {
  animation: fadein 0.75s;
}
@keyframes fadein {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
.metric-card {
  border-radius: 10px;
  padding: 12px;
  text-align: center;
  color: #EAF4FF;
  box-shadow: 0 6px 18px rgba(0,0,0,0.45);
}
.metric-label { font-size: 12px; color: #94a3b8; margin-bottom:6px; }
.metric-value { font-size:22px; font-weight:700; }
.small-muted { color:#94a3b8; font-size:12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📦 Smart Inventory Dashboard")

#FILE CHECK 
required_files = ["forecast_results.csv", "inventory_plan.csv"]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"⚠ Missing files: {', '.join(missing_files)}. Place them in the app folder.")
    st.stop()

#LOAD DATA
forecast_df = pd.read_csv("forecast_results.csv")
inv_plan_df = pd.read_csv("inventory_plan.csv")
sales_df = pd.read_csv("cleaned_retail_sales.csv") if os.path.exists("cleaned_retail_sales.csv") else None

#  DETECT PRODUCT COLUMN
def detect_product_col(df):
    if df is None:
        return None
    for c in ["Product", "product", "Product ID", "product_id", "Product_ID", "product_code", "SKU"]:
        if c in df.columns:
            return c
    return None

f_col = detect_product_col(forecast_df)
i_col = detect_product_col(inv_plan_df)
s_col = detect_product_col(sales_df) if sales_df is not None else None

if f_col is None or i_col is None:
    st.error("⚠ Could not find product column in one of the CSVs. Expected column named like 'Product' or 'Product ID'.")
    st.write("Forecast columns:", list(forecast_df.columns))
    st.write("Inventory plan columns:", list(inv_plan_df.columns))
    st.stop()

for dcol in ["date", "Date", "DATE"]:
    if dcol in forecast_df.columns:
        forecast_df["date"] = pd.to_datetime(forecast_df[dcol], errors="coerce")
        break
if "date" not in forecast_df.columns:
    forecast_df["date"] = pd.NaT

if sales_df is not None:
    for dcol in ["date", "Date", "DATE"]:
        if dcol in sales_df.columns:
            sales_df["date"] = pd.to_datetime(sales_df[dcol], errors="coerce")
            break

if "Category" in inv_plan_df.columns:
    forecast_df = forecast_df.merge(
        inv_plan_df[[i_col, "Category"]].drop_duplicates(),
        left_on=f_col, right_on=i_col, how="left"
    )
else:
    forecast_df["Category"] = np.nan
forecast_df["Category"] = forecast_df["Category"].fillna("Unknown")
forecast_df["DisplayName"] = forecast_df.apply(
    lambda r: f"{r[f_col]} — {r['Category']}" if pd.notna(r["Category"]) else str(r[f_col]),
    axis=1
)

# SIDEBAR CONTROLs
st.sidebar.header("⚙ Dashboard Controls")
accent_color = "#4FC3F7"  

lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.number_input("Ordering Cost ($)", 10, 500, 50)
holding_cost = st.sidebar.number_input("Holding Cost ($/unit)", 1, 100, 2)
service_level = st.sidebar.selectbox("Service Level", ["90%", "95%", "99%"], index=1)
z_map = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z_value = z_map[service_level]

#  CATEGORY FILTER & PRODUCT SELECTION 
cat_list = sorted(forecast_df["Category"].dropna().unique())
selected_cat = st.selectbox("Filter by Category", ["All"] + cat_list)
if selected_cat != "All":
    product_choices = sorted(forecast_df[forecast_df["Category"] == selected_cat]["DisplayName"].unique())
else:
    product_choices = sorted(forecast_df["DisplayName"].unique())
selected_display = st.selectbox("Select Product", product_choices)
selected_product = selected_display.split(" — ")[0]
prod_df = forecast_df[forecast_df[f_col] == selected_product].sort_values("date")

#DYNAMIC INVENTORY PLAN
plan = []
for p in forecast_df[f_col].unique():
    d = forecast_df[forecast_df[f_col] == p]
    avg_daily = d["forecast_best"].mean() / 30 if "forecast_best" in d.columns else 0
    total_demand = d["forecast_best"].sum() if "forecast_best" in d.columns else 0
    std_dev = d["forecast_best"].std() if "forecast_best" in d.columns else 0
    eoq = np.sqrt((2 * total_demand * ordering_cost) / max(1e-6, holding_cost))
    ss = z_value * std_dev * np.sqrt(lead_time)
    rop = (avg_daily * lead_time) + ss
    cat_val = d["Category"].iloc[0] if "Category" in d.columns else "Unknown"
    plan.append({
        "Product": p,
        "Category": cat_val,
        "AvgDailySales": round(avg_daily,2),
        "TotalDemand": int(total_demand),
        "EOQ": round(eoq,2),
        "SafetyStock": round(ss,2),
        "ReorderPoint": round(rop,2)
    })

inv_df = pd.DataFrame(plan)
np.random.seed(42)
inv_df["CurrentStock"] = np.random.randint(50,2000,len(inv_df))
inv_df["Action"] = np.where(inv_df["CurrentStock"] < inv_df["ReorderPoint"], "Reorder ⚠", "OK ✅")

def render_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card" style="background:linear-gradient(145deg,#0f234a, #07102a);">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent_color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# TABS 
tab1, tab2, tab3, tab4 = st.tabs(["Forecast Visualization", "Inventory Planning", "Stock Alerts", "Reports"])

#TAB 1: Forecast 
with tab1:
    st.subheader("📈 Demand Forecast Visualization")
    if not prod_df.empty and "forecast_best" in prod_df.columns:
        avg_forecast = prod_df["forecast_best"].mean()
        peak = prod_df["forecast_best"].max()
        trough = prod_df["forecast_best"].min()
        col1, col2, col3 = st.columns(3)
        with col1: render_metric("Average Forecast", f"{avg_forecast:.2f}")
        with col2: render_metric("Peak Forecast", f"{peak:.2f}")
        with col3: render_metric("Lowest Forecast", f"{trough:.2f}")

        fig1 = px.line(prod_df, x="date", y="forecast_best", markers=True, color_discrete_sequence=[accent_color])
        fig1.update_layout(template="plotly_dark", xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Forecast data missing for the selected product.")

#  TAB 2: Inventory Planning
with tab2:
    st.subheader("📦 Inventory Planning")
    st.dataframe(inv_df[["Product","Category","AvgDailySales","TotalDemand","EOQ","SafetyStock","ReorderPoint","CurrentStock","Action"]], use_container_width=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a: render_metric("Average EOQ", f"{inv_df['EOQ'].mean():.2f}")
    with col_b: render_metric("Average Safety Stock", f"{inv_df['SafetyStock'].mean():.2f}")
    with col_c: render_metric("Average ROP", f"{inv_df['ReorderPoint'].mean():.2f}")
    fig3 = px.bar(inv_df.sort_values("EOQ",ascending=False), x="Product", y=["EOQ","ReorderPoint"], barmode="group", color_discrete_sequence=[accent_color,"#64748b"])
    fig3.update_layout(template="plotly_dark", xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

#  TAB 3: Stock Alerts 
with tab3:
    st.subheader("🚨 Stock Alerts and Levels")
    st.dataframe(inv_df[["Product","Category","CurrentStock","ReorderPoint","Action"]].sort_values("Action",ascending=False), use_container_width=True)
    alert_fig = px.bar(inv_df, x="Product", y=["CurrentStock","ReorderPoint"], barmode="group", color_discrete_sequence=[accent_color,"#f97316"], title="Stock vs Reorder Point")
    alert_fig.update_layout(template="plotly_dark", xaxis_tickangle=--45)
    st.plotly_chart(alert_fig, use_container_width=True)
    low_count = inv_df[inv_df["Action"].str.contains("Reorder")].shape[0]
    if low_count>0: st.markdown(f"*AI Insight:* {low_count} products below reorder level 🟠")
    else: st.markdown("*AI Insight:* All products above safe stock ✅")

#TAB 4: Reports
with tab4:
    st.subheader("📥 Reports and Downloads")
    st.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c1,c2 = st.columns(2)
    c1.download_button("⬇ Download Inventory Plan", data=inv_df.to_csv(index=False), file_name="inventory_plan_dynamic.csv", mime="text/csv")
    c2.download_button("⬇ Download Forecast Results", data=forecast_df.to_csv(index=False), file_name="forecast_results_with_category.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Category-wise Forecast Summary")
    if "forecast_best" in forecast_df.columns:
        cat_summary = forecast_df.groupby("Category")["forecast_best"].sum().reset_index().rename(columns={"forecast_best":"TotalForecast"})
        st.dataframe(cat_summary.sort_values("TotalForecast",ascending=False), use_container_width=True)
        fig4 = px.bar(cat_summary, x="Category", y="TotalForecast", color="Category", title="Total Forecast by Category", color_discrete_sequence=px.colors.qualitative.Plotly)
        fig4.update_layout(template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No 'forecast_best' column in forecast dataset.")
st.success("✅ Dashboard loaded")
