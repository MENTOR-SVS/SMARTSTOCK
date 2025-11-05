# inventory.py - Milestone 3: Navy-Blue Product-wise Inventory Dashboard
# Features: navy blue background, white text, animated metric cards, Plotly professional chart, ABC classification, downloadable CSV

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ------------------------
# Page config & styling
# ------------------------
st.set_page_config(page_title="📦 SmartStock - Inventory Optimizer", layout="wide")
st.markdown("""
<style>
body, .stApp, .block-container { background-color: #0B1D51; color: #FFFFFF; }
h1, h2, h3, h4 { color: #FFFFFF !important; }
.small-muted { color:#B8D1E6; font-size:0.9em; }

.metric-card.hover-anim {
  background: linear-gradient(90deg,#0A2B66,#0C3B99);
  border-radius:12px;
  padding:12px;
  transition: transform 0.3s, box-shadow 0.3s;
  text-align:center;
}
.metric-card.hover-anim:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

.dataframe td { color: #FFFFFF; background-color: transparent; }
button { background-color: #0A2B66 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 SmartStock — Inventory Optimization")
st.caption("Forecast-driven Inventory Logic with ABC Analysis")

# ------------------------
# Load & normalize data
# ------------------------
@st.cache_data
def load_data(path="data/forecast_results.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Auto-detect columns
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    forecast_col = next((c for c in df.columns if "forecast" in c.lower() or "yhat" in c.lower()), None)
    product_col = next((c for c in df.columns if "product" in c.lower() or "sku" in c.lower()), None)
    if date_col is None or forecast_col is None:
        return None
    df = df.rename(columns={date_col: "date", forecast_col: "forecast"})
    if product_col:
        df = df.rename(columns={product_col: "product"})
    else:
        df["product"] = "Single Product"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df = df.dropna(subset=["date", "forecast"]).sort_values("date")
    return df

df = load_data()
if df is None or df.shape[0] == 0:
    st.error("❌ Failed to load or detect required columns from data/forecast_results.csv")
    st.stop()

# ------------------------
# Sidebar controls
# ------------------------
st.sidebar.header("⚙️ Inventory Parameters")
product_list = sorted(df["product"].unique())
selected_product = st.sidebar.selectbox("Select Product", product_list)

lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.number_input("Ordering Cost ($)", min_value=1.0, value=50.0, step=1.0)
holding_cost = st.sidebar.number_input("Holding Cost ($/unit)", min_value=0.01, value=5.0, step=0.5)
service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
service_choice = st.sidebar.selectbox("Service Level", list(service_levels.keys()), index=1)
z = service_levels[service_choice]

# ------------------------
# Product-specific calculations
# ------------------------
prod_df = df[df["product"] == selected_product].copy().sort_values("date")
if prod_df.empty:
    st.warning("No forecast rows for the selected product.")
    st.stop()

avg_daily_demand = prod_df["forecast"].mean()
total_demand = prod_df["forecast"].sum()
std_demand = prod_df["forecast"].std(ddof=0) if len(prod_df) > 1 else 0.0
eoq = np.sqrt((2 * total_demand * float(ordering_cost)) / float(holding_cost)) if holding_cost > 0 else np.nan
ss = float(z) * float(std_demand) * np.sqrt(float(lead_time))
rop = (avg_daily_demand * lead_time) + ss

# ------------------------
# Metric cards
# ------------------------
st.markdown("### 🔷 Inventory Snapshot")
m1, m2, m3, m4 = st.columns([1,1,1,1])
with m1:
    st.markdown(f"<div class='metric-card hover-anim'><h4>📦 Reorder Point</h4><h2>{rop:.2f}</h2><div class='small-muted'>units</div></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card hover-anim'><h4>🧮 EOQ</h4><h2>{eoq:.2f}</h2><div class='small-muted'>units</div></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='metric-card hover-anim'><h4>🛡 Safety Stock</h4><h2>{ss:.2f}</h2><div class='small-muted'>units</div></div>", unsafe_allow_html=True)
with m4:
    st.markdown(f"<div class='metric-card hover-anim'><h4>📈 Avg Daily Demand</h4><h2>{avg_daily_demand:.2f}</h2><div class='small-muted'>units/day</div></div>", unsafe_allow_html=True)

# ------------------------
# Plotly professional chart
# ------------------------
prod_df = prod_df.reset_index(drop=True)
prod_df["rolling7"] = prod_df["forecast"].rolling(window=7, min_periods=1).mean()
fig = go.Figure()

# Forecast with gradient fill
fig.add_trace(go.Scatter(
    x=prod_df["date"], y=prod_df["forecast"],
    mode="lines+markers",
    line=dict(color="#00BFFF", width=3),
    marker=dict(size=6, color="#00BFFF", line=dict(width=1, color="#021B2F")),
    name="Forecast",
    hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=prod_df["date"], y=prod_df["forecast"],
    mode="lines",
    line=dict(width=0),
    fill='tozeroy',
    fillcolor='rgba(0,191,255,0.1)',
    showlegend=False
))

# Rolling average
fig.add_trace(go.Scatter(
    x=prod_df["date"], y=prod_df["rolling7"],
    mode="lines",
    line=dict(color="#FFFFFF", width=2, dash="dash"),
    name="7-day avg",
    hovertemplate="%{x|%Y-%m-%d}<br>7-day avg: %{y:.2f}<extra></extra>"
))

# ROP and SS lines
fig.add_hline(y=rop, line_dash="dash", line_color="#FFA500",
              annotation_text=f"ROP: {rop:.2f}", annotation_position="top right",
              annotation_font=dict(color="#FFA500"))
fig.add_hline(y=ss, line_dash="dot", line_color="#FF4C4C",
              annotation_text=f"SS: {ss:.2f}", annotation_position="bottom right",
              annotation_font=dict(color="#FF4C4C"))

fig.update_layout(
    title=f"Forecast & Inventory Levels — {selected_product}",
    xaxis_title="Date",
    yaxis_title="Units",
    plot_bgcolor="#0B1D51",
    paper_bgcolor="#0B1D51",
    font=dict(color="#FFFFFF"),
    hovermode="x unified",
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    margin=dict(t=60, b=40),
    height=480
)
fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', tickcolor='#FFFFFF')
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', tickcolor='#FFFFFF')

st.plotly_chart(fig, use_container_width=True)

# ------------------------
# ABC classification table
# ------------------------
st.markdown("### 🔠 ABC Classification (by Total Forecast × Holding Cost)")
abc_df = df.groupby("product")["forecast"].sum().reset_index().rename(columns={"forecast": "TotalForecast"})
abc_df["Value"] = abc_df["TotalForecast"] * holding_cost
abc_df = abc_df.sort_values("Value", ascending=False).reset_index(drop=True)
abc_df["Cumulative%"] = abc_df["Value"].cumsum() / abc_df["Value"].sum() * 100
abc_df["Category"] = abc_df["Cumulative%"].apply(lambda x: "A" if x <= 20 else ("B" if x <= 50 else "C"))
st.dataframe(abc_df.style.format({"TotalForecast": "{:.2f}", "Value": "{:.2f}", "Cumulative%": "{:.1f}%"}), use_container_width=True)

# ------------------------
# Download inventory plan
# ------------------------
inv_df = pd.DataFrame([{
    "GeneratedOn": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Product": selected_product,
    "AvgDailyDemand": round(avg_daily_demand, 2),
    "TotalForecast": round(total_demand, 2),
    "EOQ": round(eoq, 2),
    "SafetyStock": round(ss, 2),
    "ReorderPoint": round(rop, 2),
    "LeadTime": lead_time,
    "OrderingCost": ordering_cost,
    "HoldingCost": holding_cost,
    "ServiceLevel": service_choice
}])

st.markdown("---")
st.download_button(
    label="📥 Download Inventory Plan (CSV)",
    data=inv_df.to_csv(index=False),
    file_name=f"{selected_product}_inventory_plan.csv",
    mime="text/csv"
)

# ------------------------
# Insights
# ------------------------
with st.expander("🧠 Insights & Recommendations"):
    st.markdown(f"""
**Product:** `{selected_product}`  
**Avg daily demand:** **{avg_daily_demand:.2f}** units/day  
**Std dev of forecast:** **{std_demand:.2f}**  
**Safety stock (for {service_choice}):** **{ss:.2f}** units  
**Reorder Point (ROP):** **{rop:.2f}** units  

**Recommendations**
- Maintain safety stock of **{ss:.2f}** units to reduce stockouts.
- Place orders when inventory ≤ **{rop:.2f}** units.
- Order quantity of approx **{eoq:.2f}** units for cost-optimal replenishment.
- Recompute safety stock if lead time or demand variability changes.
""")

st.caption("💡 SmartStock — navy-blue, product-wise, professional, and readable.")
