
# Milestone 3: Inventory Optimization Logic
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Streamlit Page Setup
st.set_page_config(page_title="🧮 Inventory Optimization Dashboard", layout="wide")
st.title("📦 SmartStock – Milestone 3: Inventory Optimization Logic")
st.markdown(
    """
    This dashboard uses **forecasted sales data** to compute:
    - **Economic Order Quantity (EOQ)**  
    - **Reorder Point (ROP)**  
    - **Safety Stock**  
    - and perform **ABC Classification** for effective inventory management.
    """
)
# Load Dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "forecast_results.csv")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not read file at: {DATA_PATH}\n\nError: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")

products = df['product'].unique()
selected_product = st.sidebar.selectbox("Select Product", products)

lead_time_days = st.sidebar.slider("Lead Time (days)", 1, 60, 7)
ordering_cost = st.sidebar.number_input("Ordering Cost ($ per order)", min_value=0.0, value=50.0, step=1.0)
holding_cost = st.sidebar.number_input("Holding Cost ($ per unit per period)", min_value=0.0, value=2.0, step=0.1)

service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
svc_choice = st.sidebar.selectbox("Service Level", list(service_levels.keys()), index=1)
z_factor = service_levels[svc_choice]

DAYS_PER_PERIOD = 30  # Forecast period (monthly)

# Inventory Calculations
inventory_rows = []
for prod, prod_df in df.groupby('product'):
    total_forecast = prod_df['forecast'].sum()
    avg_daily = prod_df['forecast'].mean() / DAYS_PER_PERIOD
    std_demand = prod_df['forecast'].std(ddof=0)

    eoq = np.sqrt((2 * max(total_forecast, 1e-6) * ordering_cost) / max(holding_cost, 1e-6))
    sigma_daily = std_demand / DAYS_PER_PERIOD
    safety_stock = z_factor * sigma_daily * np.sqrt(lead_time_days)
    reorder_point = (avg_daily * lead_time_days) + safety_stock

    inventory_rows.append({
        "Product": prod,
        "AvgDailySales": round(avg_daily, 2),
        "TotalForecast": round(total_forecast, 2),
        "DemandStd_PerPeriod": round(std_demand, 2),
        "EOQ": round(eoq, 2),
        "SafetyStock": round(safety_stock, 2),
        "ReorderPoint": round(reorder_point, 2)
    })

inv_df = pd.DataFrame(inventory_rows)

# ABC Classification
inv_df["InventoryValue"] = inv_df["TotalForecast"] * holding_cost
inv_df = inv_df.sort_values(by="InventoryValue", ascending=False).reset_index(drop=True)
inv_df["Cumulative%"] = inv_df["InventoryValue"].cumsum() / inv_df["InventoryValue"].sum() * 100

def abc_label(pct):
    if pct <= 20:
        return "A (High)"
    elif pct <= 50:
        return "B (Medium)"
    else:
        return "C (Low)"

inv_df["ABC_Category"] = inv_df["Cumulative%"].apply(abc_label)

# Dynamic Visualization 
st.subheader(f"📈 Inventory Levels and Reorder Plan – {selected_product}")

selected_row = inv_df[inv_df["Product"] == selected_product].iloc[0]
weeks = np.arange(1, 9)

# Simulated inventory pattern
start_stock = selected_row["ReorderPoint"] * 2.5
depletion_rate = selected_row["AvgDailySales"] * 6
inventory_level = [start_stock - depletion_rate * (i - 1) for i in weeks]

# Refill logic when stock < ROP
for i in range(1, len(inventory_level)):
    if inventory_level[i] <= selected_row["ReorderPoint"]:
        inventory_level[i] += selected_row["EOQ"]

# --- Matplotlib Version ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(weeks, inventory_level, color="#1E88E5", linewidth=3, marker="o", markersize=8, label="Inventory Level")
ax.fill_between(weeks, inventory_level, selected_row["SafetyStock"], color="#BBDEFB", alpha=0.4)
ax.axhline(y=selected_row["ReorderPoint"], color="#FFA726", linestyle="--", linewidth=2, label="Reorder Point (ROP)")
ax.axhline(y=selected_row["SafetyStock"], color="#EF5350", linestyle="--", linewidth=2, label="Safety Stock")
ax.set_title(f"Inventory Level vs Reorder Thresholds ({selected_product})", fontsize=13, fontweight="bold")
ax.set_xlabel("Weeks", fontsize=11)
ax.set_ylabel("Stock Units", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(facecolor="white", framealpha=1, loc="upper right")
fig.patch.set_alpha(0)
st.pyplot(fig)

# Optional Interactive Plotly Graph
with st.expander("📊 View Interactive Plotly Chart"):
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(
        x=weeks, y=inventory_level,
        mode='lines+markers', name='Inventory Level',
        line=dict(color='royalblue', width=3),
        fill='tozeroy', fillcolor='rgba(173,216,230,0.3)'
    ))
    fig_plotly.add_hline(y=selected_row["ReorderPoint"], line_dash="dash", line_color="orange", annotation_text="ROP")
    fig_plotly.add_hline(y=selected_row["SafetyStock"], line_dash="dash", line_color="red", annotation_text="Safety Stock")
    fig_plotly.update_layout(
        title=f"Interactive Inventory Chart – {selected_product}",
        xaxis_title="Weeks",
        yaxis_title="Stock Units",
        template="simple_white",
        height=450
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

# Metrics Section
col1, col2, col3 = st.columns(3)
col1.metric("📦 EOQ", f"{selected_row['EOQ']:.2f}")
col2.metric("⚙️ Reorder Point", f"{selected_row['ReorderPoint']:.2f}")
col3.metric("🛡️ Safety Stock", f"{selected_row['SafetyStock']:.2f}")

st.markdown("---")

# ABC Summary Indicators
st.subheader("🔤 ABC Classification Summary")

abc_counts = inv_df["ABC_Category"].value_counts(normalize=True) * 100
colA, colB, colC = st.columns(3)
colA.metric("Category A", f"{abc_counts.get('A (High)', 0):.0f}%", "High Value", delta_color="inverse")
colB.metric("Category B", f"{abc_counts.get('B (Medium)', 0):.0f}%", "Medium Value", delta_color="off")
colC.metric("Category C", f"{abc_counts.get('C (Low)', 0):.0f}%", "Low Value", delta_color="normal")

st.markdown("**Optimization Active:** Inventory levels and reorder points automatically calculated using forecast data and cost parameters.")

# Final Table and Download
st.subheader("📋 Inventory Optimization Table (All Products)")
st.dataframe(inv_df)

csv_bytes = inv_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Inventory Optimization Report", csv_bytes, "inventory_optimization_plan.csv", "text/csv")
