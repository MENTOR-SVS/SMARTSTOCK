
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os


st.set_page_config(page_title="📦 Inventory Optimization Dashboard", layout="wide")
st.title("📦 Milestone 3: Inventory Optimization Dashboard")

# Try loading forecast_results.csv; fallback to retail_store_inventory.xls
if os.path.exists("data/forecast_results.csv"):
    df = pd.read_csv("data/forecast_results.csv")
elif os.path.exists("retail_store_inventory.xls"):
    df = pd.read_excel("retail_store_inventory.xls")
else:
    st.error("⚠️ File not found! Please ensure 'forecast_results.csv' or 'retail_store_inventory.xls' exists.")
    st.stop()


forecast_col = None
for col in df.columns:
    if "forecast" in col.lower():
        forecast_col = col
        break
if forecast_col is None:
    for col in df.columns:
        if "units sold" in col.lower() or "sales" in col.lower():
            forecast_col = col
            break

if forecast_col is None:
    st.error("❌ Could not find a suitable sales or forecast column (like 'Units Sold' or 'forecast_best').")
    st.stop()

df["forecast_best"] = df[forecast_col]


products = df["Product ID"].unique()
selected_product = st.sidebar.selectbox("Select Product", products)

st.sidebar.header("⚙️ Inventory Parameters")
lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit)", 1, 20, 2)
service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = service_levels[st.sidebar.selectbox("Service Level", list(service_levels.keys()), index=1)]



inventory_plan = []

for product in products:
    prod_df = df[df["Product ID"] == product]

    avg = prod_df["forecast_best"].mean() / 30
    demand = prod_df["forecast_best"].sum()
    std = prod_df["forecast_best"].std()

    eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost)
    ss = z * std * np.sqrt(lead_time)
    rop = (avg * lead_time) + ss

    inventory_plan.append({
        "Product": product,
        "AvgDailySales": round(avg, 2),
        "TotalDemand": round(demand, 2),
        "EOQ": round(eoq, 2),
        "SafetyStock": round(ss, 2),
        "ReorderPoint": round(rop, 2)
    })

inv_df = pd.DataFrame(inventory_plan)


inv_df["Value"] = inv_df["TotalDemand"] * holding_cost
inv_df = inv_df.sort_values(by="Value", ascending=False)
inv_df["Cumulative%"] = inv_df["Value"].cumsum() / inv_df["Value"].sum() * 100
inv_df["ABC_Category"] = inv_df["Cumulative%"].apply(lambda x: "A" if x <= 20 else "B" if x <= 50 else "C")


row = inv_df[inv_df["Product"] == selected_product].iloc[0]
weeks = np.arange(1, 9)
inv_level = np.linspace(100, 30, 8)

plt.figure(figsize=(7, 4))
plt.plot(weeks, inv_level, marker="o", label="Inventory Level")
plt.axhline(y=row["ReorderPoint"], color="orange", linestyle="--", label="Reorder Point")
plt.axhline(y=row["SafetyStock"], color="red", linestyle="--", label="Safety Stock")
plt.title(f"Inventory Level Simulation - {selected_product}")
plt.xlabel("Weeks")
plt.ylabel("Inventory Units")
plt.legend()
st.pyplot(plt.gcf())


st.metric("Reorder Point", f"{row['ReorderPoint']:.2f}")
st.metric("EOQ", f"{row['EOQ']:.2f}")
st.metric("Safety Stock", f"{row['SafetyStock']:.2f}")

st.dataframe(inv_df)


st.download_button("📥 Download Inventory Plan", inv_df.to_csv(index=False), "inventory_plan.csv")

st.success("✅ Inventory Optimization Completed Successfully!")
