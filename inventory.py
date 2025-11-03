import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Inventory Optimization Dashboard", layout="wide")
st.title("📦 Milestone 3: Inventory Optimization Logic")

csv_path = "forecast_results.csv"  
if not os.path.exists(csv_path):
    st.error(f"CSV file not found at: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)
st.write("CSV Columns:", df.columns.tolist())  


products = df['product_id'].unique()
selected_product = st.sidebar.selectbox("Select Product", products)

lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit)", 1, 20, 2)
service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = service_levels[st.sidebar.selectbox("Service Level", list(service_levels.keys()), 1)]

inventory_plan = []
for product in products:
    prod_df = df[df['product_id'] == product]
    avg = prod_df['forecast_best'].mean() / 30
    demand = prod_df['forecast_best'].sum()
    std = prod_df['forecast_best'].std()
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

# ABC categorization
inv_df["Value"] = inv_df["TotalDemand"] * holding_cost
inv_df = inv_df.sort_values(by="Value", ascending=False)
inv_df["Cumulative%"] = inv_df["Value"].cumsum() / inv_df["Value"].sum() * 100
inv_df["ABC_Category"] = inv_df["Cumulative%"].apply(lambda x: "A" if x <= 20 else "B" if x <= 50 else "C")

row = inv_df[inv_df["Product"] == selected_product].iloc[0]
weeks = np.arange(1, 9)
inv_level = np.linspace(100, 30, 8)
fig, ax = plt.subplots()
ax.plot(weeks, inv_level, label="Inventory Level")
ax.axhline(y=row["ReorderPoint"], color="orange", linestyle="--", label="ROP")
ax.axhline(y=row["SafetyStock"], color="red", label="Safety Stock")
ax.legend()
st.pyplot(fig)

st.metric("Reorder Point", f"{row['ReorderPoint']:.2f}")
st.metric("EOQ", f"{row['EOQ']:.2f}")
st.metric("Safety Stock", f"{row['SafetyStock']:.2f}")

st.subheader("Inventory Plan Table")
st.dataframe(inv_df)

st.download_button("📥 Download Inventory Plan", inv_df.to_csv(index=False), "inventory_plan.csv")
