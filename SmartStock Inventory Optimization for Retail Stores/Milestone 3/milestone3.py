# inventory.py - Milestone 3: Inventory Optimization Logic
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Inventory Optimization", layout="wide")

#  Main Title 
st.title("📈 Inventory Optimization Dashboard")

# The script reads the output from your forecasting milestone
try:
    df = pd.read_csv("data/forecast_results.csv")
except FileNotFoundError:
    st.error("Error: 'data/forecast_results.csv' not found. Please run the forecasting script (Milestone 2) first to generate this file.")
    st.stop()


# Sidebar for User Inputs
st.sidebar.header("Configuration")
products = df['Product_ID'].unique()
selected_product = st.sidebar.selectbox("Select Product", products)

lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit)", 1, 20, 2)
service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = service_levels[st.sidebar.selectbox("Service Level", list(service_levels.keys()), 1)]

# This part calculates inventory metrics for all products
inventory_plan = []
for product in products:
    prod_df = df[df['Product_ID'] == product]
    
    # Calculate daily average sales and standard deviation from forecast
    avg = prod_df['Forecasted_Sales'].mean()
    std = prod_df['Forecasted_Sales'].std()
    
    # Calculate annual demand for EOQ formula
    annual_demand = avg * 365
    
    # EOQ Calculation
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    
    # Safety Stock (SS) and Reorder Point (ROP) Calculation
    ss = z * std * np.sqrt(lead_time)
    rop = (avg * lead_time) + ss
    
    inventory_plan.append({
        "Product": product, 
        "AvgDailySales": round(avg, 2), 
        "TotalDemand": round(annual_demand, 2),
        "EOQ": round(eoq, 2), 
        "SafetyStock": round(ss, 2), 
        "ReorderPoint": round(rop, 2)
    })

inv_df = pd.DataFrame(inventory_plan)


# Dashboard Display
st.header(f"Inventory Plan for: {selected_product}")

# Get the data for the product selected in the sidebar
row = inv_df[inv_df["Product"] == selected_product].iloc[0]

# Split the layout into columns for metrics and the graph
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Reorder Point (ROP)", f"{row['ReorderPoint']:.2f} units")
    st.metric("Economic Order Quantity (EOQ)", f"{row['EOQ']:.2f} units")
    st.metric("Safety Stock", f"{row['SafetyStock']:.2f} units")

with col2:
    # Dynamic Graph Logic
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Start at the maximum inventory level (after an EOQ order arrives)
    max_inventory = row["EOQ"] + row["SafetyStock"]
    
    # Calculate depletion over the 8-week (56-day) period
    depletion_over_period = row["AvgDailySales"] * 56
    end_inventory = max_inventory - depletion_over_period
    
    # Ensure the simulated inventory doesn't fall below zero
    end_inventory = max(0, end_inventory)
    
    # Create the array for the inventory level line
    weeks = np.arange(1, 9)
    inv_level = np.linspace(max_inventory, end_inventory, 8)
    
    ax.plot(weeks, inv_level, label="Projected Inventory Level", marker='o', color='green')
    ax.axhline(y=row["ReorderPoint"], color="orange", linestyle="--", label="Reorder Point (ROP)")
    ax.axhline(y=row["SafetyStock"], color="red", linestyle="--", label="Safety Stock")
    
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Inventory Units")
    ax.set_title("8-Week Inventory Projection")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# Data Download Button
st.download_button(
    "📥 Download Full Inventory Plan", 
    inv_df.to_csv(index=False), 
    "inventory_plan.csv"
)