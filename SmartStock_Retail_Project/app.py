# app.py — SmartStock – Milestone 4: Forecast & Inventory Dashboard
from login import login_page, forgot_password_page, logout
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- Page config ---
st.set_page_config(page_title="SmartStock Dashboard", layout="wide")

# --- Initialize session state keys (avoid KeyError, ensure login flow works) ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "forgot_mode" not in st.session_state:
    st.session_state["forgot_mode"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "selected_product" not in st.session_state:
    st.session_state["selected_product"] = None

# --- Login / Forgot-password flow: show these BEFORE rendering dashboard ---
if not st.session_state["logged_in"]:
    if st.session_state.get("forgot_mode", False):
        forgot_password_page()
        st.stop()
    else:
        login_page()
        st.stop()

# From here onward, user is logged in (st.session_state["logged_in"] == True)

# CSS for styling (kept as in your original)
st.markdown("""
    <style>
    /* Gradient header */
    .app-header {
        background: linear-gradient(90deg, #1f77b4, #00c6ff);
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .app-header img { height: 40px; margin-right: 15px; }

    /* Metrics cards */
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; }
    .stMetric {
        background: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .stMetric:hover { transform: translateY(-5px); }

    /* Tabs */
    div[role="tab"] {
        background-color: #f0f2f6;
        color: #1f77b4;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        padding: 10px;
        margin-right: 5px;
    }
    div[role="tab"]:hover { background-color: #d0e7ff; }
    div[role="tab"][aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }

    /* Table */
    .dataframe { border-collapse: collapse; width: 100%; }
    .dataframe th, .dataframe td { border: 1px solid #ddd; padding: 8px; }
    .dataframe tr:nth-child(even){background-color: #f2f2f2;}
    .dataframe tr:hover {background-color: #d0e7ff;}
    .dataframe th { padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #1f77b4; color: white; }

    /* Alerts */
    .stWarning { background-color: #fff3cd !important; color: #856404 !important; border-left: 6px solid #ffeeba; }
    .stInfo { background-color: #d1ecf1 !important; color: #0c5460 !important; border-left: 6px solid #bee5eb; }

    </style>
""", unsafe_allow_html=True)

# Header (rendered only after login)
st.markdown(f"""
    <div class="app-header">
        <div style="display:flex;align-items:center;">
            <img src="https://cdn-icons-png.flaticon.com/512/686/686104.png" alt="Logo">
            SmartStock – Retail Inventory Dashboard
        </div>
        <div>📦 Inventory Intelligence — Logged in as: <strong>{st.session_state.get('username','Owner')}</strong></div>
    </div>
""", unsafe_allow_html=True)

# Add Logout (sidebar) — uses your logout() from login.py
logout(place="sidebar")

# --- Load Forecast & Cleaned Data ---
# Accept either "data/forecast_results.csv" or "data/forecast_result.csv"
FORECAST_PATHS = [Path("data/forecast_results.csv"), Path("data/forecast_result.csv"), Path("forecast_results.csv"), Path("forecast_result.csv")]
FORECAST_FILE = None
for p in FORECAST_PATHS:
    if p.exists():
        FORECAST_FILE = p
        break

CLEANED_FILE = Path("output/cleaned_retail_inventory_dataset.csv")

if FORECAST_FILE is None:
    st.error("⚠️ forecast_results.csv (or forecast_result.csv) not found. Please run forecasting step or upload a forecast CSV to data/.")
    st.stop()

# Read forecast
df_forecast = pd.read_csv(FORECAST_FILE)
# canonicalize column names
df_forecast.columns = df_forecast.columns.str.strip().str.lower()

# mapping expected lower-case source names to canonical names used in app
mapping = {"product": "Product_ID", "forecast": "Forecasted_Demand", "date": "Date"}
for old, new in mapping.items():
    if old in df_forecast.columns:
        df_forecast = df_forecast.rename(columns={old: new})
    else:
        st.error(f"⚠️ forecast CSV missing required column: '{old}'. Found: {list(df_forecast.columns)}")
        st.stop()

# ensure Date column is datetime
df_forecast["Date"] = pd.to_datetime(df_forecast["Date"], errors="coerce")
if df_forecast["Date"].isna().all():
    st.error("⚠️ Could not parse any dates in the forecast file. Ensure 'date' column values are valid dates.")
    st.stop()

# Create ABC category safely (if Forecasted_Demand is all same/constant, qcut may fail; fallback to simple bins)
if df_forecast["Forecasted_Demand"].nunique() > 1:
    df_forecast['ABC_Category'] = pd.qcut(df_forecast['Forecasted_Demand'], 3, labels=['C', 'B', 'A'])
else:
    # fallback: equal-width bins
    df_forecast['ABC_Category'] = pd.cut(df_forecast['Forecasted_Demand'], bins=3, labels=['C', 'B', 'A'])

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 500, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit/year)", 1, 100, 5)
unit_cost = st.sidebar.slider("Unit Purchase Cost ($)", 1, 200, 50)
stockout_cost = st.sidebar.slider("Stockout Cost ($/unit)", 1, 200, 50)
service_level = st.sidebar.select_slider("Service Level", ["90%", "95%", "99%"], value="95%")
service_z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}[service_level]

# Function to color-code ABC categories (for Styler)
def highlight_abc(row):
    color = ""
    if row['ABC_Category'] == 'A':
        color = 'background-color: #d4edda; color: #155724'
    elif row['ABC_Category'] == 'B':
        color = 'background-color: #fff3cd; color: #856404'
    elif row['ABC_Category'] == 'C':
        color = 'background-color: #f8d7da; color: #721c24'
    return [color] * len(row)

# Ensure a selected product exists in session_state (default to first)
all_products = df_forecast["Product_ID"].unique().tolist()
if not all_products:
    st.error("No products found in forecast file.")
    st.stop()

if st.session_state.get("selected_product") not in all_products:
    st.session_state["selected_product"] = all_products[0]

# Allow changing product from sidebar for consistent behavior across tabs
st.sidebar.markdown("### Product selection")
st.sidebar.selectbox("Select Product", all_products, key="selected_product")

selected_product = st.session_state["selected_product"]
product_data = df_forecast[df_forecast["Product_ID"] == selected_product].sort_values("Date")

# Tabs Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Data Preview", "📈 Forecast", "⚠️ Stock Alerts", "📄 Reports", "🔄 Upload & Recalculate"])

# TAB 1: Data Preview
with tab1:
    st.markdown("### Data Preview")
    st.dataframe(df_forecast.head(50))

# TAB 2: Forecast
with tab2:
    st.markdown("### Forecast & Inventory Optimization")
    if product_data.empty:
        st.warning(f"No data for product '{selected_product}'.")
    else:
        # EOQ & ROP
        # convert available forecast to approximate annual demand: assume values are per-day or per-period
        # if forecast file represents monthly totals adjust accordingly — here we use mean->daily assumption
        daily_demand = product_data["Forecasted_Demand"].mean()
        annual_demand = daily_demand * 365
        EOQ = np.sqrt((2 * annual_demand * ordering_cost) / (holding_cost if holding_cost > 0 else 1))
        std_demand = product_data["Forecasted_Demand"].std(ddof=0) if len(product_data) > 1 else 0.0
        reorder_point = (daily_demand * lead_time) + (service_z * std_demand * np.sqrt(lead_time))

        col1, col2 = st.columns(2)
        col1.metric("Reorder Point", f"{reorder_point:.0f}")
        col2.metric("Avg EOQ", f"{EOQ:.2f}")

        # Forecast Plot - create independent figure to avoid reuse issues
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(product_data["Date"], product_data["Forecasted_Demand"], marker='o', linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Forecasted Units")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

# TAB 3: Stock Alerts
with tab3:
    st.markdown("### Stock Alerts")
    if product_data.empty:
        st.info("No product forecast available to evaluate stock alerts.")
    else:
        # Recompute suitable values for the selected product
        daily_demand = product_data["Forecasted_Demand"].mean()
        std_demand = product_data["Forecasted_Demand"].std(ddof=0) if len(product_data) > 1 else 0.0
        reorder_point = (daily_demand * lead_time) + (service_z * std_demand * np.sqrt(lead_time))

        if daily_demand > reorder_point:
            st.warning(f"⚠️ Stockout risk for {selected_product}! (Avg daily demand {daily_demand:.2f} > ROP {reorder_point:.2f})")
        elif daily_demand * 1.2 < reorder_point:
            st.info(f"📌 Overstock warning for {selected_product}. (Avg daily demand {daily_demand:.2f} << ROP {reorder_point:.2f})")
        else:
            st.success("Stock levels look reasonable vs. reorder point.")

        # Color-coded ABC table (Streamlit supports Styler)
        styled_table = product_data[["Date", "Forecasted_Demand", "ABC_Category"]].style.apply(highlight_abc, axis=1)
        st.dataframe(styled_table)

# TAB 4: Reports
with tab4:
    st.markdown("### Download Reports")
    if CLEANED_FILE.exists():
        with open(CLEANED_FILE, "rb") as f:
            data = f.read()
            st.download_button("⬇️ Download Cleaned Dataset", data, file_name="cleaned_data.csv")
    if FORECAST_FILE.exists():
        with open(FORECAST_FILE, "rb") as f:
            data = f.read()
            st.download_button("⬇️ Download Forecast Results", data, file_name=FORECAST_FILE.name)

    # Single-product EOQ / ROP summary
    if product_data.empty:
        st.info("No product selected (or product has no data) to summarize.")
    else:
        daily_demand = product_data["Forecasted_Demand"].mean()
        annual_demand = daily_demand * 365
        EOQ = np.sqrt((2 * annual_demand * ordering_cost) / (holding_cost if holding_cost > 0 else 1))
        std_demand = product_data["Forecasted_Demand"].std(ddof=0) if len(product_data) > 1 else 0.0
        reorder_point = (daily_demand * lead_time) + (service_z * std_demand * np.sqrt(lead_time))

        summary_data = pd.DataFrame({
            "Product_ID": [selected_product],
            "EOQ": [round(EOQ, 2)],
            "Reorder_Point": [round(reorder_point, 2)],
            "Lead_Time": [lead_time],
            "Service_Level": [service_level],
            "ABC_Category": [product_data["ABC_Category"].iloc[0] if len(product_data) > 0 else None]
        })
        summary_csv = summary_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download EOQ Summary", summary_csv, file_name="eoq_summary.csv")

# TAB 5: Upload & Recalculate
with tab5:
    st.markdown("### Upload New Sales/Forecast CSV & Recalculate")
    uploaded_file = st.file_uploader("Upload new forecast CSV (must contain columns: product, forecast, date or product, forecast, date)", type="csv", key="recalc_upload")
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        df_new.columns = df_new.columns.str.strip().str.lower()
        mapping_new = {"product": "Product_ID", "forecast": "Forecasted_Demand", "date": "Date"}
        for old, new in mapping_new.items():
            if old in df_new.columns:
                df_new = df_new.rename(columns={old: new})
            else:
                st.error(f"⚠️ Uploaded file missing required column: {old}")
                st.stop()
        df_new["Date"] = pd.to_datetime(df_new["Date"], errors="coerce")
        if df_new["Date"].isna().all():
            st.error("⚠️ Uploaded file's dates could not be parsed.")
            st.stop()

        if df_new["Forecasted_Demand"].nunique() > 1:
            df_new['ABC_Category'] = pd.qcut(df_new['Forecasted_Demand'], 3, labels=['C', 'B', 'A'])
        else:
            df_new['ABC_Category'] = pd.cut(df_new['Forecasted_Demand'], bins=3, labels=['C', 'B', 'A'])

        st.success("✅ New data uploaded and processed (local preview below).")

        product_list_new = df_new["Product_ID"].unique()
        selected_product_new = st.selectbox("Select Product (Updated Data)", product_list_new, key="updated_select")
        product_data_new = df_new[df_new["Product_ID"] == selected_product_new].sort_values("Date")

        # EOQ & ROP
        daily_demand_new = product_data_new["Forecasted_Demand"].mean() if not product_data_new.empty else 0
        annual_demand_new = daily_demand_new * 365
        EOQ_new = np.sqrt((2 * annual_demand_new * ordering_cost) / (holding_cost if holding_cost>0 else 1))
        std_demand_new = product_data_new["Forecasted_Demand"].std(ddof=0) if len(product_data_new) > 1 else 0.0
        reorder_point_new = (daily_demand_new * lead_time) + (service_z * std_demand_new * np.sqrt(lead_time))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Products", len(product_list_new))
        col2.metric("Avg Daily Demand", round(daily_demand_new, 2))
        col3.metric("Reorder Point", f"{reorder_point_new:.0f}")
        col4.metric("Avg EOQ", f"{EOQ_new:.2f}")

        # Stock Alerts
        if daily_demand_new > reorder_point_new:
            st.warning(f"⚠️ Stockout risk for {selected_product_new}!")
        elif daily_demand_new * 1.2 < reorder_point_new:
            st.info(f"📌 Overstock warning for {selected_product_new}.")

        # Forecast Plot
        if not product_data_new.empty:
            fig_new, ax_new = plt.subplots(figsize=(10, 4))
            ax_new.plot(product_data_new["Date"], product_data_new["Forecasted_Demand"], marker='o', linewidth=2)
            ax_new.set_xlabel("Date")
            ax_new.set_ylabel("Forecasted Units")
            ax_new.grid(True)
            st.pyplot(fig_new)
            plt.close(fig_new)

            # Color-coded ABC table
            styled_table_new = product_data_new[["Date", "Forecasted_Demand", "ABC_Category"]].style.apply(highlight_abc, axis=1)
            st.dataframe(styled_table_new)

        # EOQ summary download
        summary_data_new = pd.DataFrame({
            "Product_ID": [selected_product_new],
            "EOQ": [round(EOQ_new, 2)],
            "Reorder_Point": [round(reorder_point_new, 2)],
            "Lead_Time": [lead_time],
            "Service_Level": [service_level],
            "ABC_Category": [product_data_new["ABC_Category"].iloc[0] if len(product_data_new)>0 else None]
        })
        summary_csv_new = summary_data_new.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Updated EOQ Summary", summary_csv_new, file_name="eoq_summary_updated.csv")
