# milestone1_preprocessing.py
# -----------------------------
# Milestone 1: Data Preprocessing & EDA
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
csv_path = "data/sales_data.csv"  # Make sure the CSV is in 'data/' folder
df = pd.read_csv(csv_path)

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# Create Year and Month columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

print("Dataset loaded. Shape:", df.shape)
print("Years in dataset:", df['Year'].unique())
print(df.head())

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df['Units Sold'] = df['Units Sold'].fillna(df['Units Sold'].median())
df['Units Ordered'] = df['Units Ordered'].fillna(df['Units Ordered'].median())
df = df.drop_duplicates()
df = df[(df['Units Sold'] >= 0) & (df['Units Ordered'] >= 0)]

missing_percent = df.isnull().mean() * 100
print("Missing Values (%):\n", missing_percent)

# -----------------------------
# 3. EDA
# -----------------------------
# Monthly Sales Trend
monthly_sales = df.groupby(df['Date'].dt.to_period("M"))['Units Sold'].sum()
plt.figure(figsize=(12,6))
monthly_sales.plot(marker='o', title="Monthly Sales Trend (Units Sold)")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.show()

# Outlier Detection
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Units Sold'])
plt.title("Outlier Detection: Units Sold")
plt.show()

# Day of Week Sales
df['DayOfWeek'] = df['Date'].dt.day_name()
dow_sales = df.groupby('DayOfWeek')['Units Sold'].mean().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)
plt.figure(figsize=(8,5))
dow_sales.plot(kind='bar', title="Average Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Avg Units Sold")
plt.show()

# Holiday Season Effect (Nov-Dec)
df['is_holiday_season'] = df['Date'].dt.month.isin([11,12]).astype(int)
holiday_sales = df.groupby('is_holiday_season')['Units Sold'].mean()
plt.figure(figsize=(6,4))
holiday_sales.plot(kind='bar', title="Average Sales: Holiday vs Non-Holiday Season")
plt.xlabel("Holiday Season Flag (0=No, 1=Yes)")
plt.ylabel("Avg Units Sold")
plt.show()

# Product-wise Sales
if 'Product ID' in df.columns:
    top_products = df.groupby(['Product ID','Category'])['Units Sold'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12,6))
    top_products.plot(kind='bar', title="Top 10 Best-Selling Products (with Category)")
    plt.xlabel("Product ID, Category")
    plt.ylabel("Total Units Sold")
    plt.show()

# Sales Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Units Sold'], bins=30, kde=True)
plt.title("Distribution of Units Sold")
plt.show()

# Promotions vs Sales
if 'Promotion' in df.columns:
    df['promotion_flag'] = df['Promotion'].astype(int)
    promo_sales = df.groupby('promotion_flag')['Units Sold'].mean()
    plt.figure(figsize=(6,4))
    promo_sales.plot(kind='bar', title="Average Sales: Promotion vs No Promotion")
    plt.xlabel("Promotion Flag (0=No, 1=Yes)")
    plt.ylabel("Avg Units Sold")
    plt.show()
    
# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 4. Feature Engineering
# -----------------------------
df['lag_sales'] = df['Units Sold'].shift(1)
df['weekly_rolling_avg'] = df['Units Sold'].rolling(window=7).mean()

print("Feature Engineering Done. Columns now:", df.columns)

# -----------------------------
# 5. Save Cleaned Dataset
# -----------------------------
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cleaned_csv = os.path.join(output_dir, "sales_data_cleaned.csv")
df.to_csv(cleaned_csv, index=False)
print(f"✅ Cleaned data saved as: {cleaned_csv}")
