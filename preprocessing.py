import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("graphs", exist_ok=True)
df = pd.read_csv("retail_sales_dataset.csv")
print("Raw Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)
df = df.drop_duplicates()
df = df.sort_values(by=['product_id', 'date'])

print("\nAfter Cleaning Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Create 7-day rolling average feature for units sold
df['units_sold_ma7'] = df.groupby('product_id')['units_sold'].transform(
    lambda x: x.rolling(7, 1).mean()
)

# Create lag feature (previous day units sold)
df['lag_1'] = df.groupby('product_id')['units_sold'].shift(1)

# Calculate revenue metric
df['revenue'] = df['units_sold'] * df['price']

# ------------------- DAILY SALES TREND -------------------
daily_sales = df.groupby('date')['units_sold'].sum().reset_index()
plt.figure(figsize=(10,4))
plt.plot(daily_sales['date'], daily_sales['units_sold'])
plt.title("Daily Total Units Sold")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.savefig("graphs/daily_total_units_sold.png")
plt.show()

# ------------------- MONTHLY SALES TREND -------------------
df['month'] = df['date'].dt.to_period('M')
monthly_sales = df.groupby('month')['units_sold'].sum().reset_index()
plt.figure(figsize=(10,4))
plt.plot(monthly_sales['month'].astype(str), monthly_sales['units_sold'])
plt.title("Monthly Units Sold")
plt.xticks(rotation=45)
plt.savefig("graphs/monthly_units_sold.png")
plt.show()

# ------------------- CATEGORY-WISE SALES -------------------
cat_sales = df.groupby('category')['units_sold'].sum().sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=cat_sales.index, y=cat_sales.values)
plt.title("Category-wise Total Units Sold")
plt.xticks(rotation=45)
plt.savefig("graphs/category_wise_units_sold.png")
plt.show()

# ------------------- SEASON-WISE SALES -------------------
season_sales = df.groupby('season')['units_sold'].sum()
plt.figure(figsize=(6,4))
season_sales.plot(kind='bar')
plt.title("Season-wise Units Sold")
plt.savefig("graphs/season_wise_units_sold.png")
plt.show()

# ------------------- WEATHER-WISE SALES -------------------
weather_sales = df.groupby('weather')['units_sold'].sum()
plt.figure(figsize=(6,4))
weather_sales.plot(kind='bar')
plt.title("Weather-wise Units Sold")
plt.savefig("graphs/weather_wise_units_sold.png")
plt.show()

# ------------------- TOP 10 PRODUCTS -------------------
top_products = df.groupby('product_id')['units_sold'].sum().nlargest(10)
plt.figure(figsize=(10,4))
top_products.plot(kind='bar')
plt.title("Top 10 Best Selling Products")
plt.savefig("graphs/top10_products.png")
plt.show()

# ------------------- DISCOUNT IMPACT -------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x='discount(%)', y='units_sold', data=df)
plt.title("Discount vs Units Sold")
plt.savefig("graphs/discount_vs_units_sold.png")
plt.show()

# ------------------- PROMOTION IMPACT -------------------
promo_sales = df.groupby('promotion')['units_sold'].mean()
plt.figure(figsize=(6,4))
promo_sales.plot(kind='bar')
plt.title("Average Units Sold - With vs Without Promotion")
plt.savefig("graphs/promotion_vs_units_sold.png")
plt.show()

# ------------------- HOLIDAY IMPACT -------------------
holiday_sales = df.groupby('holiday')['units_sold'].mean()
plt.figure(figsize=(6,4))
holiday_sales.plot(kind='bar')
plt.title("Average Units Sold - Holiday vs Non-Holiday")
plt.savefig("graphs/holiday_vs_units_sold.png")
plt.show()

# ------------------- PRICE DISTRIBUTION -------------------
plt.figure(figsize=(6,4))
sns.histplot(df['price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.savefig("graphs/price_distribution.png")
plt.show()

# ------------------- DAILY REVENUE TREND -------------------
revenue_daily = df.groupby('date')['revenue'].sum().reset_index()
plt.figure(figsize=(10,4))
plt.plot(revenue_daily['date'], revenue_daily['revenue'])
plt.title("Daily Total Revenue")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.savefig("graphs/daily_revenue.png")
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_retail_sales.csv", index=False)
print("\nPreprocessed data saved as cleaned_retail_sales.csv")
print("All graphs saved to 'graphs' folder ✅")
