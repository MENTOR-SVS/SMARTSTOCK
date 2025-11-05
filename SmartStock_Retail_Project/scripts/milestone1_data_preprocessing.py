# Milestone 1: Data Preprocessing, EDA & Outlier Removal (Final Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
# 0. Create folders
os.makedirs("eda_plots", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv("./data/retail_inventory_dataset.csv")  # adjust path if needed
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# 2. Data Cleaning
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
df = df.dropna(subset=['transaction_date'])

# Compute/recompute sales
df['sales'] = (df['quantity_sold'] * df['unit_price']) - df['discount_applied']
df['sales'] = df['sales'].fillna(0)

# Fill missing stock_level with median
df['stock_level'] = df['stock_level'].fillna(df['stock_level'].median())

# Fill missing numeric columns with median
for col in ['unit_price', 'quantity_sold', 'discount_applied']:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Fill festival_name where no festival
df.loc[df['festival_flag'] == 0, 'festival_name'] = "None"

# Remove duplicates
df = df.drop_duplicates()

print("\n Data cleaning complete")

# 3. EDA Plots
# Sales trend
plt.figure(figsize=(12,5))
df.groupby('transaction_date')['sales'].sum().plot(color="purple")
plt.title("Total Sales Over Time")
plt.ylabel("Sales")
plt.savefig("eda_plots/sales_trend.png")
plt.close()

# Store-wise sales
store_sales = df.groupby('store_location')['sales'].sum().reset_index()
plt.figure(figsize=(10,5))
ax = sns.barplot(data=store_sales, x='store_location', y='sales',
                 hue='store_location', palette="Set2", legend=False, errorbar=None)
plt.title("Store-wise Total Sales")
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f"{p.get_height():,.0f}", 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, color="black")
plt.tight_layout()
plt.savefig("eda_plots/store_sales.png")
plt.close()

# Top 10 Products
top_products = df.groupby('item_name')['sales'].sum().nlargest(10).reset_index()
plt.figure(figsize=(10,6))
ax = sns.barplot(data=top_products, y='item_name', x='sales',
                 hue='item_name', palette="coolwarm", legend=False, errorbar=None)
for p in ax.patches:
    ax.annotate(f"{p.get_width():,.0f}", 
                (p.get_width(), p.get_y() + p.get_height()/2), 
                ha='left', va='center', fontsize=9, color="black",
                xytext=(5,0), textcoords="offset points")
plt.title("Top 10 Products by Sales")
plt.savefig("eda_plots/top_products.png")
plt.close()

# Season-wise sales
season_sales = df.groupby('season')['sales'].sum().reset_index()
plt.figure(figsize=(7,7))
plt.pie(season_sales['sales'], labels=season_sales['season'], autopct='%1.1f%%',
        startangle=140, colors=sns.color_palette("Set2"),
        wedgeprops=dict(width=0.4))
plt.title("Season-wise Sales Distribution")
plt.savefig("eda_plots/season_donut.png")
plt.close()

# Festival vs Non-festival sales by category
festival_category = df.groupby(['festival_flag','category'])['sales'].mean().reset_index()
festival_category['festival_flag'] = festival_category['festival_flag'].map({0:"Non-Festival",1:"Festival"})
plt.figure(figsize=(12,6))
ax = sns.barplot(data=festival_category, x='category', y='sales',
                 hue='festival_flag', palette="Paired", errorbar=None)
plt.title("Festival vs Non-Festival Sales by Category")
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f"{p.get_height():,.0f}", 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', fontsize=8, color="black")
plt.tight_layout()
plt.savefig("eda_plots/festival_category.png")
plt.close()

# Category-wise pie
category_sales = df.groupby("category")["sales"].sum()
plt.figure(figsize=(7,7))
plt.pie(category_sales, labels=category_sales.index, autopct="%1.1f%%",
        startangle=140, colors=sns.color_palette("pastel"))
plt.title("Sales Share by Category")
plt.savefig("eda_plots/category_pie.png")
plt.close()

# Histogram + KDE
plt.figure(figsize=(8,5))
sns.histplot(df['sales'], bins=40, kde=True, color="skyblue")
plt.title("Sales Distribution")
plt.savefig("eda_plots/sales_hist.png")
plt.close()

# Sales distribution by category
plt.figure(figsize=(10,6))
sns.histplot(data=df, x="sales", bins=40, hue="category",
             element="step", fill=True, alpha=0.4)
plt.title("Sales Distribution by Category")
plt.savefig("eda_plots/sales_hist_category.png")
plt.close()

# Seasonal FacetGrid
g = sns.FacetGrid(df, col="season", height=4, aspect=1.2)
g.map(sns.histplot, "sales", bins=30, color="teal")
g.set_axis_labels("Sales", "Frequency")
g.fig.suptitle("Sales Distribution Across Seasons", y=1.05)
plt.savefig("eda_plots/sales_facet_season.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['sales','quantity_sold','stock_level','unit_price','discount_applied']].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.savefig("eda_plots/correlation_heatmap.png")
plt.close()

# 4. Feature Engineering
df['year'] = df['transaction_date'].dt.year
df['month'] = df['transaction_date'].dt.month
df['day'] = df['transaction_date'].dt.day
df['day_of_week'] = df['transaction_date'].dt.day_name()
print("\n Feature engineering complete")

# 5. Outlier Detection & Removal
plt.figure(figsize=(8,5))
sns.boxplot(x=df['sales'], color="lightblue")
plt.title("Boxplot of Sales (Outlier Detection)")
plt.savefig("eda_plots/sales_boxplot.png")
plt.close()

Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["outlier_flag"] = np.where((df['sales'] < lower_bound) | (df['sales'] > upper_bound), 1, 0)
outliers = df[df["outlier_flag"] == 1]
print(f"\n Outliers detected: {len(outliers)} rows")

# Scatter with outliers highlighted
plt.figure(figsize=(8,6))
sns.scatterplot(x="quantity_sold", y="sales", hue="outlier_flag",
                data=df, palette={0:"blue",1:"red"}, alpha=0.6)
plt.title("Scatter: Quantity vs Sales (Outliers Highlighted)")
plt.savefig("eda_plots/scatter_outliers.png")
plt.close()

# Remove outliers
print(f"Removing {len(outliers)} rows as outliers...")
df = df[df["outlier_flag"] == 0]

# 6. Auto Insights
print("\n AUTO-GENERATED INSIGHTS:")

top_store = store_sales.loc[store_sales['sales'].idxmax()]
print(f"- Highest sales: {top_store['store_location']} with {top_store['sales']:,.2f}")

low_store = store_sales.loc[store_sales['sales'].idxmin()]
print(f"- Lowest sales: {low_store['store_location']} with {low_store['sales']:,.2f}")

best_product = top_products.iloc[0]
print(f"- Best-selling product: {best_product['item_name']} with {best_product['sales']:,.2f}")

peak_season = season_sales.loc[season_sales['sales'].idxmax(), 'season']
print(f"- Peak season: {peak_season}")

if df.groupby('festival_flag')['sales'].mean().get(1,0) > df.groupby('festival_flag')['sales'].mean().get(0,0):
    print("- Sales are HIGHER during festivals.")

if df.groupby('promotion_flag')['sales'].mean().get(1,0) > df.groupby('promotion_flag')['sales'].mean().get(0,0):
    print("- Promotions boost sales.")

if df.groupby('holiday_flag')['sales'].mean().get(1,0) > df.groupby('holiday_flag')['sales'].mean().get(0,0):
    print("- Holidays increase sales.")

# Aggregate monthly sales per product
monthly_df = df.groupby(['item_name', pd.Grouper(key='transaction_date', freq='ME')])['sales'].sum().reset_index()

#  Outlier capping (per product, IQR method)
def cap_outliers(x):
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = max(0, q1 - 1.5 * iqr)
    return x.clip(lower, upper)

monthly_df['sales_capped'] = monthly_df.groupby('item_name')['sales'].transform(cap_outliers)

# 3. Rolling average smoothing (3-month window)
monthly_df['sales_smooth'] = monthly_df.groupby('item_name')['sales_capped'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# 4. Apply sales floor (avoid denominator issues for MAPE)
monthly_df['sales'] = monthly_df['sales_smooth'].clip(lower=5)

# Keep only final cols
final_df = monthly_df[['item_name', 'transaction_date', 'sales']]

# Save single forecasting-ready dataset
final_df.to_csv("output/cleaned_retail_inventory_dataset.csv", index=False)
print("\n cleaned dataset saved as output/cleaned_retail_inventory_dataset.csv")
