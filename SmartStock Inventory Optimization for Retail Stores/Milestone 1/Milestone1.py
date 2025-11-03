import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('retail_store_inventory.csv')
print(df.shape)
print(df.info())
print(df.head())
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '_').str.replace(')', '_').str.replace('/','_').str.replace('-','_')
print(df.columns.tolist())
missing=df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print("missing values percentage : \n", missing_pct)
print(df.duplicated().sum())
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
print("object columns : \n", obj_cols)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
print("Missing Date values after fix:", df['Date'].isna().sum())
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
print("missing values :\n", df.isnull().sum())
print(df.duplicated().sum())
#outlier detection
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {outliers.shape[0]} outliers")
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
# Outlier handling using IQR capping
for col in ['Units_Sold', 'Demand_Forecast']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Cap the values
    df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
    print(f"Outliers in {col} after capping:", ((df[col] < lower) | (df[col] > upper)).sum())
for col in ['Units_Sold', 'Demand_Forecast']:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"{col} after outlier handling")
    plt.show()
# Monthly Sales Trend
monthly_sales = df.groupby(df['Date'].dt.to_period("M"))['Units_Sold'].sum()
monthly_sales.plot(kind='line', figsize=(10,5), marker='o', title="Monthly Units Sold Trend")
plt.ylabel("Total Units Sold")
plt.xlabel("Month")
plt.show()
# Holiday season (Nov & Dec)
df['is_holiday_season'] = df['Date'].dt.month.isin([11, 12]).astype(int)
# Promotion flag (ensure binary 0/1)
df['promotion_flag'] = df['Holiday_Promotion'].apply(lambda x: 1 if x == 1 else 0)
# Lag Feature: Previous day Units Sold
df['lag_1'] = df['Units_Sold'].shift(1)
# Rolling Mean
df['rolling_mean_3'] = df['Units_Sold'].rolling(window=3).mean()
# 5. Summary
# Data Quality 
missing_percent = df.isnull().mean() * 100
data_quality = 100 - missing_percent.mean()
print("Data Quality: {:.2f}%".format(data_quality))
# Count new features created
base_columns = 15  # original dataset had 15 columns
new_features = len(df.columns) - base_columns
print("Features Created:", new_features)
print("Seasonal Patterns Considered: Holiday, Promotion, Lag, Rolling Mean")
df.to_csv("processed_retail_store_inventory.csv", index=False)
print("✅ Data Preprocessing & EDA Completed. File saved as processed_retail_store_inventory.csv")
