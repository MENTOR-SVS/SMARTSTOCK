import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("retail_sales_dataset_2022_2024.csv")

print("Data Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = df.dropna(subset=['date'])


df = df.drop_duplicates()


df = df.sort_values(by=['product_id','date'])


df['revenue'] = df['units_sold'] * df['price'] * (1 - df['discount(%)']/100)

print("\nAfter Cleaning:\n", df.info())

df['revenue_ma7'] = df.groupby('product_id')['revenue'].transform(lambda x: x.rolling(7,1).mean())


df['lag_1'] = df.groupby('product_id')['revenue'].shift(1)


df['promotion_flag'] = df['promotion'].map({'Yes':1, 'No':0})
df['holiday_flag'] = df['holiday'].map({'Yes':1, 'No':0})

df['month'] = df['date'].dt.to_period('M')

products = df['product_id'].unique()
for p in products[:3]:
    temp = df[df['product_id'] == p]
    plt.figure(figsize=(10,4))
    plt.plot(temp['date'], temp['revenue'], label="Daily Revenue")
    plt.plot(temp['date'], temp['revenue_ma7'], label="7-Day MA")
    plt.title(f"Revenue Trend - Product {p}")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['revenue'], bins=30, kde=True)
plt.title("Revenue Distribution")
plt.show()


monthly_sales = df.groupby(['month'])['revenue'].sum().reset_index()
plt.figure(figsize=(10,4))
plt.plot(monthly_sales['month'].astype(str), monthly_sales['revenue'])
plt.title("Monthly Revenue Trend")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(6,4))
sns.barplot(x='season', y='revenue', data=df, estimator=sum)
plt.title("Revenue by Season")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x='category', y='revenue', data=df)
plt.title("Outliers in Revenue by Category")
plt.xticks(rotation=45)
plt.show()

df.to_csv("cleaned_retail_sales.csv", index=False)
print("\nPreprocessed data saved as cleaned_retail_sales.csv")
