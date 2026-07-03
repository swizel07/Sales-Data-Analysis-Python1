import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("SampleSuperstore.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

print("\n-----------------------------")

# Number of rows and columns
print("Shape of Dataset:")
print(df.shape)

print("\n-----------------------------")

# Column names
print("Columns:")
print(df.columns)

print("\n-----------------------------")

# Information about the dataset
print("Dataset Information:")
df.info()

print("\n-----------------------------")

print("Duplicate Rows:")

print(df.duplicated().sum())

print("\n-----------------------------")

print("Descriptive Statistics:")

print(df.describe())

# Remove duplicate rows
df = df.drop_duplicates()

print("\n-----------------------------")
print("Shape after removing duplicates:")
print(df.shape)

print("\n-----------------------------")

print("Total Sales:")
print(f"Total Sales: ${df['Sales'].sum():,.2f}")


print("\nTotal Profit:")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")

print("\nAverage Sales:")
print(f"Average Sales: ${df['Sales'].mean():,.2f}")

print("\nAverage Profit:")
print(f"Average Profit: ${df['Profit'].mean():,.2f}")

print("\n-----------------------------")
print("Sales by Category")

category_sales = df.groupby("Category")["Sales"].sum()

print(category_sales)

print("\n-----------------------------")
print("Profit by Category")

category_profit = df.groupby("Category")["Profit"].sum()

print(category_profit)

print("\n-----------------------------")
print("Sales by Region")

region_sales = df.groupby("Region")["Sales"].sum()

print(region_sales)

plt.figure(figsize=(8,5))

category_sales.plot(kind="bar")

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.savefig("sales_by_category.png")
plt.show()

plt.figure(figsize=(8,5))

region_sales.plot(kind="bar", edgecolor="black")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.savefig("sales_by_region.png")
plt.show()
print("\n-----------------------------")
print("Sales by Sub-Category")

subcategory_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)

print(subcategory_sales)

print("\n-----------------------------")
print("Top 10 Selling Sub-Categories")

subcategory_sales = (
    df.groupby("Sub-Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(subcategory_sales)

plt.figure(figsize=(10,6))

subcategory_sales.head(10).plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Top 10 Selling Sub-Categories")
plt.xlabel("Sub-Category")
plt.ylabel("Sales")

plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()

plt.savefig("top10_subcategories.png")

plt.show()

plt.figure(figsize=(8,5))

category_profit.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()

plt.savefig("profit_by_category.png")

plt.show()

print("\n-----------------------------")
print("Correlation Matrix")

correlation = df[["Sales", "Profit", "Quantity", "Discount"]].corr()

print(correlation)

plt.figure(figsize=(8,6))

plt.scatter(df["Discount"], df["Profit"])

plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")

plt.grid(True)

plt.tight_layout()

plt.savefig("discount_vs_profit.png")

plt.show()