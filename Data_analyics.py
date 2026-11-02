import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os .makedirs("charts", exist_ok=True)
df = pd.read_csv("raw_sales_data.csv")

print(df.head())
print(df.shape)

df.info()
print(df.isnull().sum())
print(df.duplicated().sum())

print(df.describe())

rows_before = len(df)

df = df.drop_duplicates()
print("Rows after removing duplicate", len(df))

text_columns = ["Customer_Name", "City","Product","Category","Payment_Method"]

for column in text_columns:
    df[column] = df[column].str.strip()
    df[column] = df[column].str.title()

df["City"] = df["City"].replace({"Gzb": "Ghaziabad","New Delhi": "Delhi"})
df["Payment_Method"] = df["Payment_Method"].replace({"Upi":"UPI"})

df["Unit_price"]=df["Unit_price"].astype(str)

df["Unit_price"] = df["Unit_price"].str.replace("Rs.","",regex=False)
df["Unit_price"] = df["Unit_price"].str.replace("\u20b9","",regex=False)
df["Unit_price"] = df["Unit_price"].str.replace(',',"",regex=False)
df["Unit_price"] = df["Unit_price"].str.replace()
df["Unit_price"] = pd.to_numeric(df["Unit_price"],errors="coerce")

df["Order_Date"] = pd.to_datetime(df["Order_Date"],format="mixed")

df.loc[(df["Customer_Age"] < 18)| (df["Customer_Age"] > 70),"Customer_Age"] = np.nan
df.loc[(df["Quantity"] < 1) | (df["Quantity"] > 10), "Quantity"] = np.nan
df.loc[(df["Rating"] < 1) | (df["Rating"] > 5), "Rating"] = np.nan
df.loc[(df["Discount_Payment"] < 0) | (df["Discount_Payment"] > 50),"Discount_Payments"] = np.nan

median_age = df["Customer_Age"].median()
df["Customer_Age"] = df["Customer_Age"].fillna(median_age)

median_quantity = df["Quantity"].median()
df["Quantity"] = df["Quantity"].fillna(median_quantity)

df["Discount_Percent"] = df["Discount_Percent"].fillna(0)

for product in df["Product"].unique():
    same_product = df["Product"] == product
    median_price = df.loc[same_product, "Unit_Price"].median()
    price_missing = same_product & df["Unit_price"].isnull()
    df.loc[price_missing, "Unit_Price"] = median_price


most_common_city = df["City"].mode()[0]
df["City"] = df["City"].fillna(most_common_city)
df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")
df["Customer_Name"] = df["Customer_Name"].fillna("Unknown")


df["Customer_Age"] = df["Customer_Age"].astype(int)
df["Quantity"] = df["Quantity"].astype(int)
df["Discount_Percent"] = df["Discount_Percent"].astype(int)

df = df.reset_index(drop=True)

print(df.isnull().sum())
print(df.describe())

df["Revenue"] = df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Percent"] / 100)
df["Revenue"] = df["Revenue"].round(2)


df["Month"] = df["Order_Date"].dt.month            # 1 to 12
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Weekday"] = df["Order_Date"].dt.day_name()
df["Quarter"] = df["Order_Date"].dt.quarter  


df["Age_Group"] = "46-65"
df.loc[df["Customer_Age"] <= 45, "Age_Group"] = "31-45"
df.loc[df["Customer_Age"] <= 30, "Age_Group"] = "18-30"

df.to_csv("clean_sales_data.csv", index=False)
print(df.head())


total_revenue = df["Revenue"].sum()
total_orders = len(df)
average_order = df["Revenue"].mean()
average_rating = df["Rating"].mean()

print("Total revenue:", round(total_revenue))
print("Total orders:", total_orders)
print("Average order value:", round(average_order))
print("Average rating:", round(average_rating, 2))


electronics = df[df["Category"] == "Electronics"]
big_upi_orders = df[(df["Revenue"] > 50000) & (df["Payment_Method"] == "UPI")]
delhi_q4 = df[(df["City"] == "Delhi") & (df["Quarter"] == 4)]

print("Electronics orders:", len(electronics))
print("UPI orders above Rs. 50,000:", len(big_upi_orders))
print("Delhi orders in Q4:", len(delhi_q4))


category_revenue = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
product_revenue = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
city_revenue = df.groupby("City")["Revenue"].sum().sort_values(ascending=False)
monthly_revenue = df.groupby("Month")["Revenue"].sum()
payment_counts = df["Payment_Method"].value_counts()
average_rating_by_product = df.groupby("Product")["Rating"].mean().round(2)

print(category_revenue)
print(product_revenue)
print(city_revenue)
print(payment_counts)
print(average_rating_by_product)


city_category = df.pivot_table(index="City", columns="Category",
                               values="Revenue", aggfunc="sum")
print(city_category.round(0))

# 5.5 Quick statistics with NumPy
revenue_values = df["Revenue"].values       # a NumPy array
print("Mean   :", round(np.mean(revenue_values)))
print("Median :", round(np.median(revenue_values)))
print("Maximum:", round(np.max(revenue_values)))
print("Minimum:", round(np.min(revenue_values)))
print("Std dev:", round(np.std(revenue_values)))


BLUE = "#2a78d6"
ORANGE = "#eb6834"
sns.set_style("whitegrid")
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


names = list(category_revenue.index)
values = list(category_revenue.values / 100000)    # rupees -> lakh

plt.figure(figsize=(7, 4))
plt.bar(names, values, color=BLUE)
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue (Rs. lakh)")
plt.tight_layout()
plt.savefig("charts/01_revenue_by_category.png")
plt.show()


sorted_products = product_revenue.sort_values()      # smallest first
names = list(sorted_products.index)
values = list(sorted_products.values / 100000)

plt.figure(figsize=(7, 4.5))
plt.barh(names, values, color=BLUE)
plt.title("Revenue by Product")
plt.xlabel("Revenue (Rs. lakh)")
plt.tight_layout()
plt.savefig("charts/02_revenue_by_product.png")
plt.show()


plt.figure(figsize=(7.5, 4))
plt.plot(monthly_revenue.index, monthly_revenue.values / 100000, color=BLUE, marker="o")
plt.xticks(range(1, 13), month_names)
plt.ylim(bottom=0)
plt.title("Monthly Revenue, 2025")
plt.xlabel("Month")
plt.ylabel("Revenue (Rs. lakh)")
plt.tight_layout()
plt.savefig("charts/03_monthly_revenue.png")
plt.show()


names = list(city_revenue.index)
values = list(city_revenue.values / 100000)

plt.figure(figsize=(7.5, 4))
plt.bar(names, values, color=BLUE)
plt.xlabel("City")
plt.ylabel("Revenue (Rs. lakh)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("charts/04_revenue_by_city.png")
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Payment_Method", order=payment_counts.index, color=BLUE)
plt.title("Orders by Payment Method")
plt.xlabel("Payment method")
plt.ylabel("Number of orders")
plt.tight_layout()
plt.savefig("charts/05_payment_methods.png")
plt.show()


plt.figure(figsize=(7, 4))
plt.hist(df["Customer_Age"], bins=12, color=BLUE, edgecolor="white")
plt.axvline(df["Customer_Age"].median(), color=ORANGE, linewidth=2)
plt.title("Customer Age (orange line = median)")
plt.xlabel("Age")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig("charts/06_customer_age.png")
plt.show()


plt.figure(figsize=(7.5, 4.5))
sns.heatmap(city_category / 100000, annot=True, fmt=".1f", cmap="Blues")
plt.title("Revenue (Rs. lakh): City vs Category")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/07_city_category_heatmap.png")
plt.show()


plt.figure(figsize=(7.5, 4.5))
sns.heatmap(city_category / 100000, annot=True, fmt=".1f", cmap="Blues")
plt.title("Revenue (Rs. lakh): City vs Category")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/07_city_category_heatmap.png")
plt.show()


plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Category", y="Revenue", color=BLUE)
plt.yscale("log")           # log scale because prices differ a lot
plt.title("Order Value by Category (log scale)")
plt.xlabel("Category")
plt.ylabel("Revenue per order (Rs.)")
plt.tight_layout()
plt.savefig("charts/08_order_value_boxplot.png")
plt.show()


plt.figure(figsize=(7, 4))
plt.scatter(df["Unit_Price"], df["Revenue"], color=BLUE, alpha=0.4, s=15)
plt.title("Unit Price vs Order Revenue")
plt.xlabel("Unit price (Rs.)")
plt.ylabel("Order revenue (Rs.)")
plt.tight_layout()
plt.savefig("charts/09_price_vs_revenue.png")
plt.show()


plt.figure(figsize=(11, 7))

plt.subplot(2, 2, 1)                       # 2 rows, 2 columns, chart number 1
plt.bar(category_revenue.index, category_revenue.values / 100000, color=BLUE)
plt.title("Revenue by Category (lakh)")

plt.subplot(2, 2, 2)
plt.plot(monthly_revenue.index, monthly_revenue.values / 100000, color=BLUE, marker="o")
plt.xticks(range(1, 13), month_names)
plt.ylim(bottom=0)
plt.title("Monthly Revenue (lakh)")

plt.subplot(2, 2, 3)
plt.bar(payment_counts.index, payment_counts.values, color=BLUE)
plt.xticks(rotation=20)
plt.title("Orders by Payment Method")

plt.subplot(2, 2, 4)
plt.hist(df["Customer_Age"], bins=12, color=BLUE, edgecolor="white")
plt.title("Customer Age")

plt.suptitle("Sales Dashboard 2025", fontsize=14)
plt.tight_layout()
plt.savefig("charts/10_dashboard.png")
plt.show()


best_category = category_revenue.index[0]
best_product = product_revenue.index[0]
best_city = city_revenue.index[0]
best_month = month_names[monthly_revenue.idxmax() - 1]
favourite_payment = payment_counts.index[0]

print("=" * 50)
print("KEY FINDINGS")
print("=" * 50)
print("1. Rows in raw file:", rows_before, "| rows after cleaning:", len(df))
print("2. Total revenue: Rs.", round(total_revenue), "from", total_orders, "orders")
print("3. Best category:", best_category)
print("4. Best product:", best_product)
print("5. Best city:", best_city)
print("6. Best month:", best_month)
print("7. Favourite payment method:", favourite_payment)
print("8. Average rating:", round(average_rating, 2))

