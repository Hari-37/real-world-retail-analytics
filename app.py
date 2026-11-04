import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Indian Retail Business Analytics", page_icon="🛍️", layout="wide")
st.title("🛍️ Real-World Data Project: Indian Retail Analytics")
st.write("End-to-end analysis of retail orders, revenue, profitability, customer segments and regional performance.")

@st.cache_data
def load_sample():
    return pd.read_csv(Path(__file__).parent / "indian_retail_sales_raw.csv")

uploaded = st.sidebar.file_uploader("Upload a retail CSV (optional)", type=["csv"])
raw = pd.read_csv(uploaded) if uploaded else load_sample()
st.sidebar.caption("Included sample: synthetic Indian retail transactions for educational use.")

required = ["order_date", "customer_id", "customer_type", "city", "region", "sales_channel",
            "payment_method", "product", "category", "quantity", "unit_price_inr",
            "discount_pct", "revenue_inr", "cost_inr", "profit_inr"]
missing_required = [c for c in required if c not in raw.columns]
if missing_required:
    st.error("The uploaded dataset is missing required columns: " + ", ".join(missing_required))
    st.stop()

# Cleaning and validation
data = raw.copy()
data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
duplicate_rows = int(data.duplicated().sum())
missing_cells = int(data.isna().sum().sum())
data = data.drop_duplicates().copy()
data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
for c in ["quantity", "unit_price_inr", "discount_pct", "revenue_inr", "cost_inr", "profit_inr"]:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data.loc[(data["quantity"] <= 0) | (data["quantity"] > 100), "quantity"] = np.nan
data.loc[(data["unit_price_inr"] <= 0) | (data["unit_price_inr"] > 200000), "unit_price_inr"] = np.nan
data.loc[(data["discount_pct"] < 0) | (data["discount_pct"] > 80), "discount_pct"] = np.nan

for c in data.select_dtypes(include=np.number).columns:
    data[c] = data[c].fillna(data[c].median())
for c in data.select_dtypes(include=["object", "string"]).columns:
    mode = data[c].mode(dropna=True)
    data[c] = data[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
if data["order_date"].isna().any():
    mode_date = data["order_date"].mode()
    data["order_date"] = data["order_date"].fillna(mode_date.iloc[0] if not mode_date.empty else pd.Timestamp("2025-01-01"))

# Recalculate business measures after correcting input fields
data["discount_rate"] = data["discount_pct"] / 100
data["revenue_inr"] = (data["quantity"] * data["unit_price_inr"] * (1-data["discount_rate"])).round(2)
# Preserve the supplied unit cost ratio where available; derive profit from cost
data["cost_inr"] = pd.to_numeric(data["cost_inr"], errors="coerce").fillna(data["cost_inr"].median())
data["profit_inr"] = (data["revenue_inr"] - data["cost_inr"]).round(2)
data["profit_margin_pct"] = np.where(data["revenue_inr"] != 0, data["profit_inr"] / data["revenue_inr"] * 100, 0)
data["month"] = data["order_date"].dt.to_period("M").astype(str)

st.sidebar.header("Filter the analysis")
regions = sorted(data["region"].unique().tolist())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
categories = sorted(data["category"].unique().tolist())
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
date_min, date_max = data["order_date"].min().date(), data["order_date"].max().date()
date_range = st.sidebar.date_input("Order date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

filtered = data[data["region"].isin(selected_regions) & data["category"].isin(selected_categories)].copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["order_date"] >= start) & (filtered["order_date"] <= end + pd.Timedelta(days=1)-pd.Timedelta(seconds=1))]

st.subheader("1. Business performance overview")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Total revenue", f"₹{filtered['revenue_inr'].sum():,.0f}")
k2.metric("Total profit", f"₹{filtered['profit_inr'].sum():,.0f}")
k3.metric("Orders analysed", f"{len(filtered):,}")
k4.metric("Average profit margin", f"{filtered['profit_margin_pct'].mean():.1f}%")

with st.expander("Data quality and preprocessing details"):
    st.write(f"Original records: **{len(raw):,}**")
    st.write(f"Exact duplicate rows removed: **{duplicate_rows:,}**")
    st.write(f"Missing cells in raw file: **{missing_cells:,}**")
    st.write("Cleaning included type conversion, invalid-value checks, median/mode imputation and recalculation of revenue and profit measures.")
    st.dataframe(pd.DataFrame({"Missing values": raw.isna().sum()}).query("`Missing values` > 0"))

st.subheader("2. Revenue and profit trends")
monthly = filtered.groupby("month")[["revenue_inr","profit_inr"]].sum().sort_index()
if not monthly.empty:
    fig, ax = plt.subplots(figsize=(11,4))
    monthly.plot(marker="o", ax=ax)
    ax.set_title("Monthly revenue and profit")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (₹)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

left, right = st.columns(2)
with left:
    cat_sales = filtered.groupby("category")["revenue_inr"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    cat_sales.plot(kind="bar", ax=ax)
    ax.set_title("Revenue by product category")
    ax.set_ylabel("Revenue (₹)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    region_profit = filtered.groupby("region")["profit_inr"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    region_profit.plot(kind="bar", ax=ax)
    ax.set_title("Profit by region")
    ax.set_ylabel("Profit (₹)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("3. Customer and channel analysis")
a,b = st.columns(2)
with a:
    customer_rev = filtered.groupby("customer_type")["revenue_inr"].sum()
    fig, ax = plt.subplots(figsize=(6,4))
    customer_rev.plot(kind="pie", autopct="%1.1f%%", ax=ax, ylabel="")
    ax.set_title("Revenue share by customer type")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with b:
    channel_rev = filtered.groupby("sales_channel")["revenue_inr"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    channel_rev.plot(kind="bar", ax=ax)
    ax.set_title("Revenue by sales channel")
    ax.set_ylabel("Revenue (₹)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("4. Product performance")
product_perf = filtered.groupby(["product","category"]).agg(
    orders=("order_id","count"),
    units_sold=("quantity","sum"),
    revenue=("revenue_inr","sum"),
    profit=("profit_inr","sum"),
    avg_margin=("profit_margin_pct","mean")
).sort_values("revenue", ascending=False)
st.dataframe(product_perf.style.format({
    "revenue":"₹{:,.0f}", "profit":"₹{:,.0f}", "avg_margin":"{:.1f}%"
}), use_container_width=True)

st.subheader("5. Factors associated with profitability")
numeric = filtered[["quantity","unit_price_inr","discount_pct","revenue_inr","cost_inr","profit_inr","profit_margin_pct"]].corr()
fig, ax = plt.subplots(figsize=(9,6))
sns.heatmap(numeric, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
ax.set_title("Correlation heatmap of numeric business variables")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Correlation describes linear association and does not establish that one factor causes another.")

st.subheader("6. Business insights and recommendations")
insights = []
if len(filtered):
    top_cat = filtered.groupby("category")["revenue_inr"].sum().idxmax()
    top_region = filtered.groupby("region")["profit_inr"].sum().idxmax()
    top_product = filtered.groupby("product")["revenue_inr"].sum().idxmax()
    top_channel = filtered.groupby("sales_channel")["revenue_inr"].sum().idxmax()
    insights.append(f"**Leading category:** {top_cat} contributes the highest revenue in the selected data.")
    insights.append(f"**Most profitable region:** {top_region} generates the highest total profit.")
    insights.append(f"**Top product by revenue:** {top_product}. Consider reviewing its stock availability and customer demand.")
    insights.append(f"**Leading channel:** {top_channel} has the highest recorded revenue.")
    insights.append(f"**Average profit margin:** {filtered['profit_margin_pct'].mean():.1f}%. Monitor discounts and product costs to protect profitability.")
    for item in insights:
        st.markdown("- " + item)
else:
    st.info("No rows match the current filters. Widen the filters to see insights.")

st.subheader("7. Download project outputs")
report = [
    "INDIAN RETAIL BUSINESS ANALYTICS REPORT",
    f"Orders analysed: {len(filtered)}",
    f"Total revenue (INR): {filtered['revenue_inr'].sum():.2f}",
    f"Total profit (INR): {filtered['profit_inr'].sum():.2f}",
    f"Average profit margin (%): {filtered['profit_margin_pct'].mean():.2f}",
    f"Duplicate rows removed: {duplicate_rows}",
    f"Missing cells in original dataset: {missing_cells}",
    "",
    "INSIGHTS AND RECOMMENDATIONS:"
] + [x.replace("**","") for x in insights]
st.download_button("⬇️ Download business analysis report (TXT)", data="\n".join(report),
                   file_name="indian_retail_business_report.txt", mime="text/plain")
st.download_button("⬇️ Download cleaned and analysed data (CSV)", data=filtered.to_csv(index=False).encode("utf-8"),
                   file_name="cleaned_indian_retail_sales.csv", mime="text/csv")

st.caption("Educational project using synthetic retail transactions. Currency values are in Indian Rupees (₹).")
