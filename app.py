import streamlit as st
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import os
from datetime import datetime
import matplotlib.pyplot as plt

# --------------------------------------
# Set Streamlit page config
st.set_page_config(page_title="Stock Regression Analysis", layout="wide")
st.title("📈 NSE Stock Regression Analysis by Sector & Market Cap")

# --------------------------------------
# File Upload
uploaded_file = st.file_uploader("Upload `tickers.csv` (symbol, sector, market_cap)", type=["csv"])

# Create cache folder
CACHE_FOLDER = "price_cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)

# --------------------------------------
# Load or fetch data with caching
@st.cache_data(show_spinner=False)
def get_price_data(ticker, start_date, end_date):
    fname = os.path.join(CACHE_FOLDER, f"{ticker.replace('^','')}.parquet")
    if os.path.exists(fname):
        return pd.read_parquet(fname)
    df = yf.download(ticker, start=start_date, end=end_date)
    df.to_parquet(fname)
    return df

# --------------------------------------
# Regression logic
def run_regression(df_tickers, start_date, end_date):
    results = []

    market = get_price_data("^NSEI", start_date, end_date)
    market_price = market.get("Adj Close") or market.get("Close")
    market_return = market_price.pct_change().dropna()

    for _, row in df_tickers.iterrows():
        ticker = row["symbol"]
        try:
            stock = get_price_data(ticker, start_date, end_date)
            price = stock.get("Adj Close") or stock.get("Close")
            if price.dropna().empty:
                continue

            stock_return = price.pct_change().dropna()
            df = pd.concat([stock_return, market_return], axis=1, join="inner").dropna()
            df.columns = ["Stock_Return", "Market_Return"]

            if df.empty:
                continue

            X = sm.add_constant(df["Market_Return"])
            model = sm.OLS(df["Stock_Return"], X).fit()

            results.append({
                "Ticker": ticker,
                "Sector": row["sector"],
                "MarketCap": row["market_cap"],
                "Alpha": model.params.get("const", float("nan")),
                "Beta": model.params.get("Market_Return", float("nan")),
                "R_squared": model.rsquared,
                "P_value_Alpha": model.pvalues.get("const", float("nan")),
                "P_value_Beta": model.pvalues.get("Market_Return", float("nan")),
                "Observations": int(model.nobs)
            })

        except Exception as e:
            st.warning(f"Error with {ticker}: {e}")

    return pd.DataFrame(results)

# --------------------------------------
# UI Workflow
if uploaded_file:
    df_tickers = pd.read_csv(uploaded_file)
    if not {'symbol', 'sector', 'market_cap'}.issubset(df_tickers.columns):
        st.error("CSV must contain columns: symbol, sector, market_cap")
    else:
        with st.spinner("Running regressions..."):
            df_results = run_regression(df_tickers, "2023-01-01", "2024-12-31")

        st.success("✅ Regressions complete.")
        st.dataframe(df_results)

        # Download button
        st.download_button("📥 Download Results CSV", df_results.to_csv(index=False), file_name="regression_results.csv")

        # --------------------------------------
        # Sector analysis
        st.subheader("📊 Sector-wise Averages")
        sector_stats = df_results.groupby("Sector").agg({
            "Alpha": "mean", "Beta": "mean", "R_squared": "mean", "Ticker": "count"
        }).rename(columns={"Ticker": "Stock_Count"}).sort_values("Beta", ascending=False)
        st.dataframe(sector_stats)

        st.pyplot(plt.figure(figsize=(10,6)))
        sector_stats["Beta"].sort_values().plot(kind='barh', color='skyblue')
        plt.title("Average Beta by Sector")
        plt.xlabel("Beta")
        plt.grid(True)
        st.pyplot(plt.gcf())

        # --------------------------------------
        # Market Cap analysis
        st.subheader("🏢 Market Cap Analysis")

        def bucket(cap):
            if cap >= 1_000_000:
                return "Large Cap"
            elif cap >= 200_000:
                return "Mid Cap"
            else:
                return "Small Cap"

        df_results["Cap_Category"] = df_results["MarketCap"].apply(bucket)
        cap_stats = df_results.groupby("Cap_Category").agg({
            "Alpha": "mean", "Beta": "mean", "R_squared": "mean", "Ticker": "count"
        }).rename(columns={"Ticker": "Stock_Count"})

        st.dataframe(cap_stats)

        cap_stats["Beta"].plot(kind='bar', color=['#3A8FB7', '#F7C242', '#D94F4F'])
        plt.title("Average Beta by Market Cap Category")
        plt.ylabel("Beta")
        plt.xticks(rotation=0)
        plt.grid(True)
        st.pyplot(plt.gcf())

        # --------------------------------------
        # Scatter: Beta vs Market Cap
        st.subheader("📈 Beta vs Market Cap")
        st.pyplot(plt.figure(figsize=(10,6)))
        df_results.plot.scatter(x="MarketCap", y="Beta", alpha=0.6)
        plt.title("Beta vs Market Cap")
        plt.grid(True)
        st.pyplot(plt.gcf())
