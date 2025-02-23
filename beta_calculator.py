import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Define parameters
bitcoin_symbol = "BTC-USD"  # Bitcoin symbol on Yahoo Finance
market_symbol = "^GSPC"     # S&P 500 as market benchmark
inception_year = 2009  # Bitcoin trading started around 2010-2011

# Function to fetch data and compute Beta
def calculate_beta(start_date, end_date):
    """Fetches data, computes returns, and calculates Bitcoin Beta."""
    btc_data = yf.download(bitcoin_symbol, start=start_date, end=end_date)
    market_data = yf.download(market_symbol, start=start_date, end=end_date)

    # Ensure data is not empty
    if btc_data.empty or market_data.empty:
        print(f"⚠️ No data available for {start_date} to {end_date}")
        return None

    # Calculate daily returns
    btc_returns = btc_data['Close'].pct_change()
    market_returns = market_data['Close'].pct_change()

    # Merge returns based on index (date)
    returns_df = pd.concat([btc_returns, market_returns], axis=1, keys=['Bitcoin', 'Market']).dropna()

    # Check if there is enough data
    if len(returns_df) < 2:
        print(f"⚠️ Not enough data points for {start_date} to {end_date}")
        return None

    # Convert Pandas Series to NumPy arrays and flatten to 1D
    btc_array = returns_df['Bitcoin'].to_numpy().flatten()
    market_array = returns_df['Market'].to_numpy().flatten()

    # Compute covariance and variance
    covariance = np.cov(btc_array, market_array)[0, 1]  # BTC vs Market
    market_variance = np.var(market_array, ddof=1)  # Sample variance

    # Check if variance is zero to avoid division error
    if market_variance == 0:
        print(f"⚠️ Market variance is zero for {start_date} to {end_date}. Cannot compute beta.")
        return None

    # Compute beta
    beta = covariance / market_variance

    return beta

# Get today's date
end_date = datetime.today().strftime('%Y-%m-%d')

# Compute betas for fixed periods
timeframes = {
    "Last 1 Year": (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d'),
    "Last 5 Years": (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d'),
    "Last 10 Years": (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d'),
}

# Compute betas for fixed timeframes
for label, start_date in timeframes.items():
    beta = calculate_beta(start_date, end_date)
    if beta is not None:
        print(f"{label} Bitcoin Beta: {beta:.4f}")

# Compute Beta for every single year since Bitcoin's inception
print("\nBitcoin Beta for Each Year Since Inception:")
for year in range(inception_year, datetime.today().year):
    start_date = f"{year}-01-01"
    end_date = f"{year+1}-01-01"
    
    beta = calculate_beta(start_date, end_date)
    if beta is not None:
        print(f"{year} Bitcoin Beta: {beta:.4f}")
