import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from tiingo import TiingoClient
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HTTP_PROXY"):
    os.environ["http_proxy"] = os.getenv("HTTP_PROXY")
if os.getenv("HTTPS_PROXY"):
    os.environ["https_proxy"] = os.getenv("HTTPS_PROXY")

print(f">>> Set proxy: {os.getenv("HTTP_PROXY"), os.environ["https_proxy"]} ...")

def fetch_market_data(tickers, start_date, end_date):
    """
    Fetches historical data using the official Tiingo Python Client.
    
    Args:
        tickers (list): List of ticker symbols (e.g., ['GLD', 'GDX']).
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: A DataFrame containing adjusted closing prices.
                      Columns are Tickers, Index is Date.
    """
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise ValueError("Error: TIINGO_API_KEY not found in .env file.")

    print(f">>> Fetching 'adjClose' for {tickers} using official TiingoClient...")
    
    client = TiingoClient({'session': True, 'api_key': api_key})

    try:
        # This results in a clean DataFrame where columns are the tickers.
        df = client.get_dataframe(tickers, 
                                        metric_name='adjClose', 
                                        startDate=start_date, 
                                        endDate=end_date, 
                                        frequency='daily')
        
        # Drop rows with NaN to ensure data alignment
        df.dropna(inplace=True)

        # Index Standardization
        # 1. Convert to Datetime
        df.index = pd.to_datetime(df.index)
        # 2. Remove Timezone (Make it Naive) to clean up display and plotting
        df.index = df.index.tz_localize(None)
        # 3. Rename index to 'Date(Index)'
        df.index.name = 'Date(Index)'

        print(f">>> Data fetched successfully. Shape: {df.shape}")
        # Preview data to ensure columns are correct
        print(f">>> Columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        print(f"[FATAL ERROR] Failed to fetch data: {e}")
        sys.exit(1)

def calculate_hedge_ratio(series_y, series_x):
    """
    Calculates Beta (Hedge Ratio) using OLS Regression.
    Formula: Y = Beta * X + Alpha
    We hold Y (GDX) and Short Beta*X (GLD).
    """
    x_with_const = sm.add_constant(series_x)
    model = sm.OLS(series_y, x_with_const).fit()
    return model.params.iloc[1]

def preview_dataset(df):
    """
    [NEW FUNCTION] 
    Inspects the DataFrame structure, boundaries, and statistics.
    Equivalent to SQL: SELECT * FROM table LIMIT 5 + Describe.
    """
    print("\n" + "="*50)
    print("DATASET INSPECTION")
    print("="*50)
    
    # 1. Structural Integrity
    print(f"Shape: {df.shape[0]} Rows x {df.shape[1]} Columns")
    print(f"Index: {df.index.name} ({df.index.dtype})")
    
    # 2. Time Boundary Check (Crucial for Time Series)
    print(f"Start Date: {df.index.min().date()}")
    print(f"End Date:   {df.index.max().date()}")
    
    # 3. Sample Data (Head & Tail)
    print("\n>>> First 3 Rows (Head):")
    print(df.head(3))
    
    print("\n>>> Last 3 Rows (Tail):")
    print(df.tail(3))
    
    # 4. Statistical Summary (Detect Anomalies)
    # count: missing values? | mean/std: scale? | min: zeros or negatives?
    print("\n>>> Statistical Summary:")
    print(df.describe())
    print("="*50 + "\n")

def visualize_relationship(df, asset_a, asset_b, beta):
    """
    Plots the raw prices and the linear relationship.
    """
    print("\n>>> Generating Visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Normalized Comparison (Rebased to 1.0)
    # This lets us see relative performance on the same scale
    (df[asset_a] / df[asset_a].iloc[0]).plot(ax=ax1, label=asset_a)
    (df[asset_b] / df[asset_b].iloc[0]).plot(ax=ax1, label=asset_b)
    ax1.set_title(f"Normalized Price Performance: {asset_a} vs {asset_b}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter Plot with Regression Line
    # Visualizing the correlation
    ax2.scatter(df[asset_a], df[asset_b], alpha=0.5, s=10, label='Daily Data')
    
    # Draw the regression line
    import numpy as np
    x_range = np.linspace(df[asset_a].min(), df[asset_a].max(), 100)
    y_pred = beta * x_range  # Simplifying intercept for visualization
    # Note: For accurate line we need intercept, but beta slope is key here
    
    ax2.set_title(f"Correlation Scatter: Beta = {beta:.4f}")
    ax2.set_xlabel(f"{asset_a} Price")
    ax2.set_ylabel(f"{asset_b} Price")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    # 1. Configuration
    asset_a = 'GLD'  # Gold ETF
    asset_b = 'GDX'  # Gold Miners ETF
    start_date = '2020-01-01'
    end_date = '2023-12-30'

    # 2. Data Ingestion
    df = fetch_market_data([asset_a, asset_b], start_date, end_date)

    # 3. Data Inspection
    preview_dataset(df)

    # 4. Logic: Calculate Beta
    # We want to find relationship: GDX = Beta * GLD
    beta = calculate_hedge_ratio(df[asset_b], df[asset_a])
    print(f"\n>>> Calculated Hedge Ratio (Beta): {beta:.4f}")
    print(f"    Interpretation: For every $1 of {asset_b}, you hedge with ${beta:.2f} of {asset_a}.")


    # 5. Visualize
    visualize_relationship(df, asset_a, asset_b, beta)

if __name__ == "__main__":
    main()