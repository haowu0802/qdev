"""
Cointegration Analysis Tool for Financial Time Series

This module provides functionality to analyze cointegration relationships
between financial assets, calculate hedge ratios, and visualize relationships.
"""

import os
import sys
import logging
import argparse
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tiingo import TiingoClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure proxy settings
http_proxy = os.getenv("HTTP_PROXY")
https_proxy = os.getenv("HTTPS_PROXY")

if http_proxy:
    os.environ["http_proxy"] = http_proxy
if https_proxy:
    os.environ["https_proxy"] = https_proxy

logger.info(f"Proxy configured - HTTP: {http_proxy}, HTTPS: {https_proxy}")

def fetch_market_data(
    tickers: List[str], 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Fetches historical market data using the official Tiingo Python Client.
    
    Args:
        tickers: List of ticker symbols (e.g., ['GLD', 'GDX']).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        
    Returns:
        DataFrame containing adjusted closing prices.
        Columns are tickers, index is Date.
        
    Raises:
        ValueError: If API key is missing or data fetch fails.
    """
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise ValueError("TIINGO_API_KEY not found in .env file.")

    logger.info(f"Fetching 'adjClose' for {tickers} from {start_date} to {end_date}")
    
    try:
        client = TiingoClient({'session': True, 'api_key': api_key})
        
        # Fetch data - results in a DataFrame where columns are the tickers
        df = client.get_dataframe(
            tickers, 
            metric_name='adjClose', 
            startDate=start_date, 
            endDate=end_date, 
            frequency='daily'
        )
        
        if df.empty:
            raise ValueError(f"No data returned for tickers {tickers}")
        
        # Drop rows with NaN to ensure data alignment
        initial_rows = len(df)
        df.dropna(inplace=True)
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with NaN values")

        # Index Standardization
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone
        df.index.name = 'Date'

        logger.info(f"Data fetched successfully. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}", exc_info=True)
        raise

def calculate_hedge_ratio(
    series_y: pd.Series, 
    series_x: pd.Series
) -> Tuple[float, float, sm.regression.linear_model.RegressionResults]:
    """
    Calculates Beta (Hedge Ratio) and Alpha using OLS Regression.
    
    Formula: Y = Beta * X + Alpha
    Interpretation: We hold Y and short Beta*X to hedge.
    
    Args:
        series_y: Dependent variable series (e.g., GDX prices).
        series_x: Independent variable series (e.g., GLD prices).
        
    Returns:
        Tuple of (beta, alpha, regression_model) where:
        - beta: Hedge ratio (slope coefficient)
        - alpha: Intercept term
        - model: Full regression model results for further analysis
    """
    if len(series_y) != len(series_x):
        raise ValueError("Series must have the same length")
    
    if len(series_y) < 2:
        raise ValueError("Insufficient data points for regression")
    
    x_with_const = sm.add_constant(series_x)
    model = sm.OLS(series_y, x_with_const).fit()
    
    beta = model.params.iloc[1]
    alpha = model.params.iloc[0]
    
    logger.info(f"Regression R-squared: {model.rsquared:.4f}")
    logger.info(f"Beta (Hedge Ratio): {beta:.4f}")
    logger.info(f"Alpha (Intercept): {alpha:.4f}")
    
    return beta, alpha, model

def preview_dataset(df: pd.DataFrame) -> None:
    """
    Inspects the DataFrame structure, boundaries, and statistics.
    
    Provides comprehensive data quality checks including:
    - Structural integrity (shape, index type)
    - Time boundaries
    - Sample data preview
    - Statistical summary
    
    Args:
        df: DataFrame to inspect (must have datetime index).
    """
    logger.info("="*50)
    logger.info("DATASET INSPECTION")
    logger.info("="*50)
    
    # 1. Structural Integrity
    logger.info(f"Shape: {df.shape[0]} Rows x {df.shape[1]} Columns")
    logger.info(f"Index: {df.index.name} ({df.index.dtype})")
    
    # 2. Time Boundary Check (Crucial for Time Series)
    logger.info(f"Start Date: {df.index.min().date()}")
    logger.info(f"End Date:   {df.index.max().date()}")
    logger.info(f"Date Range: {(df.index.max() - df.index.min()).days} days")
    
    # 3. Missing Values Check
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values detected")
    
    # 4. Sample Data (Head & Tail)
    logger.info("\nFirst 3 Rows (Head):")
    logger.info(f"\n{df.head(3)}")
    
    logger.info("\nLast 3 Rows (Tail):")
    logger.info(f"\n{df.tail(3)}")
    
    # 5. Statistical Summary
    logger.info("\nStatistical Summary:")
    logger.info(f"\n{df.describe()}")
    logger.info("="*50 + "\n")

def visualize_relationship(
    df: pd.DataFrame, 
    asset_a: str, 
    asset_b: str, 
    beta: float, 
    alpha: float,
    save_path: Optional[str] = None
) -> None:
    """
    Creates comprehensive visualizations of the asset relationship.
    
    Generates two plots:
    1. Normalized price performance comparison
    2. Scatter plot with accurate regression line
    
    Args:
        df: DataFrame with price data.
        asset_a: Name of first asset (independent variable).
        asset_b: Name of second asset (dependent variable).
        beta: Hedge ratio (slope).
        alpha: Intercept term.
        save_path: Optional path to save the figure. If None, displays interactively.
    """
    logger.info("Generating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Normalized Comparison (Rebased to 1.0)
    # This lets us see relative performance on the same scale
    normalized_a = df[asset_a] / df[asset_a].iloc[0]
    normalized_b = df[asset_b] / df[asset_b].iloc[0]
    
    normalized_a.plot(ax=ax1, label=asset_a, linewidth=1.5)
    normalized_b.plot(ax=ax1, label=asset_b, linewidth=1.5)
    ax1.set_title(f"Normalized Price Performance: {asset_a} vs {asset_b}", fontsize=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized Price (Base = 1.0)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter Plot with Accurate Regression Line
    ax2.scatter(df[asset_a], df[asset_b], alpha=0.5, s=10, label='Daily Data')
    
    # Draw the accurate regression line with intercept
    x_range = np.linspace(df[asset_a].min(), df[asset_a].max(), 100)
    y_pred = beta * x_range + alpha  # Full regression line with intercept
    
    ax2.plot(x_range, y_pred, 'r-', linewidth=2, 
             label=f'Regression: y = {beta:.4f}x + {alpha:.2f}')
    
    ax2.set_title(f"Correlation Scatter: Beta = {beta:.4f}, Alpha = {alpha:.2f}", fontsize=14)
    ax2.set_xlabel(f"{asset_a} Price ($)")
    ax2.set_ylabel(f"{asset_b} Price ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()

def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main execution function for cointegration analysis.
    
    Args:
        args: Optional command-line arguments. If None, uses default values.
    """
    # Parse arguments or use defaults
    if args is None:
        asset_a = 'GLD'  # Gold ETF
        asset_b = 'GDX'  # Gold Miners ETF
        start_date = '2020-01-01'
        end_date = '2023-12-30'
        save_plot = None
    else:
        asset_a = args.asset_a
        asset_b = args.asset_b
        start_date = args.start_date
        end_date = args.end_date
        save_plot = args.save_plot

    try:
        # 1. Data Ingestion
        df = fetch_market_data([asset_a, asset_b], start_date, end_date)

        # 2. Data Inspection
        preview_dataset(df)

        # 3. Calculate Hedge Ratio
        # We want to find relationship: asset_b = Beta * asset_a + Alpha
        beta, alpha, model = calculate_hedge_ratio(df[asset_b], df[asset_a])
        
        logger.info("\n" + "="*50)
        logger.info("HEDGE RATIO ANALYSIS")
        logger.info("="*50)
        logger.info(f"Calculated Hedge Ratio (Beta): {beta:.4f}")
        logger.info(f"Intercept (Alpha): {alpha:.4f}")
        logger.info(f"R-squared: {model.rsquared:.4f}")
        logger.info(f"\nInterpretation:")
        logger.info(f"  For every $1 of {asset_b}, hedge with ${beta:.2f} of {asset_a}.")
        logger.info(f"  The relationship explains {model.rsquared*100:.2f}% of variance.")
        logger.info("="*50)

        # 4. Visualize
        visualize_relationship(df, asset_a, asset_b, beta, alpha, save_plot)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cointegration Analysis Tool for Financial Time Series',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default analysis (GLD vs GDX, 2020-2023)
  python check_cointegration.py
  
  # Custom assets and date range
  python check_cointegration.py --asset-a SPY --asset-b QQQ --start-date 2021-01-01 --end-date 2023-12-31
  
  # Save plot to file
  python check_cointegration.py --save-plot output.png
        """
    )
    
    parser.add_argument(
        '--asset-a',
        type=str,
        default='GLD',
        help='First asset ticker (independent variable, default: GLD)'
    )
    
    parser.add_argument(
        '--asset-b',
        type=str,
        default='GDX',
        help='Second asset ticker (dependent variable, default: GDX)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date in YYYY-MM-DD format (default: 2020-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-30',
        help='End date in YYYY-MM-DD format (default: 2023-12-30)'
    )
    
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Path to save the plot (default: display interactively)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)