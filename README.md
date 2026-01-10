# QDev - Financial Time Series Cointegration Analysis Tool

A Python library for analyzing cointegration relationships between financial assets, calculating hedge ratios, and visualizing relationships.

## Features

- 📊 **Data Fetching**: Fetch historical market data using Tiingo API
- 📈 **Hedge Ratio Calculation**: Calculate hedge ratios (Beta) between assets using OLS regression
- 📉 **Data Visualization**: Generate normalized price comparison charts and scatter regression plots
- 🔍 **Data Quality Checks**: Automatically detect missing values, anomalies, and data integrity
- ⚙️ **Flexible Configuration**: Support for command-line arguments and configuration files
- 📝 **Comprehensive Logging**: Complete logging system

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/haowu0802/qdev.git
cd qdev
```

### 2. Install Package and Dependencies

```bash
pip install .
```

Alternatively, if you only want to install dependencies without installing the package:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file and add the following configuration:

```env
TIINGO_API_KEY=your_tiingo_api_key_here
HTTP_PROXY=your_http_proxy_if_needed
HTTPS_PROXY=your_https_proxy_if_needed
```

**Get Tiingo API Key:**
1. Visit [Tiingo](https://api.tiingo.com/)
2. Register an account and get a free API Key

## Usage

### Basic Usage

Use default parameters (GLD vs GDX, 2020-2023):

```bash
python check_cointegration.py
```

### Custom Parameters

```bash
# Specify different asset pairs
python check_cointegration.py --asset-a SPY --asset-b QQQ

# Specify date range
python check_cointegration.py --start-date 2021-01-01 --end-date 2023-12-31

# Save plot to file
python check_cointegration.py --save-plot output.png

# Complete example
python check_cointegration.py \
    --asset-a GLD \
    --asset-b GDX \
    --start-date 2020-01-01 \
    --end-date 2023-12-30 \
    --save-plot results.png
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--asset-a` | First asset ticker (independent variable) | GLD |
| `--asset-b` | Second asset ticker (dependent variable) | GDX |
| `--start-date` | Start date (YYYY-MM-DD) | 2020-01-01 |
| `--end-date` | End date (YYYY-MM-DD) | 2023-12-30 |
| `--save-plot` | Path to save plot (optional) | None (interactive display) |

## Code Examples

### Python API Usage

```python
from check_cointegration import (
    fetch_market_data,
    calculate_hedge_ratio,
    preview_dataset,
    visualize_relationship
)

# Fetch data
df = fetch_market_data(
    tickers=['GLD', 'GDX'],
    start_date='2020-01-01',
    end_date='2023-12-30'
)

# Inspect data
preview_dataset(df)

# Calculate hedge ratio
beta, alpha, model = calculate_hedge_ratio(df['GDX'], df['GLD'])
print(f"Hedge Ratio (Beta): {beta:.4f}")
print(f"R-squared: {model.rsquared:.4f}")

# Visualize
visualize_relationship(df, 'GLD', 'GDX', beta, alpha, save_path='output.png')
```

## Output Explanation

### Hedge Ratio (Beta)

The hedge ratio represents the linear relationship between two assets:
- **Formula**: `Y = Beta * X + Alpha`
- **Interpretation**: For every $1 of asset B, hedge with $Beta of asset A

### R-squared

R-squared value indicates the goodness of fit of the regression model:
- **Range**: 0 to 1
- **Interpretation**: The closer to 1, the stronger the correlation between the two assets

### Visualization Charts

1. **Normalized Price Comparison Chart**: Shows relative performance of two assets (based on starting price)
2. **Scatter Regression Plot**: Shows price relationship and regression line

## Project Structure

```
qdev/
├── check_cointegration.py  # Main program file
├── requirements.txt        # Dependency list
├── README.md              # Project documentation
├── .env                   # Environment variable configuration (create manually)
└── .gitignore            # Git ignore file
```

## Dependencies

- **pandas**: Data processing and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **statsmodels**: Statistical models and regression analysis
- **tiingo**: Tiingo API client
- **python-dotenv**: Environment variable management

## Notes

1. **API Limits**: Tiingo free accounts have API call limits, please use responsibly
2. **Data Quality**: The program automatically handles missing values, but it's recommended to check data quality
3. **Timezone Handling**: Data index has timezone information removed for easier display and plotting
4. **Proxy Settings**: If using in mainland China, you may need to configure HTTP/HTTPS proxy

## FAQ

### Q: How to get Tiingo API Key?
A: Visit https://api.tiingo.com/ to register and get a free API Key.

### Q: Which asset codes are supported?
A: All stock, ETF, and other financial instrument codes supported by Tiingo API.

### Q: What to do if data fetching fails?
A: Check:
1. Whether API Key is correctly configured
2. Network connection is normal
3. Proxy settings are correct (if needed)
4. Whether asset codes exist

## License

[Please add your license information]

## Contributing

Issues and Pull Requests are welcome!

## Changelog

### v1.0.0 (2024)
- ✅ Fixed syntax errors
- ✅ Added type hints
- ✅ Improved error handling and logging system
- ✅ Fixed visualization regression line calculation (added intercept)
- ✅ Added command-line argument support
- ✅ Improved documentation and code comments
