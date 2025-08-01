# Funding Rate Calculation Method

## Overview
The funding rate table in the Streamlit report displays **annualized funding rates** for various crypto assets across different time periods and exchanges. This document explains how these values are calculated.

## Data Source
The funding rates are sourced from accumulated funding rate data files stored in the following format:
- `data/[DATE]/futures/funding_rate/api_futures_fundingRate_accumulated_exchange_list_[PERIOD].parquet`
- Where `[PERIOD]` can be: `1d`, `7d`, `30d`, or `365d`

## Data Structure
Each parquet file contains three columns:
1. **symbol**: The asset symbol (e.g., BTC, ETH, SOL, XRP)
2. **stablecoin_margin_list**: JSON array of exchange-specific funding rates for stablecoin-margined contracts
3. **token_margin_list**: JSON array of exchange-specific funding rates for token-margined contracts

Each exchange entry contains:
- `exchange`: Exchange name (e.g., BINANCE, OKX, BYBIT)
- `funding_rate`: The accumulated funding rate for that period (already as a percentage)

## Calculation Process

### 1. Data Loading
The `load_all_accumulated_funding_data()` function loads funding rate data for all time periods (1d, 7d, 30d, 365d) from the parquet files.

### 2. Funding Rate Extraction
For each asset and time period, the `prepare_comprehensive_funding_rate_table()` function:
- Filters for top 20 crypto assets by market cap
- Extracts funding rates from both stablecoin and token margin lists
- Parses the JSON data to get individual exchange rates

### 3. Annualization
The **key calculation** happens in the `annualize_funding_rate()` function:

```python
def annualize_funding_rate(rate, period):
    multipliers = {
        '1d': 365,      # Daily to annual
        '7d': 52.14,    # Weekly to annual (365/7)
        '30d': 12.17,   # Monthly to annual (365/30)
        '365d': 1       # Already annual
    }
    
    return rate * multipliers.get(period, 1)
```

**Important**: The input `rate` is already a percentage value from the data source (e.g., 6.89 for 6.89%).

### 4. Example Calculation
If the 7-day accumulated funding rate for BTC on Binance is 0.21%:
- Annualized rate = 0.21 × 52.14 = 10.95%

This represents what the funding rate would be if the 7-day rate continued for an entire year.

### 5. Table Generation
The final table shows:
- **Asset**: Filtered to top 20 crypto assets
- **Exchange**: Individual exchange names
- **Period Columns**: Annualized rates for each period (1d, 7d, 30d, 365d)
- Format: Values displayed as percentages with 2 decimal places

## Key Points
1. The raw funding rates in the source data are **already accumulated** for each period
2. These accumulated rates are **already in percentage format** (not decimal)
3. The annualization simply extrapolates the period rate to a yearly equivalent
4. Stablecoin-margined and token-margined contracts are processed separately
5. Token-margined exchanges are distinguished with "_TOKEN" suffix in the exchange name
6. The table only displays stablecoin-margined rates (token-margined are filtered out in the display)

## Data Flow Summary
```
Parquet Files → Load Data → Extract Exchange Rates → Annualize → Format Table → Display
```

The funding rate values shown in the table represent the theoretical annual return (or cost) from funding payments if the observed period's rate continued unchanged for a full year.