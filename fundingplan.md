# Funding Rate Tables Implementation Plan

## Overview
This document outlines the implementation plan for adding two funding rate tables to the bottom of the report page (`streamlit/pages/01_report.py`). The tables will display current funding rates and accumulated funding rates across different exchanges for the top 20 crypto assets.

## Data Structure Analysis

### 1. Current Funding Rates Data
**File**: `data/20250527/futures/funding_rate/api_futures_fundingRate_exchange_list.parquet`

**Structure**:
- `symbol`: Cryptocurrency symbol (e.g., 'BTC', 'ETH', 'SOL', 'XRP', etc.)
- `stablecoin_margin_list`: Array of objects containing:
  - `exchange`: Exchange name (e.g., 'Binance', 'OKX', 'Bybit')
  - `funding_rate`: Current funding rate value
  - `funding_rate_interval`: Funding interval in hours (e.g., 8.0, 1.0)
  - `next_funding_time`: Timestamp for next funding time
- `token_margin_list`: Similar structure for token margin funding rates

**Total Assets**: 832 rows with different crypto assets

### 2. Accumulated Funding Rates Data
**File**: `data/20250527/futures/funding_rate/api_futures_fundingRate_accumulated_exchange_list.parquet`

**Structure**:
- `symbol`: Cryptocurrency symbol
- `stablecoin_margin_list`: Array of objects containing:
  - `exchange`: Exchange name (uppercase format)
  - `funding_rate`: Accumulated funding rate value
- `token_margin_list`: Similar structure for token margin accumulated rates

## Implementation Details

### Table 1: Current Funding Rates

#### Features:
1. **Display**: Top 20 crypto assets by default (expandable to all)
2. **Title**: "Current Funding Rates (Live)"
3. **Columns**:
   - Asset Symbol
   - Exchange columns dynamically created based on available exchanges
   - Each cell shows the funding rate based on display mode
4. **Filters**:
   - **Asset Filter**: Multi-select dropdown to choose specific assets
   - **Exchange Filter**: Multi-select dropdown to show/hide specific exchanges
   - **Rate Display**: Toggle between:
     - **Annualized Rate** (default): Calculated as `current_rate × (24 / funding_rate_interval) × 365`
       - For example: If current rate = 0.005 and interval = 8 hours, then annualized = 0.005 × 3 × 365 = 5.475%
     - **Current Rate**: Raw funding rate as provided by the exchange
   - **Margin Type**: Radio button to switch between:
     - Stablecoin Margin (default)
     - Token Margin
5. **Formatting**:
   - Percentage format with appropriate decimal places (2 for annualized, 4 for current)
   - Color coding: Green for positive, Red for negative
   - Sort by asset symbol by default
   - Include tooltip explaining the calculation method

#### Implementation Steps:
1. Create a new function `prepare_current_funding_rate_table()` that:
   - Loads the current funding rate data
   - Transforms nested array structure into a flat DataFrame
   - Calculates annualized rates when in 'annualized' mode:
     - For each rate: `annualized = current_rate × (24 / funding_rate_interval) × 365`
     - Handle missing or null intervals appropriately
   - Pivots data to have assets as rows and exchanges as columns
   - Handles missing data gracefully
2. Add filter components above the table with clear labeling
3. Apply filters and formatting based on user selections
4. Use `create_formatted_table()` with custom formatting
5. Add explanatory text about the annualization calculation

### Table 2: Accumulated Funding Rates (Past Year)

#### Features:
1. **Display**: Top 20 crypto assets by default (same as current rates table)
2. **Title**: "Accumulated Funding Rates (365 Days)"
3. **Time Period**: Fixed at 365 days (past year) - this is the data provided
4. **Columns**:
   - Asset Symbol
   - Exchange columns with accumulated funding rates over the past year
5. **Filters**:
   - **Asset Filter**: Multi-select (synchronized with Table 1)
   - **Exchange Filter**: Multi-select (synchronized with Table 1)  
   - **Margin Type**: Radio button (synchronized with Table 1)
6. **Formatting**:
   - Percentage format with 2 decimal places
   - Color gradient based on magnitude (darker colors for higher absolute values)
   - Include summary row showing average accumulated rate by exchange
   - Include note explaining this represents the total accumulated funding over 365 days

#### Implementation Steps:
1. Create a new function `prepare_accumulated_funding_rate_table()` that:
   - Loads the accumulated funding rate data
   - Transforms and pivots similar to current rates
   - Calculates averages and summary statistics
2. Synchronize filters with the current rates table
3. Add summary statistics row at bottom
4. Use `create_formatted_table()` with gradient styling

### UI Layout

```
## Funding Rates Analysis

[Filter Section]
Row 1: Asset Selection | Exchange Selection | Margin Type
Row 2: Rate Display Type (Annualized/Current)

### Current Funding Rates (Live)
[Table 1: Current funding rates with dynamic columns - Default showing annualized rates]

### Accumulated Funding Rates (365 Days)
[Table 2: Accumulated funding rates over past year with summary row]
```

### Code Structure

1. **New Functions to Add**:
   ```python
   def prepare_current_funding_rate_table(data, selected_assets=None, selected_exchanges=None, 
                                         margin_type='stablecoin', rate_display='annualized'):
       """
       Prepare current funding rate data for table display
       
       Parameters:
       - rate_display: 'annualized' (default) or 'current'
         - 'annualized': rate × (24 / interval) × 365
         - 'current': raw funding rate from exchange
       """
       pass
   
   def prepare_accumulated_funding_rate_table(data, selected_assets=None, selected_exchanges=None,
                                             margin_type='stablecoin'):
       """
       Prepare accumulated funding rate data for table display
       Note: Data represents 365 days of accumulated funding
       """
       pass
   
   def create_funding_rate_filters():
       """Create unified filter controls for both tables"""
       pass
   
   def calculate_annualized_rate(current_rate, interval_hours):
       """
       Calculate annualized funding rate
       Formula: current_rate × (24 / interval_hours) × 365
       """
       if interval_hours and interval_hours > 0:
           return current_rate * (24 / interval_hours) * 365
       return current_rate * 365  # Default to daily if interval not specified
   ```

2. **Integration Points**:
   - Add to `load_report_data()`: Load funding rate exchange list files
   - Add to `main()`: New section after Open Interest chart
   - Utilize existing `create_formatted_table()` from `components/tables.py`

### Technical Considerations

1. **Data Transformation**:
   - Handle nested array structures in parquet files
   - Efficiently pivot data for table display
   - Handle missing exchanges for certain assets

2. **Performance**:
   - Limit initial display to top 20 assets
   - Use caching for data transformation
   - Implement pagination if needed

3. **User Experience**:
   - Synchronized filters between tables
   - Clear visual hierarchy
   - Responsive design for different screen sizes
   - Tooltips explaining funding rates

4. **Error Handling**:
   - Handle missing data gracefully
   - Provide meaningful error messages
   - Default to sensible values

### Testing Requirements

1. Verify data loading and transformation
2. Test filter interactions and synchronization
3. Validate calculations for annualized rates
4. Ensure proper formatting and styling
5. Test with different asset/exchange combinations
6. Performance testing with full dataset

### Future Enhancements

1. Add export functionality for tables
2. Historical funding rate trends visualization
3. Funding rate arbitrage opportunities highlighting
4. Integration with existing funding rate charts
5. Real-time updates (if API supports)

## Summary

This implementation will add two comprehensive funding rate tables to the report page, providing users with:

1. **Current Funding Rates Table**:
   - Live funding rates across multiple exchanges
   - Default display shows annualized rates (calculated as current rate × (24/interval) × 365)
   - Option to view raw current rates
   - Clear labeling that these are real-time rates

2. **Accumulated Funding Rates Table**:
   - Historical accumulated funding rates over the past 365 days
   - Fixed time period (no time range selection needed)
   - Shows total funding accumulated over a full year
   - Helps identify long-term funding trends

Both tables will feature:
- Synchronized filtering for assets, exchanges, and margin types
- Professional formatting with color coding
- Top 20 assets by default with expansion options
- Clear explanatory text about calculations and time periods

The tables will integrate seamlessly with the existing report page structure and utilize the established component library for consistency.