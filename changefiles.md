# Change Files Plan: Funding Rate Accumulated Exchange List Migration

## Overview
This document outlines the comprehensive plan to migrate from the single `api_futures_fundingRate_accumulated_exchange_list.py` file to four time-specific files and update the streamlit application accordingly.

## Files Affected

### Data Files Changed
- **Removed:** `api_futures_fundingRate_accumulated_exchange_list.py`
- **Added:** 
  - `api_futures_fundingRate_accumulated_exchange_list_1d.py`
  - `api_futures_fundingRate_accumulated_exchange_list_7d.py`
  - `api_futures_fundingRate_accumulated_exchange_list_30d.py`
  - `api_futures_fundingRate_accumulated_exchange_list_365d.py`

### Streamlit Files Requiring Updates
- **Primary:** `/streamlit/pages/09_basis.py` (line 108 and related functions)
- **No other files** directly reference the accumulated exchange list file

## Current Implementation Analysis

### Data Structure
**File Location:** `data/[DATE]/futures/funding_rate/api_futures_fundingRate_accumulated_exchange_list_[PERIOD].parquet`

**Data Schema:**
```
Columns: ['symbol', 'stablecoin_margin_list', 'token_margin_list']
- symbol: Asset (BTC, ETH, SOL, XRP, etc.)
- stablecoin_margin_list: List of {exchange: accumulated_rate} pairs
- token_margin_list: List of {exchange: accumulated_rate} pairs (if applicable)
```

**Sample Data (365d BTC):**
- Binance: 6.89% (accumulated over 365 days)
- OKX: 6.58% (accumulated over 365 days)
- DYDX: 14.36% (accumulated over 365 days)

### Current Usage in `09_basis.py`
1. **Line 108:** Loads single accumulated funding file
2. **Lines 790-886:** `prepare_accumulated_funding_rate_table()` processes the data
3. **Display:** Shows "365-Day Accumulated Funding" table by exchange

## Annualization Formula

### Conversion Factors
- **1d data:** Multiply by 365 (daily → annual)
- **7d data:** Multiply by 52.14 (weekly → annual: 365/7)
- **30d data:** Multiply by 12.17 (monthly → annual: 365/30)
- **365d data:** Already annualized (no multiplication needed)

### Implementation
```python
def annualize_funding_rate(rate, period):
    """
    Convert accumulated funding rate to annualized rate.
    
    Args:
        rate (float): Accumulated funding rate (as decimal, e.g., 0.0689 for 6.89%)
        period (str): Time period ('1d', '7d', '30d', '365d')
    
    Returns:
        float: Annualized rate as percentage
    """
    multipliers = {
        '1d': 365,      # Daily to annual
        '7d': 52.14,    # Weekly to annual (365/7)
        '30d': 12.17,   # Monthly to annual (365/30)
        '365d': 1       # Already annual
    }
    
    return rate * multipliers[period] * 100  # Convert to percentage
```

## Detailed Implementation Plan

### Phase 1: Update Existing Functionality

#### 1.1 Update Data Loading (`09_basis.py` line 108)
**Current Code:**
```python
accumulated_funding_file = os.path.join(data_path, 'futures', 'funding_rate', 'api_futures_fundingRate_accumulated_exchange_list.parquet')
if os.path.exists(accumulated_funding_file):
    data['accumulated_funding_rates'] = pd.read_parquet(accumulated_funding_file)
```

**Updated Code:**
```python
# Load 365d accumulated funding rates (maintains backward compatibility)
accumulated_funding_file = os.path.join(data_path, 'futures', 'funding_rate', 'api_futures_fundingRate_accumulated_exchange_list_365d.parquet')
if os.path.exists(accumulated_funding_file):
    data['accumulated_funding_rates'] = pd.read_parquet(accumulated_funding_file)
    logger.info(f"Loaded 365d accumulated funding rates: {len(data['accumulated_funding_rates'])} rows")
```

#### 1.2 Update Function Documentation
Update `prepare_accumulated_funding_rate_table()` function documentation to clarify it shows 365-day accumulated rates.

### Phase 2: Add Comprehensive Multi-Period Funding Rate Analysis

#### 2.1 New Data Loading Function
Add function to load all time periods:

```python
def load_all_accumulated_funding_data(data_path):
    """
    Load accumulated funding rate data for all time periods.
    
    Returns:
        dict: {
            '1d': DataFrame,
            '7d': DataFrame, 
            '30d': DataFrame,
            '365d': DataFrame
        }
    """
    periods = ['1d', '7d', '30d', '365d']
    funding_data = {}
    
    for period in periods:
        file_path = os.path.join(
            data_path, 
            'futures', 
            'funding_rate', 
            f'api_futures_fundingRate_accumulated_exchange_list_{period}.parquet'
        )
        
        if os.path.exists(file_path):
            funding_data[period] = pd.read_parquet(file_path)
            logger.info(f"Loaded {period} accumulated funding rates: {len(funding_data[period])} rows")
        else:
            logger.warning(f"File not found: {file_path}")
            funding_data[period] = pd.DataFrame()
    
    return funding_data
```

#### 2.2 New Comprehensive Table Function
Add new function for multi-period analysis:

```python
def prepare_comprehensive_funding_rate_table(funding_data_dict):
    """
    Create comprehensive table showing annualized funding rates across all time periods.
    
    Args:
        funding_data_dict (dict): Dictionary with DataFrames for each period
        
    Returns:
        pandas.DataFrame: Formatted table with annualized rates
    """
    if not funding_data_dict:
        return pd.DataFrame()
    
    all_rows = []
    
    # Process each asset across all time periods
    for period, df in funding_data_dict.items():
        if df.empty:
            continue
            
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            # Process stablecoin margin exchanges
            if pd.notna(row['stablecoin_margin_list']) and row['stablecoin_margin_list']:
                try:
                    exchange_data = json.loads(row['stablecoin_margin_list']) if isinstance(row['stablecoin_margin_list'], str) else row['stablecoin_margin_list']
                    
                    for exchange_info in exchange_data:
                        if isinstance(exchange_info, dict):
                            for exchange, rate_str in exchange_info.items():
                                try:
                                    # Convert rate to float and annualize
                                    rate = float(rate_str.replace('%', '')) / 100  # Convert percentage to decimal
                                    annualized_rate = annualize_funding_rate(rate, period)
                                    
                                    all_rows.append({
                                        'Asset': symbol,
                                        'Exchange': exchange,
                                        'Period': period,
                                        'Accumulated Rate (%)': f"{rate * 100:.2f}%",
                                        'Annualized Rate (%)': f"{annualized_rate:.2f}%",
                                        'Raw Rate': rate,
                                        'Raw Annualized': annualized_rate
                                    })
                                except (ValueError, AttributeError) as e:
                                    logger.warning(f"Error processing rate for {symbol}-{exchange}-{period}: {e}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing exchange data for {symbol}-{period}: {e}")
    
    if not all_rows:
        return pd.DataFrame()
    
    # Create DataFrame
    result_df = pd.DataFrame(all_rows)
    
    # Pivot to show periods as columns
    pivot_df = result_df.pivot_table(
        index=['Asset', 'Exchange'],
        columns='Period',
        values='Raw Annualized',
        aggfunc='first'
    ).reset_index()
    
    # Format columns
    period_columns = ['1d', '7d', '30d', '365d']
    for col in period_columns:
        if col in pivot_df.columns:
            pivot_df[f'{col} Annualized (%)'] = pivot_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Select and order final columns
    final_columns = ['Asset', 'Exchange']
    for period in period_columns:
        if f'{period} Annualized (%)' in pivot_df.columns:
            final_columns.append(f'{period} Annualized (%)')
    
    return pivot_df[final_columns]

def annualize_funding_rate(rate, period):
    """Convert accumulated funding rate to annualized rate."""
    multipliers = {
        '1d': 365,      # Daily to annual
        '7d': 52.14,    # Weekly to annual (365/7)
        '30d': 12.17,   # Monthly to annual (365/30)
        '365d': 1       # Already annual
    }
    
    return rate * multipliers.get(period, 1) * 100
```

#### 2.3 New Section in Basis Page
Add new section to display comprehensive funding rate analysis:

```python
# Add to main() function in 09_basis.py, after existing accumulated funding section

st.subheader("📊 Comprehensive Funding Rate Analysis")
st.write("Annualized funding rates across different time periods for all exchanges and assets.")

# Load comprehensive funding data
comprehensive_funding_data = load_all_accumulated_funding_data(latest_dir)

if any(not df.empty for df in comprehensive_funding_data.values()):
    # Period selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_mode = st.radio(
            "Display Mode:",
            ["Summary View", "Detailed View"],
            horizontal=True,
            help="Summary shows assets with highest annualized rates. Detailed shows all data."
        )
    
    with col2:
        sort_period = st.selectbox(
            "Sort by Period:",
            ["365d", "30d", "7d", "1d"],
            help="Choose which time period to use for sorting"
        )
    
    # Create comprehensive table
    comprehensive_table = prepare_comprehensive_funding_rate_table(comprehensive_funding_data)
    
    if not comprehensive_table.empty:
        # Apply filtering based on display mode
        if display_mode == "Summary View":
            # Show top 20 by selected period
            sort_col = f'{sort_period} Annualized (%)'
            if sort_col in comprehensive_table.columns:
                # Convert percentage strings to numeric for sorting
                comprehensive_table['sort_value'] = comprehensive_table[sort_col].str.replace('%', '').str.replace('N/A', '0').astype(float)
                display_table = comprehensive_table.nlargest(20, 'sort_value').drop('sort_value', axis=1)
            else:
                display_table = comprehensive_table.head(20)
        else:
            display_table = comprehensive_table
        
        # Format the table for display
        format_dict = {}
        for col in display_table.columns:
            if 'Annualized (%)' in col:
                format_dict[col] = lambda x: x  # Already formatted as percentage strings
        
        create_formatted_table(
            display_table,
            format_dict=format_dict,
            title=f"Funding Rate Analysis - {display_mode}"
        )
        
        # Add insights
        st.write("**Key Insights:**")
        
        if not comprehensive_table.empty:
            # Calculate some statistics
            for period in ['1d', '7d', '30d', '365d']:
                col_name = f'{period} Annualized (%)'
                if col_name in comprehensive_table.columns:
                    # Extract numeric values for analysis
                    numeric_values = comprehensive_table[col_name].str.replace('%', '').str.replace('N/A', '').replace('', np.nan)
                    numeric_values = pd.to_numeric(numeric_values, errors='coerce').dropna()
                    
                    if not numeric_values.empty:
                        avg_rate = numeric_values.mean()
                        max_rate = numeric_values.max()
                        max_asset_exchange = comprehensive_table.loc[
                            comprehensive_table[col_name] == f"{max_rate:.2f}%", 
                            ['Asset', 'Exchange']
                        ]
                        
                        if not max_asset_exchange.empty:
                            asset = max_asset_exchange.iloc[0]['Asset']
                            exchange = max_asset_exchange.iloc[0]['Exchange']
                            st.write(f"- **{period} Period**: Average annualized rate: {avg_rate:.2f}%, Highest: {max_rate:.2f}% ({asset} on {exchange})")
    else:
        st.warning("No comprehensive funding rate data available.")
else:
    st.warning("No accumulated funding rate data found for any time period.")
```

### Phase 3: Enhanced User Experience

#### 3.1 Add Time Period Comparison Charts
```python
def create_funding_rate_comparison_chart(comprehensive_table, selected_assets=['BTC', 'ETH']):
    """Create chart comparing funding rates across time periods for selected assets."""
    if comprehensive_table.empty:
        return None
    
    # Filter for selected assets
    filtered_data = comprehensive_table[comprehensive_table['Asset'].isin(selected_assets)]
    
    if filtered_data.empty:
        return None
    
    fig = go.Figure()
    
    periods = ['1d', '7d', '30d', '365d']
    colors = ['#FF9800', '#3F51B5', '#9C27B0', '#00BCD4']
    
    for i, period in enumerate(periods):
        col_name = f'{period} Annualized (%)'
        if col_name in filtered_data.columns:
            # Extract numeric values
            y_values = filtered_data[col_name].str.replace('%', '').str.replace('N/A', '0').astype(float)
            
            fig.add_trace(go.Bar(
                name=f'{period} Period',
                x=[f"{row['Asset']} ({row['Exchange']})" for _, row in filtered_data.iterrows()],
                y=y_values,
                marker_color=colors[i],
                hovertemplate=f'{period} Annualized Rate: %{{y:.2f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        title="Annualized Funding Rates Comparison by Time Period",
        xaxis_title="Asset (Exchange)",
        yaxis_title="Annualized Rate (%)",
        barmode='group',
        height=600
    )
    
    return apply_chart_theme(fig)
```

#### 3.2 Add Download Functionality
```python
# Add download button for comprehensive data
if not comprehensive_table.empty:
    csv_data = comprehensive_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Funding Rate Data (CSV)",
        data=csv_data,
        file_name=f"comprehensive_funding_rates_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
```

## Implementation Schedule

### Phase 1 (Immediate - Backward Compatibility)
1. Update line 108 in `09_basis.py` to use `_365d.parquet` file
2. Test existing functionality works with new file structure
3. Update function documentation

### Phase 2 (Enhanced Functionality)
1. Add new data loading functions
2. Implement comprehensive funding rate table
3. Add new section to basis page
4. Test all time period data loading and annualization

### Phase 3 (Advanced Features)
1. Add comparison charts
2. Implement download functionality  
3. Add period-based filtering and sorting
4. Performance optimization

## Testing Checklist

### Data Validation
- [ ] Verify all 4 time period files load correctly
- [ ] Confirm data structure matches expected schema
- [ ] Validate annualization calculations for each period
- [ ] Test edge cases (missing data, malformed rates)

### UI Testing
- [ ] Existing 365d table displays correctly
- [ ] New comprehensive table shows all periods
- [ ] Sorting and filtering work properly
- [ ] Download functionality works
- [ ] Mobile responsiveness maintained

### Performance Testing
- [ ] Page load times with multiple datasets
- [ ] Memory usage with large comprehensive tables
- [ ] Chart rendering performance

## Risk Mitigation

### Backward Compatibility
- Maintain existing function signatures where possible
- Graceful degradation if some period files are missing
- Clear error messages for data loading issues

### Data Quality
- Robust error handling for malformed rate data
- Validation of percentage string formats
- Fallback values for missing exchanges/assets

### User Experience
- Progressive loading indicators for large datasets
- Clear labeling of annualized vs. accumulated rates
- Helpful tooltips explaining calculations

## Success Criteria

1. **Functional**: Existing accumulated funding table continues to work
2. **Enhanced**: New comprehensive table shows all time periods with proper annualization
3. **Accurate**: All rate calculations are mathematically correct
4. **Performant**: Page loads within 3 seconds with full dataset
5. **Intuitive**: Users can easily understand and navigate the new features

This comprehensive plan ensures a smooth migration while significantly enhancing the funding rate analysis capabilities of the streamlit application.