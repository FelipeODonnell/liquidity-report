# Perpetual Futures Volume Chart Implementation Plan

## Overview
Add a combined spot price and perpetual futures volume chart to the bottom of the Report page (01_report.py). This chart will display spot price as a line with futures volume on a secondary y-axis for each selected asset.

## Data Sources

### Primary Data Files
1. **Asset-Specific Volume Data**
   - Files: `data/[DATE]/futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history_[ASSET].parquet`
   - Columns:
     - `time` (int64): Unix timestamp in milliseconds
     - `aggregated_buy_volume_usd` (float64): Total buy volume in USD
     - `aggregated_sell_volume_usd` (float64): Total sell volume in USD
   - Available for: BTC, ETH, SOL, XRP
   - Granularity: Daily intervals (181 rows for ~6 months)

2. **Spot Price Data**
   - Files: `data/[DATE]/spot/spot_market/api_spot_price_history_[ASSET].parquet`
   - Contains timestamp and price data for each asset
   - Already loaded in the report page

## Implementation Details

### 1. Location in Report Page
- Replace or enhance the existing Spot Price Performance section (around line 703)
- Or add as a new section after Open Interest chart (after line 739)

### 2. Chart Components

#### A. Section Header
```python
# 5. Spot Price & Perpetual Futures Volume
st.subheader("Spot Price & Perpetual Futures Volume")
```

#### B. Chart Options
- **Timeframe selector**: Use existing `selected_timeframe` variable (24h, 7d, 30d)
- **Asset selector**: Single asset selection (BTC, ETH, SOL, XRP)
- **Display options**: Toggle to show/hide volume, show buy/sell split

#### C. Data Processing Function
Create a new function `prepare_price_volume_combined_data()`:
```python
def prepare_price_volume_combined_data(data, asset='BTC', timeframe='24h'):
    """
    Prepare combined spot price and futures volume data for chart display.
    
    Parameters:
    - data: Dictionary containing loaded data
    - asset: Specific asset like 'BTC', 'ETH', 'SOL', 'XRP'
    - timeframe: '24h', '7d', or '30d'
    
    Returns:
    - DataFrame with columns: datetime, price, total_volume, buy_volume, sell_volume
    """
```

#### D. Chart Creation Function
Create a new function `create_price_volume_combined_chart()`:
```python
def create_price_volume_combined_chart(df, asset='BTC', timeframe='24h'):
    """
    Create combined price and volume chart with dual y-axis using plotly.
    
    Parameters:
    - df: DataFrame with price and volume data
    - asset: Asset name for labeling
    - timeframe: Display timeframe
    
    Returns:
    - Plotly figure object with dual y-axis
    """
```

### 3. Chart Design

#### Primary Chart: Combined Price & Volume
- **Primary Y-axis (left)**: Spot Price - Line chart
- **Secondary Y-axis (right)**: Futures Volume - Line chart with fill
- **X-axis**: Time
- **Layout**:
  - Price line: Solid line in asset's theme color (e.g., orange for BTC)
  - Volume line: Semi-transparent line with area fill below
  - Grid lines for both y-axes
  - Hover data showing both price and volume

#### Chart Features:
1. **Dual Y-axis Configuration**:
   - Left axis: Price in USD (formatted with currency)
   - Right axis: Volume in USD (formatted with abbreviations)
   
2. **Visual Elements**:
   - Price: Solid line chart
   - Total Volume: Line chart with light fill to axis
   - Optional: Buy/Sell volume as stacked areas

3. **Interactive Elements**:
   - Synchronized hover showing both metrics
   - Zoom and pan capabilities
   - Range selector for timeframe

### 4. Implementation Example

```python
def create_price_volume_combined_chart(df, asset='BTC', timeframe='24h'):
    """Create a dual-axis chart with price and volume."""
    from plotly.subplots import make_subplots
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line trace (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['price'],
            name=f'{asset} Price',
            line=dict(color='#FF9800', width=2),
            hovertemplate='Price: $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Add volume line trace with fill (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['total_volume'],
            name='Futures Volume',
            line=dict(color='#4CAF50', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(76, 175, 80, 0.1)',
            hovertemplate='Volume: $%{y:,.0f}<extra></extra>',
            yaxis='y2'
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Volume (USD)", secondary_y=True)
    
    # Apply theme and formatting
    fig = apply_chart_theme(fig)
    
    return fig
```

### 5. Integration Steps

1. **Update data loading** in `load_report_data()`:
   ```python
   # Add futures volume data loading for each asset
   for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
       volume_key = f'futures_volume_{asset}'
       data[volume_key] = load_specific_data_file(
           f'futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history_{asset}.parquet'
       )
   ```

2. **Add chart display** in main function:
   ```python
   # 5. Spot Price & Perpetual Futures Volume
   st.subheader("Spot Price & Perpetual Futures Volume")
   
   # Asset selector
   col1, col2 = st.columns([2, 4])
   with col1:
       selected_asset = st.selectbox(
           "Select Asset", 
           options=['BTC', 'ETH', 'SOL', 'XRP'], 
           index=0,
           key="price_volume_asset"
       )
   
   # Prepare combined data
   combined_df = prepare_price_volume_combined_data(
       data, 
       asset=selected_asset, 
       timeframe=selected_timeframe
   )
   
   # Create and display chart
   if combined_df is not None and not combined_df.empty:
       price_volume_chart = create_price_volume_combined_chart(
           combined_df, 
           asset=selected_asset, 
           timeframe=selected_timeframe
       )
       display_chart(price_volume_chart)
   else:
       st.warning(f"No data available for {selected_asset}")
   ```

### 6. Styling Considerations
- Use asset-specific colors for price lines (BTC: orange, ETH: blue, etc.)
- Volume lines should be semi-transparent with light fill
- Ensure dual y-axis labels are clearly distinguished
- Apply `apply_chart_theme()` for consistency
- Format price with currency formatter, volume with abbreviations (K, M, B)

### 7. Performance Optimization
- Cache combined data processing with `@st.cache_data`
- Merge price and volume data efficiently using pandas
- Handle timezone conversions properly
- Sample data for longer timeframes if needed

## Testing Checklist
- [ ] Dual y-axis displays correctly with proper scaling
- [ ] Price and volume data align temporally
- [ ] Chart updates when asset selection changes
- [ ] Chart updates when timeframe changes
- [ ] Hover shows both price and volume data
- [ ] Volume fill doesn't obscure price line
- [ ] Chart handles missing data gracefully

## Future Enhancements
1. Add option to show buy/sell volume split
2. Include moving averages for both price and volume
3. Add correlation coefficient display
4. Implement volume-weighted average price (VWAP)
5. Add option to normalize scales for better comparison