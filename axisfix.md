# Y-Axis Formatting Issue: 'G' Instead of 'M' for Millions

## Problem Description

In the Streamlit app charts, when displaying values in millions on the y-axis, the label shows 'G' instead of the expected 'M'. This is confusing for users expecting standard financial notation.

## Root Cause Analysis

The issue stems from using Plotly's SI (International System of Units) notation format in the `tickformat` parameter.

### Current Implementation
```python
fig.update_yaxes(title_text="Volume (USD)", secondary_y=True, tickformat='$,.0s')
```

### Why 'G' Appears

When using `tickformat=',.0s'` or similar SI notation formats in Plotly:
- The 's' format specifier uses SI prefixes
- SI notation follows scientific standards:
  - k = kilo (10³) = thousands
  - M = mega (10⁶) = millions
  - **G = giga (10⁹) = billions** ← This is what you're seeing
  - T = tera (10¹²) = trillions

So when your data is in billions, Plotly correctly shows 'G' according to SI standards, but this is confusing in financial contexts where 'B' is expected for billions.

## The Solution

### Option 1: Remove SI Notation (Recommended)
Let Plotly use its default formatting which is more appropriate for financial data:

```python
# Instead of tickformat='$,.0s'
fig.update_yaxes(
    title_text="Volume (USD)", 
    secondary_y=True,
    tickprefix="$",
    tickformat=","  # Just use comma separator without SI notation
)
```

This will display:
- 1K for thousands
- 1M for millions
- 1B for billions

### Option 2: Custom Tick Formatting
For complete control over the formatting, use a custom function:

```python
def format_axis_value(value):
    """Format axis values with financial notation"""
    if value >= 1e12:
        return f'${value/1e12:.1f}T'
    elif value >= 1e9:
        return f'${value/1e9:.1f}B'
    elif value >= 1e6:
        return f'${value/1e6:.1f}M'
    elif value >= 1e3:
        return f'${value/1e3:.1f}K'
    else:
        return f'${value:.0f}'

# Apply custom formatting
fig.update_yaxes(
    tickmode='array',
    tickvals=tick_values,
    ticktext=[format_axis_value(v) for v in tick_values]
)
```

### Option 3: Use Plotly's Built-in Financial Formatting
For simpler cases, use Plotly's abbreviated number format:

```python
fig.update_yaxes(
    title_text="Volume (USD)",
    tickformat="$~s",  # The ~ uses common abbreviations (K, M, B)
    secondary_y=True
)
```

## Files Affected

1. **streamlit/pages/01_report.py**
   - Line 671: `tickformat='$,.0s'` → Change to `tickformat=","` with `tickprefix="$"`

2. **streamlit/pages/04_spot.py**
   - Check for similar SI notation usage

3. **streamlit/pages/07_options.py**
   - Check for similar SI notation usage

## Implementation Steps

1. Search for all occurrences of `tickformat` containing 's' format specifier
2. Replace with appropriate financial formatting
3. Test with data in different ranges (thousands, millions, billions)
4. Ensure consistency across all charts in the application

## Testing

After making changes, verify:
- Values in thousands show as "1K" or "1,000"
- Values in millions show as "1M" or "1,000,000"
- Values in billions show as "1B" or "1,000,000,000"
- Currency symbols appear correctly
- Decimal places are appropriate for the context

## Additional Considerations

- Consider creating a standardized formatting function in `utils/formatters.py`
- Apply consistent formatting across all financial charts
- Document the formatting convention for future developers