# Data Fix: XRP Funding Rate API Response Issue

## Problem

The script `coinglass-api/futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history_XRP.py` fails with the following error:

```
Error saving data: ("Expected bytes, got a 'float' object", 'Conversion failed for column low with type object')
```

### Root Cause

The API response contains inconsistent data types in the JSON. Most values are returned as strings (e.g., `"0.005892"`), but at least one value in the response is returned as a raw float without quotes (e.g., `0.005892`). 

Looking at the last entry in the response:
```json
{"time":1749556800000,"open":"0.009312","high":"0.009345","low":0.005892,"close":"0.005892"}
```

Notice that the `low` field value `0.005892` is not quoted, while all other numeric values in the response are quoted strings.

## Solution

The script needs to handle mixed data types in the API response. Before converting to a DataFrame, we should normalize all numeric fields to ensure they are strings. Here's the fix:

1. After parsing the JSON response, iterate through the data
2. Convert any numeric values to strings for consistency
3. Then proceed with DataFrame creation and parquet export

### Code Fix

Replace lines 33-34 in the script with:

```python
# Convert to DataFrame - handle mixed types
# Normalize all numeric values to strings
for item in response_data['data']:
    for key in ['open', 'high', 'low', 'close']:
        if key in item and not isinstance(item[key], str):
            item[key] = str(item[key])

df = pd.DataFrame(response_data['data'])
```

This ensures all OHLC values are strings before creating the DataFrame, which prevents the type conversion error.

### Alternative Solution

If you prefer to work with numeric types directly, you could convert all string values to floats instead:

```python
# Convert all OHLC values to float
for item in response_data['data']:
    for key in ['open', 'high', 'low', 'close']:
        if key in item:
            item[key] = float(item[key])

df = pd.DataFrame(response_data['data'])
```

This would create a DataFrame with proper numeric columns instead of string columns.

## Recommendation

The second solution (converting to float) is recommended as it:
- Creates a more efficient parquet file (numeric storage vs string storage)
- Allows for direct numeric operations on the data
- Is more semantically correct since these are numeric values

This appears to be an API inconsistency issue where the Coinglass API sometimes returns numeric values without quotes in the JSON response.