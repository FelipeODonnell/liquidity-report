

'''
upgrade needed

'''

import requests

url = "https://open-api-v4.coinglass.com/api/futures/liquidation/aggregated-heatmap/model3?symbol=BTC&range=3d"

headers = {
    "accept": "application/json",
    "CG-API-KEY": "a5b89c9d85dc40ffb8144fbecf0fb18f"
}

response = requests.get(url, headers=headers)

print(response.text)

'''
Response Data
JSON

{
  "code": "0",
  "msg": "success",
  "data": {
    "y_axis": [47968.54, 48000.00, 48031.46], // Y-axis price levels
    "liquidation_leverage_data": [
      [5, 124, 2288867.26], // Each array: [X-axis index, Y-axis index, liquidation amount in USD]
      [6, 123, 318624.82],
      [7, 122, 1527940.12]
    ],
    "price_candlesticks": [
      [
        1722676500, // Timestamp (seconds)
        "61486",    // Open price
        "61596.4",  // High price
        "61434.4",  // Low price
        "61539.9",  // Close price
        "63753192.1129" // Trading volume (USD)
      ],
      [
        1722676800,
        "61539.9",
        "61610.0",
        "61480.0",
        "61590.5",
        "42311820.8720"
      ]
    ]
  }
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

BTC
range
string
required
Defaults to 3d
Time range for data aggregation. Supported values: 12h, 24h, 3d, 7d, 30d, 90d, 180d, 1y.

'''