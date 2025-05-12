

import requests

url = "https://open-api-v4.coinglass.com/api/index/bitcoin/bubble-index"

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
  "data": [
    {
      "price": 0.0495,                          // Current price
      "bubble_index": -29.59827206,             // Bubble index
      "google_trend_percent": 0.0287,           // Google trend percentage
      "mining_difficulty": 181.543,             // Mining difficulty
      "transaction_count": 235,                 // Transaction count
      "address_send_count": 390,                // Address send count
      "tweet_count": 0,                         // Tweet count
      "date_string": "2010-07-17"               // Date string
    },
    {
      "price": 0.0726,                          // Current price
      "bubble_index": -29.30591863,             // Bubble index
      "google_trend_percent": 0.0365,           // Google trend percentage
      "mining_difficulty": 181.543,             // Mining difficulty
      "transaction_count": 248,                 // Transaction count
      "address_send_count": 424,                // Address send count
      "tweet_count": 0,                         // Tweet count
      "date_string": "2010-07-18"               // Date string
    }
  ]
}

'''