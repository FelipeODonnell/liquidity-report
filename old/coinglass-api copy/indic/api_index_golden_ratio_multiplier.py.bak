

import requests

url = "https://open-api-v4.coinglass.com/api/index/golden-ratio-multiplier"

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
      "low_bull_high_2": 0.14,                       // Bull market low-high ratio coefficient
      "timestamp": 1282003200000,                    // Timestamp (in milliseconds)
      "price": 0.07,                                 // Current price
      "ma_350": 0.07,                                // 350-day moving average
      "accumulation_high_1_6": 0.11200000000000002,  // Accumulation high ratio (1/6 golden ratio)
      "x_3": 0.21000000000000002,                    // Golden ratio multiple x3
      "x_5": 0.35000000000000003,                    // Golden ratio multiple x5
      "x_8": 0.56,                                   // Golden ratio multiple x8
      "x_13": 0.9100000000000001,                    // Golden ratio multiple x13
      "x_21": 1.4700000000000002                     // Golden ratio multiple x21
    },
    ...
  ]
}

'''