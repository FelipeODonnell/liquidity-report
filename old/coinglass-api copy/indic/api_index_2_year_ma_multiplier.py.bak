

import requests

url = "https://open-api-v4.coinglass.com/api/index/2-year-ma-multiplier"

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
      "timestamp": 1282003200000,               // Timestamp (in milliseconds)
      "price": 0.07,                            // Current price
      "moving_average_730": 0.07,               // 2-year moving average (730 represents the period)
      "moving_average_730_multiplier_5": 0.35000000000000003, // 5 times the 2-year moving average (Multiplier)
    }
  ]
}
'''