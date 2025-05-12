
import requests

url = "https://open-api-v4.coinglass.com/api/bull-market-peak-indicator"

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
      "indicator_name": "Bitcoin Ahr999 Index",  // Indicator name
      "current_value": "0.78",                  // Current value
      "target_value": "4",                      // Target value
      "previous_value": "0.77",                 // Previous value
      "change_value": "0.0009811160359081",     // Change value
      "comparison_type": ">=",                  // Comparison type
      "hit_status": false                       // Hit status (whether the target condition is met)
    },
    {
      "indicator_name": "Pi Cycle Top Indicator",  // Indicator name
      "current_value": "85073.0",                  // Current value
      "target_value": "154582",                    // Target value
      "previous_value": "85127.0",                 // Previous value
      "change_value": "-54.0",                     // Change value
      "comparison_type": ">=",                     // Comparison type
      "hit_status": false                          // Hit status (whether the target condition is met)
    }
  ]
}

'''