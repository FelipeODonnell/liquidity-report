
import requests

url = "https://open-api-v4.coinglass.com/api/index/pi-cycle-indicator"

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
      "ma_110": 0.07,                     // 110-day moving average price
      "timestamp": 1282003200000,        // Timestamp (milliseconds)
      "ma_350_mu_2": 0.14,               // 2x value of 350-day moving average
      "price": 0.07                      // Daily price
    },
    {
      "ma_110": 0.069,                   // 110-day moving average price
      "timestamp": 1282089600000,        // Timestamp (milliseconds)
      "ma_350_mu_2": 0.138,              // 2x value of 350-day moving average
      "price": 0.068                     // Daily price
    }
  ]
}

'''