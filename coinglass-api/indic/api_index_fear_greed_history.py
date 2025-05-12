
import requests

url = "https://open-api-v4.coinglass.com/api/index/fear-greed-history"

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
      "data_list": [4611285.1141, ...],         // Fear and Greed Index values
      "price_list": [4788636.51145, ...],         // Corresponding price data
      "time_list": [1636588800, ...]        // Timestamps
    }
  ]
}

'''