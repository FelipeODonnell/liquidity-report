
import requests

url = "https://open-api-v4.coinglass.com/api/index/200-week-moving-average-heatmap"

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
      "timestamp": 1325203200000,          // Timestamp (in milliseconds)
      "price": 4.31063509000584,           // Current price
      "moving_average_1440": 4.143619070636635, // 200-week moving average (1440 represents the period)
      "moving_average_1440_ip": 0,         // Position of the moving average (IP indicator)
    }
  ]
}

'''