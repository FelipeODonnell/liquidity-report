

import requests

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/aum"

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
  "data": [
    {
      "time": 1704153600000,
      "aum_usd": 0
    },
    {
      "time": 1704240000000,
      "aum_usd": 0
    },
    ....
  ]
}

'''