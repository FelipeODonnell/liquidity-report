
import requests

url = "https://open-api-v4.coinglass.com/api/spot/supported-coins"

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
    "BTC",
    "ETH",
    "USDT",
    "BNB",
    "SOL",
    "USDC",
    ...
  ]
}
'''