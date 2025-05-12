
import requests

url = "https://open-api-v4.coinglass.com/api/exchange/assets?exchange=Binance"

headers = {
    "accept": "application/json",
    "CG-API-KEY": "a5b89c9d85dc40ffb8144fbecf0fb18f"
}

response = requests.get(url, headers=headers)

print(response.text)

'''

Respones Data
JSON

{
  "code": "0",
  "msg": "success",
  "data": [
    {
      "wallet_address": "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
      "balance": 248597.54,
      "balance_usd": 21757721869.92,
      "symbol": "BTC",
      "assets_name": "Bitcoin",
      "price": 87521.87117346626
    },
    {
      "wallet_address": "3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6",
      "balance": 139456.08,
      "balance_usd": 12205457068.12,
      "symbol": "BTC",
      "assets_name": "Bitcoin",
      "price": 87521.87117346626
    },
Query Params
exchange
string
required
Defaults to Binance
Exchange name (e.g., Binance). Retrieve supported exchanges via the 'support-exchange-pair' API.

Binance
per_page
string
Defaults to 10
Number of results per page.

page
string
Defaults to 1
Page number for pagination, default: 1

'''