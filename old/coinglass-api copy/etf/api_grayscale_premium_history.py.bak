
import requests

url = "https://open-api-v4.coinglass.com/api/grayscale/premium-history?symbol=BTC"

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
      "primary_market_price": [     // Primary market price list
        0.14,
        0.14
        // ...
      ],
      "date_list": [                // Date list (timestamps)
        1380171600000,
        1380258000000
        // ...
      ],
      "secondary_market_price_list": [  // Secondary market price list
        0.57,
        0.53
        // ...
      ],
      "premium_rate_list": [          // Premium rate list
        19.37,
        15.59
        // ...
      ]
    },
      ....
  ]
}
Query Params
symbol
string
required
Defaults to BTC
Supported values: ETC, LTC, BCH, SOL, XLM, LINK, ZEC, MANA, ZEN, FIL, BAT, LPT

'''