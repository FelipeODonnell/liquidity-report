

import requests

url = "https://open-api-v4.coinglass.com/api/index/bitcoin/rainbow-chart"

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
    [
      0.07,                                 // Current price
      0.033065,                             // Minimum value of shadow
      0.044064,                             // First layer
      0.059892,                             // Second layer
      0.082219,                             // Third layer
      0.110996,                             // Fourth layer
      0.149845,                             // Fifth layer
      0.205865,                             // Sixth layer
      0.283454,                             // Seventh layer
      0.380525,                             // Eighth layer
      0.517626,                             // Ninth layer
      1282003200000                        // Timestamp
    ]
  ]
}

'''