
import requests

url = "https://open-api-v4.coinglass.com/api/index/ahr999"

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
      "date_string": "2011/02/01",            // Date in string format (YYYY/MM/DD)
      "average_price": 0.1365,                // Average price on the given date
      "ahr999_value": 4.441692296429609,      // AHR999 index value
      "current_value": 0.626                  // Current value on the given date
    },
    {
      "date_string": "2011/02/02",            // Date in string format (YYYY/MM/DD)
      "average_price": 0.1383,                // Average price on the given date
      "ahr999_value": 5.642181244439729,      // AHR999 index value
      "current_value": 0.713                  // Current value on the given date
    }
    // More data entries...
  ]
}

'''