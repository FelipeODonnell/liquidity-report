import requests
import streamlit as st



url = "https://open-api-v4.coinglass.com/api/futures/supported-coins"

headers = {
    "accept": "application/json",
    "CG-API-KEY": st.secrets["coinglass_api"]["api_key"]
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
      "SOL",
      "XRP",
      ...
  ]
}

'''