

import requests
import streamlit as st



url = "https://open-api-v4.coinglass.com/api/futures/coins-price-change"

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
    {
      "symbol": "BTC", // Symbol
      "current_price": 84518.4, // Current price

      "price_change_percent_5m": -0.04, // 5m change (%)
      "price_change_percent_15m": -0.09, // 15m change (%)
      "price_change_percent_30m": -0.11, // 30m change (%)
      "price_change_percent_1h": -0.17, // 1h change (%)
      "price_change_percent_4h": -0.54, // 4h change (%)
      "price_change_percent_12h": -0.6, // 12h change (%)
      "price_change_percent_24h": 0.24, // 24h change (%)

      "price_amplitude_percent_5m": 0.07, // 5m amplitude (%)
      "price_amplitude_percent_15m": 0.16, // 15m amplitude (%)
      "price_amplitude_percent_30m": 0.18, // 30m amplitude (%)
      "price_amplitude_percent_1h": 0.26, // 1h amplitude (%)
      "price_amplitude_percent_4h": 0.63, // 4h amplitude (%)
      "price_amplitude_percent_12h": 1.17, // 12h amplitude (%)
      "price_amplitude_percent_24h": 2.06 // 24h amplitude (%)
    },
    {
      "symbol": "ETH", // Symbol
      "current_price": 1573.45, // Current price

      "price_change_percent_5m": -0.04, // 5m change (%)
      "price_change_percent_15m": -0.34, // 15m change (%)
      "price_change_percent_30m": -0.38, // 30m change (%)
      "price_change_percent_1h": -0.54, // 1h change (%)
      "price_change_percent_4h": -0.77, // 4h change (%)
      "price_change_percent_12h": -1.99, // 12h change (%)
      "price_change_percent_24h": -1.41, // 24h change (%)

      "price_amplitude_percent_5m": 0.13, // 5m amplitude (%)
      "price_amplitude_pe

'''