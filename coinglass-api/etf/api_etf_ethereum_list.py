
import requests

url = "https://open-api-v4.coinglass.com/api/etf/ethereum/list"

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
      "ticker": "ETHA",                                  // ETF ticker
      "name": "iShares Ethereum Trust ETF",              // ETF name
      "region": "us",                                    // Region
      "market_status": "closed",                         // Market status
      "primary_exchange": "XNAS",                        // Primary exchange
      "cik_code": "0002000638",                          // CIK code
      "type": "Spot",                                    // Type
      "market_cap": "544896000.00",                      // Market capitalization
      "list_date": 1721692800000,                        // Listing date
      "shares_outstanding": "28800000",                  // Shares outstanding
      "aum": "",                                         // Assets under management
      "management_fee_percent": "0.25",                  // Management fee percentage
      "last_trade_time": 1722988779939,                  // Last trade time
      "last_quote_time": 1722988799379,                  // Last quote time
      "volume_quantity": 5592645,                        // Volume quantity
      "volume_usd": 106447049.343,                       // Volume in USD
      "price": 18.92,                                    // Market price
      "price_change": 0.67,                              // Price change
      "price_change_percent": 3.67,                      // Price change percentage
      "asset_info": {
        "nav": 18.11,                                  // Net asset value
        "premium_discount": 0.77,                      // Premium/discount
        "holding_quantity": 237882.8821,                 // Holding quantity
        "change_percent_1d": 0,                        // 1-day change percentage
        "change_quantity_1d": 0,                         // 1-day change quantity
        "change_percent_7d": 56.69,                    // 7-day change percentage
        "change_quantity_7d": 86060.9115,                // 7-day change quantity
        "date": "2024-08-05"                           // Data date
      },
      "update_time": 1722995656637                       // Update time
    }
  ]
}

'''