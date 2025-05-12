


First version:

crypto:
btc, eth, xrp, sol

exchanges:
all

Contracts:
perpetual
spot 

time periods (all 4h interval)
1 day 
1 week
1 month
6 months


- Need to have a single source dataset which is added to every day

1. First need to go through each of the api requests to ensure all the data desired is obtained
2. Need to then get 4h interval data for the past 6 months for each of the datasets and save them
3. Then need to write a new folder which gets incremental data

Key initial Data:




Steps to do:

- decide on the correct parameters for each api call (have all with 6 months behind, 4h intervals, and with btc, eth, sol, xrp)
- correct all python files and corresponding data folder
- run the report.py file, ensure all the correct data is received

files and changes:

options:
options max pain - btc, eth, sol, xrp
options info - btc, eth, sol, xrp

spot:
pairs market - btc, eth, sol, xrp
price ohlc history - btc, eth, sol, xrp [also need to edit exchange and symbol] [BIG EDIT]
coin orderbook
Coin Taker Buy/Sell History

futures
Pairs Markets - btc, eth, sol, xrp


notes - need to update files in the report.py file


Perfect now I need to do this again in the same manner but for files in the 'futures' folder in the 'coinglass-api' folder. I need to change some of the files in the 'coinglass-api' folder. Some of these api requests only allow for one crypto symbol at a time, and I need data for these four  crypto symbols: 'ETH', 'BTC', 'XRP', 'SOL'. Please rename the following files in the options and spot folders of the coinglass-api folder, so that there are four different
   variations of the file which get data for each of these symbols. 
     The files to change are:
 futures Coin Taker Buy/Sell Volume History
 futures Coin Aggregated Orderbook Bid&Ask(±range)
 futures Coin Liquidation History
 futures Liquidation Exchange List
 futures Exchange Taker Buy/Sell Ratio
 futures funding rate OI Weight History (OHLC)
 futures funding rate Vol Weight History (OHLC)
 futures open interest Aggregated History (OHLC)
 futures open interest Aggregated Stablecoin Margin History (OHLC)
 futures open interest Aggregated Coin Margin History (OHLC)
 futures open interest Exchange List
 futures open interest Exchange History Chart
futures market Pairs Markets .

 All of these files should keep the same format of their current name but have an extension of the crypto symbol, e.g. 
  api_futures_liquidation_exchange_list_BTC.py . Please create a python file called 'change-files.py' which implements these changes. Ensure all changes are accurate and correct. 
  Ultrathink.

  new files in futures folder:
 futures Coin Taker Buy/Sell Volume History
 futures Coin Aggregated Orderbook Bid&Ask(±range)
 futures Coin Liquidation History
 futures Liquidation Exchange List
 futures Exchange Taker Buy/Sell Ratio
 futures funding rate OI Weight History (OHLC)
 futures funding rate Vol Weight History (OHLC)
 futures open interest Aggregated History (OHLC)
 futures open interest Aggregated Stablecoin Margin History (OHLC)
 futures open interest Aggregated Coin Margin History (OHLC)
 futures open interest Exchange List
 futures open interest Exchange History Chart
futures market Pairs Markets



 # Individual pairs are left out for now