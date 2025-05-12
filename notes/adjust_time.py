#!/usr/bin/env python3
"""
Script to generate current timestamp and timestamp from 6 months ago in milliseconds.
Useful for setting start_time parameters in API requests.
"""

import time
from datetime import datetime, timedelta
import calendar

def get_current_timestamp_ms():
    """Get current time as timestamp in milliseconds."""
    return int(time.time() * 1000)

def get_timestamp_months_ago(months=6):
    """
    Get timestamp from specified number of months ago in milliseconds.
    
    Args:
        months: Number of months to go back in time
        
    Returns:
        Timestamp in milliseconds
    """
    # Get current date
    current_date = datetime.now()
    
    # Calculate date from months ago
    # Handle year boundary correctly
    year = current_date.year
    month = current_date.month - months
    
    # Adjust year if month goes negative or beyond 12
    while month <= 0:
        year -= 1
        month += 12
    
    # Create date object for 6 months ago
    # Use min to ensure valid day for the month
    last_day_of_month = calendar.monthrange(year, month)[1]
    past_day = min(current_date.day, last_day_of_month)
    past_date = datetime(year, month, past_day,
                         current_date.hour, current_date.minute,
                         current_date.second, current_date.microsecond)
    
    # Convert to timestamp in milliseconds
    timestamp_ms = int(past_date.timestamp() * 1000)
    
    return timestamp_ms

def format_date(timestamp_ms):
    """
    Format millisecond timestamp as human-readable date string.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        
    Returns:
        Formatted date string
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    # Get current timestamp in milliseconds
    current_ts = get_current_timestamp_ms()
    
    # Get timestamp from 6 months ago in milliseconds
    past_ts = get_timestamp_months_ago(6)
    
    # Print both timestamps and their human-readable equivalents
    print("\nTimestamp Values (milliseconds):")
    print(f"Current timestamp:        {current_ts}")
    print(f"Timestamp 6 months ago:   {past_ts}")
    
    print("\nHuman-readable dates:")
    print(f"Current date:             {format_date(current_ts)}")
    print(f"Date 6 months ago:        {format_date(past_ts)}")
    
    print("\nUsage examples for API requests:")
    print(f"url = \"https://open-api-v4.coinglass.com/api/futures/open-interest/aggregated-history?symbol=BTC&interval=1d&start_time={past_ts}&end_time={current_ts}\"")
    print(f"url = \"https://open-api-v4.coinglass.com/api/futures/liquidation/aggregated-history?exchange_list=Binance&symbol=ETH&interval=1d&start_time={past_ts}\"")
    print()