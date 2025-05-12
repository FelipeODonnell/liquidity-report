#!/usr/bin/env python3
"""
Script to modify API URLs in coinglass-api files to include time ranges.

This script:
1. Calculates current timestamp and timestamp from 6 months ago
2. Finds all Python files in the coinglass-api directory
3. Identifies URLs with both 'limit' and 'interval' parameters
4. Modifies these URLs to:
   - Change interval to '4h'
   - Add start_time (6 months ago)
   - Add end_time (current time)
   - Add unit='usd'
"""

import os
import re
import time
from datetime import datetime, timedelta
import calendar
import shutil

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
    """Format millisecond timestamp as human-readable date string."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def modify_api_urls(file_path, current_ts, past_ts):
    """
    Modify API URLs in a file to add time parameters.
    
    Args:
        file_path: Path to the file to modify
        current_ts: Current timestamp in milliseconds
        past_ts: Timestamp from 6 months ago in milliseconds
        
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Regular expression to find URL assignments
        # Specifically looking for URLs with both 'limit' and 'interval' parameters
        url_pattern = r'(url\s*=\s*")(.*?)(\s*")'
        
        def replace_url(match):
            prefix = match.group(1)  # The 'url = "' part
            url = match.group(2)     # The URL itself
            suffix = match.group(3)  # The closing quote and any whitespace
            
            # Only modify URLs that have both 'limit' and 'interval' parameters
            if 'limit=' in url and 'interval=' in url:
                # Change interval to 4h
                url = re.sub(r'interval=[^&"]+', f'interval=4h', url)
                
                # Add start_time, end_time, and unit parameters if they don't exist
                params_to_add = []
                
                if 'start_time=' not in url:
                    params_to_add.append(f'start_time={past_ts}')
                    
                if 'end_time=' not in url:
                    params_to_add.append(f'end_time={current_ts}')
                    
                if 'unit=' not in url:
                    params_to_add.append('unit=usd')
                
                # Add all the new parameters
                if params_to_add:
                    # Check if URL already has a query string
                    if '?' in url:
                        # Add parameters to existing query string
                        url = url + '&' + '&'.join(params_to_add)
                    else:
                        # Start a new query string
                        url = url + '?' + '&'.join(params_to_add)
                
                return prefix + url + suffix
            
            # Return the match unchanged if it doesn't meet our criteria
            return match.group(0)
        
        # Replace URLs in the content
        new_content = re.sub(url_pattern, replace_url, content)
        
        # Only write the file if changes were made
        if new_content != content:
            # Create a backup of the original file
            backup_path = file_path + '.bak'
            shutil.copy2(file_path, backup_path)
            
            # Write the modified content back to the file
            with open(file_path, 'w') as f:
                f.write(new_content)
                
            return True
            
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    """
    Find all Python files in the coinglass-api directory and modify API URLs.
    """
    # Get timestamps
    current_ts = get_current_timestamp_ms()
    past_ts = get_timestamp_months_ago(6)
    
    print(f"Current timestamp: {current_ts} ({format_date(current_ts)})")
    print(f"6 months ago timestamp: {past_ts} ({format_date(past_ts)})")
    
    # Find and process all Python files in the coinglass-api directory
    api_dir = "coinglass-api"
    if not os.path.exists(api_dir):
        print(f"Error: Directory '{api_dir}' not found.")
        return
    
    modified_files = 0
    total_files = 0
    modified_file_paths = []
    
    # Walk through the directory and process each Python file
    for root, dirs, files in os.walk(api_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                total_files += 1
                
                print(f"Processing: {file_path}")
                if modify_api_urls(file_path, current_ts, past_ts):
                    modified_files += 1
                    modified_file_paths.append(file_path)
    
    print(f"\nSummary:")
    print(f"  - Processed {total_files} Python files in {api_dir}")
    print(f"  - Modified {modified_files} files that had URLs with 'limit' and 'interval' parameters")
    
    if modified_files > 0:
        print("\nModified files:")
        for path in modified_file_paths:
            print(f"  - {path}")
        print("\nBackup files were created with '.bak' extension.")
        print("If you need to restore the original files, you can use the backup files.")
    
    # Display example of modifications made
    print("\nModifications made to URLs:")
    print("  Before: url = \"https://open-api-v4.coinglass.com/api/futures/open-interest/aggregated-history?symbol=BTC&interval=1d&limit=4500\"")
    print(f"  After:  url = \"https://open-api-v4.coinglass.com/api/futures/open-interest/aggregated-history?symbol=BTC&interval=4h&limit=4500&start_time={past_ts}&end_time={current_ts}&unit=usd\"")

if __name__ == "__main__":
    main()