"""
Utility functions for formatting data in the Izun Crypto Liquidity Report application.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from .config import DEFAULT_CURRENCY_PRECISION, DEFAULT_PERCENTAGE_PRECISION

def format_currency(value, precision=DEFAULT_CURRENCY_PRECISION, include_symbol=True, abbreviate=False, show_decimals=True):
    """
    Format a value as currency.

    Parameters:
    -----------
    value : float or int
        The value to format
    precision : int
        The number of decimal places
    include_symbol : bool
        Whether to include the currency symbol
    abbreviate : bool
        Whether to abbreviate large numbers (e.g., 1M instead of 1,000,000)
    show_decimals : bool
        Whether to show decimal places (if False, rounds to nearest integer)

    Returns:
    --------
    str
        The formatted currency string
    """
    if value is None or pd.isna(value):
        return "N/A"

    try:
        num = float(value)
    except (ValueError, TypeError):
        return str(value)

    # Adjust precision if we don't want to show decimals
    actual_precision = precision if show_decimals else 0

    # Handle abbreviation for large numbers
    if abbreviate:
        if abs(num) >= 1_000_000_000:
            formatted = f"{num/1_000_000_000:.{actual_precision}f}B"
        elif abs(num) >= 1_000_000:
            formatted = f"{num/1_000_000:.{actual_precision}f}M"
        elif abs(num) >= 1_000:
            formatted = f"{num/1_000:.{actual_precision}f}K"
        else:
            formatted = f"{num:.{actual_precision}f}"
    else:
        formatted = f"{num:,.{actual_precision}f}"

    # Add currency symbol if requested
    if include_symbol:
        return f"${formatted}"
    else:
        return formatted

def format_percentage(value, precision=DEFAULT_PERCENTAGE_PRECISION, include_symbol=True):
    """
    Format a value as a percentage.
    
    Parameters:
    -----------
    value : float
        The value to format (e.g., 0.05 for 5%)
    precision : int
        The number of decimal places
    include_symbol : bool
        Whether to include the % symbol
        
    Returns:
    --------
    str
        The formatted percentage string
    """
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        num = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    # Determine if the number is already in percentage form
    if abs(num) <= 1 and not (abs(num) == 1 and num != 0):
        # Value is likely a decimal (e.g., 0.05), convert to percentage
        num *= 100
    
    formatted = f"{num:.{precision}f}"
    
    if include_symbol:
        return f"{formatted}%"
    else:
        return formatted

def format_timestamp(timestamp, format_string="%Y-%m-%d %H:%M:%S"):
    """
    Format a timestamp (in milliseconds since epoch) as a readable date string.
    
    Parameters:
    -----------
    timestamp : int
        Timestamp in milliseconds since epoch
    format_string : str
        Format string for datetime.strftime
        
    Returns:
    --------
    str
        The formatted date string
    """
    if timestamp is None or pd.isna(timestamp):
        return "N/A"
    
    try:
        # Convert timestamp to datetime
        if isinstance(timestamp, (int, float)) and timestamp > 1e10:
            # Timestamp is likely in milliseconds
            dt = datetime.fromtimestamp(timestamp / 1000)
        elif isinstance(timestamp, (int, float)):
            # Timestamp is likely in seconds
            dt = datetime.fromtimestamp(timestamp)
        else:
            # Try to parse directly
            dt = pd.to_datetime(timestamp)
            
        return dt.strftime(format_string)
    except:
        return str(timestamp)

def format_volume(value, precision=DEFAULT_CURRENCY_PRECISION):
    """Format a volume value with appropriate abbreviation"""
    return format_currency(value, precision=precision, abbreviate=True)

def format_delta(value, is_percentage=True, precision=DEFAULT_PERCENTAGE_PRECISION):
    """
    Format a delta value with appropriate coloring.
    
    Parameters:
    -----------
    value : float
        The delta value
    is_percentage : bool
        Whether the value is a percentage
    precision : int
        The number of decimal places
        
    Returns:
    --------
    str
        The formatted delta string with prefix
    """
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        num = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    # Format based on type
    if is_percentage:
        formatted = format_percentage(num, precision)
    else:
        formatted = format_currency(num, precision)
    
    # Add +/- prefix
    if num > 0:
        return f"+{formatted}"
    else:
        return formatted  # Negative sign is already included

def humanize_time_diff(timestamp):
    """
    Convert a timestamp to a human-readable relative time (e.g., "2 hours ago").
    
    Parameters:
    -----------
    timestamp : int or datetime
        The timestamp to convert
        
    Returns:
    --------
    str
        Human-readable relative time
    """
    if timestamp is None or pd.isna(timestamp):
        return "N/A"
    
    try:
        # Convert timestamp to datetime
        if isinstance(timestamp, (int, float)) and timestamp > 1e10:
            # Timestamp is likely in milliseconds
            dt = datetime.fromtimestamp(timestamp / 1000)
        elif isinstance(timestamp, (int, float)):
            # Timestamp is likely in seconds
            dt = datetime.fromtimestamp(timestamp)
        else:
            # Try to parse directly
            dt = pd.to_datetime(timestamp)
            
        # Calculate time difference
        now = datetime.now()
        diff = now - dt
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)} seconds ago"
        elif seconds < 3600:
            return f"{int(seconds // 60)} minutes ago"
        elif seconds < 86400:
            return f"{int(seconds // 3600)} hours ago"
        elif seconds < 604800:
            return f"{int(seconds // 86400)} days ago"
        elif seconds < 2592000:
            return f"{int(seconds // 604800)} weeks ago"
        elif seconds < 31536000:
            return f"{int(seconds // 2592000)} months ago"
        else:
            return f"{int(seconds // 31536000)} years ago"
    except:
        return str(timestamp)