"""
Utility functions for formatting data in the Izun Crypto Liquidity Report application.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from .config import DEFAULT_CURRENCY_PRECISION, DEFAULT_PERCENTAGE_PRECISION

def format_currency(value, precision=DEFAULT_CURRENCY_PRECISION, include_symbol=True, abbreviate=False, show_decimals=True, compact=False, space_after_symbol=False, strip_zeros=True, min_precision=0):  
    """
    Format a value as currency with improved formatting.

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
    compact : bool
        Whether to use a more compact format for UI display
    space_after_symbol : bool
        Whether to add a space after the currency symbol (e.g., "$ 1,000" vs "$1,000")
    strip_zeros : bool
        Whether to strip trailing zeros in decimal portion
    min_precision : int
        Minimum number of decimal places to show (only applies when strip_zeros is True)

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

    # Adjust precision based on configuration
    actual_precision = precision if show_decimals else 0
    
    # For very small numbers, ensure appropriate precision
    if abs(num) > 0 and abs(num) < 0.01 and show_decimals:
        # Increase precision for very small numbers to show at least 2 significant digits
        magnitude = abs(num)
        significant_digits = 0
        while magnitude < 1 and significant_digits < 8:  # Limit to 8 decimal places max
            magnitude *= 10
            significant_digits += 1
        actual_precision = max(actual_precision, significant_digits + 2)  # Add 2 for better readability

    # Handle abbreviation for large numbers
    if abbreviate:
        # More precise thresholds with improved spacing
        if abs(num) >= 1_000_000_000:
            formatted = f"{num/1_000_000_000:.{actual_precision}f}B"
        elif abs(num) >= 1_000_000:
            formatted = f"{num/1_000_000:.{actual_precision}f}M"
        elif abs(num) >= 1_000:
            formatted = f"{num/1_000:.{actual_precision}f}K"
        else:
            formatted = f"{num:.{actual_precision}f}"
        
        # Remove trailing zeros and decimal point if requested
        if strip_zeros and show_decimals and '.' in formatted:
            parts = formatted.split('.')
            decimal_part = parts[1].rstrip('0')
            
            # Ensure we maintain minimum precision
            if min_precision > 0 and len(decimal_part) < min_precision:
                decimal_part = decimal_part.ljust(min_precision, '0')
                
            if decimal_part:
                formatted = f"{parts[0]}.{decimal_part}"
            else:
                formatted = parts[0]
    else:
        # Standard formatting with thousands separator
        if compact:
            # For compact mode, dynamically adjust precision based on value magnitude
            if abs(num) < 1:
                # Very small numbers might need more decimals
                formatted = f"{num:,.{max(actual_precision, 4)}f}"
            elif abs(num) < 10:
                # Small numbers might need decimals
                formatted = f"{num:,.{max(actual_precision, 2)}f}"
            elif abs(num) < 1000:
                # Medium numbers need fewer decimals
                formatted = f"{num:,.{min(actual_precision, 1)}f}"
            else:
                # Larger numbers don't need decimals
                formatted = f"{num:,.0f}"
        else:
            formatted = f"{num:,.{actual_precision}f}"
        
        # Remove trailing zeros if requested
        if strip_zeros and show_decimals and '.' in formatted:
            parts = formatted.split('.')
            decimal_part = parts[1].rstrip('0')
            
            # Ensure we maintain minimum precision
            if min_precision > 0 and len(decimal_part) < min_precision:
                decimal_part = decimal_part.ljust(min_precision, '0')
                
            if decimal_part:
                formatted = f"{parts[0]}.{decimal_part}"
            else:
                formatted = parts[0]

    # Add currency symbol if requested with proper spacing
    if include_symbol:
        if space_after_symbol:
            return f"$ {formatted}"
        else:
            return f"${formatted}"
    else:
        return formatted

def format_percentage(value, precision=DEFAULT_PERCENTAGE_PRECISION, include_symbol=True, strip_zeros=True, show_plus=False, space_before_symbol=False, add_space_after=False, min_precision=0, thousands_separator=False, auto_precision=False):  
    """
    Format a value as a percentage with improved clarity.
    
    Parameters:
    -----------
    value : float
        The value to format (e.g., 0.05 for 5%)
    precision : int
        The number of decimal places
    include_symbol : bool
        Whether to include the % symbol
    strip_zeros : bool
        Whether to strip trailing zeros
    show_plus : bool
        Whether to show the plus sign for positive values
    space_before_symbol : bool
        Whether to add a space before the % symbol (e.g., "5 %" vs "5%")
    add_space_after : bool
        Whether to add a space after the formatted percentage (for consistent column alignment)
    min_precision : int
        Minimum number of decimal places to show (only applies when strip_zeros is True)
    thousands_separator : bool
        Whether to include thousands separator for large percentages
    auto_precision : bool
        Whether to automatically adjust precision based on the value magnitude
        
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
    
    # Adjust precision for very small values when auto_precision is enabled
    actual_precision = precision
    if auto_precision:
        if abs(num) > 0 and abs(num) < 0.1:
            # For very small percentages, add more decimal places
            actual_precision = max(precision, 4)
        elif abs(num) < 1:
            # For small percentages, ensure at least 2 decimal places
            actual_precision = max(precision, 2)
        elif abs(num) >= 1000 and abs(num) < 10000:
            # For large percentages, reduce decimal places
            actual_precision = min(precision, 1)
        elif abs(num) >= 10000:
            # For very large percentages, no decimal places
            actual_precision = 0
    
    # Format with specified precision and thousands separator if requested
    if thousands_separator and abs(num) >= 1000:
        formatted = f"{num:,.{actual_precision}f}"
    else:
        formatted = f"{num:.{actual_precision}f}"
    
    # Remove trailing zeros if requested, but maintain minimum precision
    if strip_zeros and '.' in formatted:
        parts = formatted.split('.')
        decimal_part = parts[1].rstrip('0')
        
        # Ensure we maintain minimum precision
        if min_precision > 0 and len(decimal_part) < min_precision:
            decimal_part = decimal_part.ljust(min_precision, '0')
            
        if decimal_part:
            formatted = f"{parts[0]}.{decimal_part}"
        else:
            formatted = parts[0]
    
    # Add plus sign for positive values if requested
    if show_plus and num > 0 and not formatted.startswith('+'):
        formatted = f"+{formatted}"
    
    # Add percentage symbol if requested
    if include_symbol:
        if space_before_symbol:
            formatted = f"{formatted} %"
        else:
            formatted = f"{formatted}%"
    
    # Add space after for consistent column alignment if requested
    if add_space_after:
        formatted = f"{formatted} "
    
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


def format_column_name(column_name):
    """
    Format a column name by removing underscores and capitalizing first letter of each word.
    
    Parameters:
    -----------
    column_name : str
        The column name to format
        
    Returns:
    --------
    str
        Formatted column name
    """
    if not isinstance(column_name, str):
        return str(column_name)
    
    # Define common acronyms and terms to preserve capitalization
    special_terms = {
        'usd': 'USD',
        'btc': 'BTC',
        'eth': 'ETH',
        'aum': 'AUM',
        'api': 'API',
        'url': 'URL',
        '24h': '24h',
        '7d': '7d',
        '30d': '30d',
        'id': 'ID',
        'roi': 'ROI',
        'ip': 'IP',
        'p2p': 'P2P',
        'defi': 'DeFi',
        'nft': 'NFT',
        'tvl': 'TVL',
        'cex': 'CEX',
        'dex': 'DEX',
        'pct': '%',
        'percent': '%'
    }
    
    # Replace underscores with spaces
    formatted = column_name.replace('_', ' ')
    
    # Title case the string (capitalize first letter of each word)
    formatted = formatted.title()
    
    # Handle special cases
    words = formatted.split()
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in special_terms:
            words[i] = special_terms[word_lower]
            
    # Join words back together
    formatted = ' '.join(words)
    
    # Fix case for common prefixes
    formatted = re.sub(r'\bMc([A-Z])', lambda m: 'Mc' + m.group(1), formatted)
    
    return formatted