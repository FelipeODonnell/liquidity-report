"""
Utility functions for loading data in the Izun Crypto Liquidity Report application.
"""

import os
import glob
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import logging
from .config import DATA_BASE_PATH, SUPPORTED_ASSETS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def list_data_directories():
    """
    List all date directories in the data folder.

    Returns:
    --------
    list
        List of directory names sorted in descending order (newest first)
    """
    try:
        # Log the data path being used
        logger.info(f"Accessing data directory at: {DATA_BASE_PATH}")
        if not os.path.exists(DATA_BASE_PATH):
            logger.error(f"Data directory does not exist: {DATA_BASE_PATH}")
            st.error(f"Data directory not found at: {DATA_BASE_PATH}")
            return []

        dirs = [d for d in os.listdir(DATA_BASE_PATH)
                if os.path.isdir(os.path.join(DATA_BASE_PATH, d)) and d[0].isdigit()]

        # Sort directories in descending order (newest first)
        sorted_dirs = sorted(dirs, reverse=True)
        logger.info(f"Found {len(sorted_dirs)} data directories")
        return sorted_dirs
    except Exception as e:
        logger.error(f"Error listing data directories: {e}")
        st.error(f"Error accessing data directory: {e}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_latest_data_directory():
    """
    Get the most recent data directory.
    
    Returns:
    --------
    str
        The name of the most recent data directory, or None if no directories exist
    """
    dirs = list_data_directories()
    return dirs[0] if dirs else None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_data_last_updated():
    """
    Get the last update timestamp from the latest data directory.
    
    Returns:
    --------
    datetime
        The last update datetime
    """
    latest_dir = get_latest_data_directory()
    if not latest_dir:
        return None
    
    try:
        # Try to parse the directory name as a date (format: YYYYMMDD)
        return datetime.strptime(latest_dir, "%Y%m%d")
    except:
        # If parsing fails, use the directory creation time
        dir_path = os.path.join(DATA_BASE_PATH, latest_dir)
        timestamp = os.path.getctime(dir_path)
        return datetime.fromtimestamp(timestamp)

@st.cache_data
def load_parquet_file(file_path):
    """
    Load a parquet file into a pandas DataFrame.

    Parameters:
    -----------
    file_path : str
        The path to the parquet file

    Returns:
    --------
    pandas.DataFrame
        The loaded DataFrame, or an empty DataFrame if loading fails
    """
    try:
        # First check if file exists to provide better error messages
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist")
            return pd.DataFrame()

        # Check if file has valid size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"File {file_path} is empty (0 bytes)")
            return pd.DataFrame()

        # Try loading the file
        df = pd.read_parquet(file_path)
        file_name = os.path.basename(file_path)

        # Check if the DataFrame is completely empty or has no rows
        if df.empty or len(df) == 0:
            logger.warning(f"File {file_name} was loaded but contains no data (empty DataFrame)")
            return pd.DataFrame()

        # Check if DataFrame has expected structure
        if len(df.columns) == 0:
            logger.warning(f"File {file_name} has no columns")
            return pd.DataFrame()

        logger.info(f"Successfully loaded {file_name} with shape {df.shape}")
        logger.debug(f"Columns in {file_name}: {list(df.columns)}")

        return df
    except pd.errors.EmptyDataError:
        file_name = os.path.basename(file_path)
        logger.warning(f"File {file_name} is empty or corrupted")
        return pd.DataFrame()
    except Exception as e:
        file_name = os.path.basename(file_path)
        logger.error(f"Error loading {file_name}: {e}")
        # Only show error to user for serious/unexpected errors
        # For common issues just log warning and return empty DataFrame
        if not isinstance(e, (pd.errors.EmptyDataError, pd.errors.ParserError)):
            st.error(f"Error loading {file_name}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_data_for_category(category, subcategory=None, asset=None, data_dir=None):
    """
    Load data for a specific category and optional subcategory.
    
    Parameters:
    -----------
    category : str
        Main data category (e.g., 'etf', 'futures')
    subcategory : str, optional
        Subcategory (e.g., 'funding_rate', 'liquidation')
    asset : str, optional
        Asset filter (e.g., 'BTC', 'ETH')
    data_dir : str, optional
        Specific data directory to use. If None, the latest directory is used.
        
    Returns:
    --------
    dict
        Dictionary of DataFrames with file names as keys
    """
    data_dict = {}
    
    # Determine base path
    if data_dir is None:
        data_dir = get_latest_data_directory()
        if not data_dir:
            return data_dict
    
    base_path = os.path.join(DATA_BASE_PATH, data_dir)
    
    # Build search path
    if subcategory:
        search_path = os.path.join(base_path, category, subcategory)
    else:
        search_path = os.path.join(base_path, category)
    
    # Check if directory exists
    if not os.path.isdir(search_path):
        return data_dict
    
    # Add asset filter if provided, with multiple patterns for better matching
    if asset:
        # Create multiple search patterns to catch various naming conventions
        search_patterns = [
            f"*_{asset}_*.parquet",  # e.g. api_futures_liquidation_exchange_list_BTC_BTC.parquet
            f"*_{asset}.parquet",     # e.g. api_futures_liquidation_BTC.parquet
            f"*{asset}*.parquet",     # Fallback: any file with asset name
            # Even more lenient patterns for specific categories
            "*_*liquidation*_*.parquet" if (subcategory == 'liquidation' and category == 'futures') else None,
            "*_*funding*_*.parquet" if (subcategory == 'funding_rate' and category == 'futures') else None,
            "*_*openInterest*_*.parquet" if (subcategory == 'open_interest' and category == 'futures') else None,
            "*_*orderbook*_*.parquet" if ('order_book' in subcategory if subcategory else False) else None
        ]
        # Filter out None values
        search_patterns = [p for p in search_patterns if p]

        # Log the patterns we're searching for
        logger.info(f"Searching for asset {asset} with patterns: {search_patterns}")

        # Find files matching any of the patterns
        matching_files = []
        for pattern in search_patterns:
            matching_files.extend(glob.glob(os.path.join(search_path, pattern)))

        # If we found matching files, use them; otherwise fall back to all files
        if matching_files:
            file_paths = sorted(list(set(matching_files)))  # Remove duplicates
            logger.info(f"Found {len(file_paths)} files matching asset {asset}")
        else:
            # Try a super lenient approach - just get all parquet files
            logger.warning(f"No files found for asset {asset} in {search_path}, using all files as fallback")
            file_paths = glob.glob(os.path.join(search_path, "*.parquet"))

            # If we actually have files, log them for debugging
            if file_paths:
                files_found = [os.path.basename(f) for f in file_paths]
                logger.info(f"Found {len(file_paths)} files in directory using lenient search: {files_found}")
    else:
        # If no asset filter, get all parquet files
        file_paths = glob.glob(os.path.join(search_path, "*.parquet"))

        # Log files found for debugging
        if file_paths:
            files_found = [os.path.basename(f) for f in file_paths]
            logger.info(f"Found {len(file_paths)} files with no asset filter: {files_found}")
    
    # Load each file that we found
    for file_path in file_paths:
        file_name = os.path.basename(file_path).replace('.parquet', '')
        try:
            df = load_parquet_file(file_path)
            if not df.empty:
                data_dict[file_name] = df
                logger.info(f"Successfully loaded {file_name}")
            else:
                logger.warning(f"File {file_name} was empty or failed to load")
        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}")
    
    return data_dict

@st.cache_data
def load_specific_data_file(category, file_name, data_dir=None, subcategory=None):
    """
    Load a specific data file.
    
    Parameters:
    -----------
    category : str
        Data category (e.g., 'etf', 'futures')
    file_name : str
        File name without extension
    data_dir : str, optional
        Specific data directory to use. If None, the latest directory is used.
    subcategory : str, optional
        Subcategory (e.g., 'funding_rate', 'liquidation')
        
    Returns:
    --------
    pandas.DataFrame
        The loaded DataFrame, or an empty DataFrame if loading fails
    """
    # Determine base path
    if data_dir is None:
        data_dir = get_latest_data_directory()
        if not data_dir:
            return pd.DataFrame()
    
    base_path = os.path.join(DATA_BASE_PATH, data_dir)
    
    # Build file path
    if subcategory:
        file_path = os.path.join(base_path, category, subcategory, f"{file_name}.parquet")
    else:
        file_path = os.path.join(base_path, category, f"{file_name}.parquet")
    
    # Check if file exists
    if not os.path.isfile(file_path):
        return pd.DataFrame()
    
    # Load and return the file
    return load_parquet_file(file_path)

def process_timestamps(df, timestamp_col=None, keep_original=True):
    """
    Convert timestamp column to datetime and handle nested list data structures.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to process
    timestamp_col : str or None
        The name of the timestamp column. If None, tries common column names.
    keep_original : bool
        Whether to keep the original timestamp column

    Returns:
    --------
    pandas.DataFrame
        The processed DataFrame with a new 'datetime' column

    Notes:
    ------
    This function also handles various nested data structures that are common in the CoinGlass API responses.
    """
    if df.empty:
        return df

    # If 'datetime' column already exists, return the DataFrame
    if 'datetime' in df.columns:
        return df

    # Special case: check if this is the bull market peak indicator format
    # (which doesn't have time series data but shows indicators at a single point in time)
    if 'indicator_name' in df.columns and 'current_value' in df.columns and 'target_value' in df.columns:
        logger.info("Detected indicator format data without timestamps (like Bull Market Peak Indicator)")
        return df  # Return as-is for special handling by the display functions

    # Special case: check if this is a nested list format (timestamp/list columns)
    # Common in many API responses from CoinGlass
    if 'timestamp' in df.columns and 'list' in df.columns:
        try:
            logger.info("Detected nested list format data structure")
            # Extract the nested data from the 'list' column
            # First convert timestamp to datetime
            result = df.copy()
            result['datetime'] = pd.to_datetime(result['timestamp'], unit='ms')

            # Check the type of the list column values
            try:
                sample = result['list'].iloc[0] if not result.empty else None

                # Handle numpy arrays
                if hasattr(sample, '__array__'):
                    try:
                        # Try to convert to list if it's a numpy array
                        sample = sample.tolist()
                    except:
                        # If conversion fails, it might not be an array after all
                        pass

                # If it's a simple list of values (not dictionaries), create a data_list column
                if isinstance(sample, list) and (len(sample) == 0 or not isinstance(sample[0], dict)):
                    logger.info("Detected simple list of values, creating data_list column")
                    # Convert all list values to regular python lists if needed
                    result['data_list'] = result['list'].apply(
                        lambda x: x.tolist() if hasattr(x, '__array__') else x)
                    return result
            except Exception as e:
                logger.warning(f"Error checking list type: {e}. Will continue with standard processing.")

            # Try to normalize the list column into a dataframe
            # This converts the list of dictionaries into a dataframe with columns for each key
            normalized_data = []
            for idx, row in result.iterrows():
                timestamp = row['datetime']
                items = row['list']
                # Handle numpy arrays
                if hasattr(items, '__array__'):
                    try:
                        # Convert to python list if it's a numpy array
                        items = items.tolist()
                    except Exception as e_convert:
                        logger.debug(f"Could not convert array to list directly: {e_convert}")
                        try:
                            # If direct conversion fails, check if it's a multidimensional array
                            if hasattr(items, 'shape') and len(items.shape) > 1:
                                # If it's 2D or higher, take the first element
                                items = items[0]
                                # Then try to convert again
                                if hasattr(items, 'tolist'):
                                    items = items.tolist()
                            # If it's 1D but conversion failed for some reason
                            elif hasattr(items, 'shape') and len(items) > 0:
                                # Try converting elements individually
                                items = [item for item in items]
                            else:
                                logger.warning(f"Empty array or conversion failed for row {idx}")
                        except Exception as e_shape:
                            logger.warning(f"Could not process array for row {idx}: {e_shape}")

                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            item_dict = item.copy()
                            item_dict['datetime'] = timestamp
                            normalized_data.append(item_dict)
                        else:
                            # If it's not a dict, create a simple data point
                            normalized_data.append({
                                'datetime': timestamp,
                                'data_value': item
                            })
                elif pd.notna(items):  # If it's not a list but not NaN
                    logger.warning(f"Unexpected type in 'list' column: {type(items)}")
                    normalized_data.append({
                        'datetime': timestamp,
                        'data_value': str(items)
                    })

            # Create a new DataFrame from the normalized data
            if normalized_data:
                normalized_df = pd.DataFrame(normalized_data)
                logger.info(f"Successfully normalized nested list data with columns: {list(normalized_df.columns)}")
                return normalized_df
            else:
                logger.warning("Failed to normalize nested list data - empty result")
                return df
        except Exception as e:
            logger.error(f"Error processing nested list format: {e}")
            return df

    # Special case: check for exchange lists with stablecoin_margin_list and token_margin_list
    if 'symbol' in df.columns and ('stablecoin_margin_list' in df.columns or 'token_margin_list' in df.columns):
        try:
            logger.info("Detected exchange list format with margin lists")

            # Create a new normalized dataframe to store the expanded data
            normalized_data = []

            # Process stablecoin margin list if it exists
            if 'stablecoin_margin_list' in df.columns:
                for idx, row in df.iterrows():
                    symbol = row['symbol']
                    try:
                        # Handle different data types of stablecoin_margin_list
                        margin_list = row['stablecoin_margin_list']
                        # Check if it's a numpy array or pandas Series
                        if hasattr(margin_list, '__array__'):
                            # Convert to python list if it's a numpy array or pandas series
                            try:
                                # Try direct conversion to list
                                margin_list = margin_list.tolist()
                            except Exception as e_convert:
                                logger.debug(f"Could not convert margin_list directly to list: {e_convert}")
                                try:
                                    # If direct conversion fails, check if it's a multidimensional array
                                    if hasattr(margin_list, 'shape') and len(margin_list.shape) > 1:
                                        # If it's 2D or higher, take the first element
                                        margin_list = margin_list[0]
                                        # Then try to convert again
                                        if hasattr(margin_list, 'tolist'):
                                            margin_list = margin_list.tolist()
                                    # If it's 1D but conversion failed for some reason
                                    elif hasattr(margin_list, 'shape') and len(margin_list) > 0:
                                        # Try converting elements individually
                                        margin_list = [item for item in margin_list]
                                    else:
                                        margin_list = []
                                except Exception as e_shape:
                                    logger.warning(f"Could not process multidimensional stablecoin_margin_list for {symbol}: {e_shape}")
                                    margin_list = []

                        if isinstance(margin_list, list):
                            for item in margin_list:
                                if isinstance(item, dict):
                                    item_dict = item.copy()
                                    item_dict['symbol'] = symbol
                                    item_dict['margin_type'] = 'stablecoin'
                                    normalized_data.append(item_dict)
                        elif isinstance(margin_list, dict):
                            item_dict = margin_list.copy()
                            item_dict['symbol'] = symbol
                            item_dict['margin_type'] = 'stablecoin'
                            normalized_data.append(item_dict)
                        elif pd.isna(margin_list):
                            # Skip NaN values
                            pass
                        else:
                            logger.warning(f"Unexpected stablecoin_margin_list type: {type(margin_list)}")
                    except Exception as e:
                        logger.error(f"Error processing stablecoin_margin_list for symbol {symbol}: {e}")

            # Process token margin list if it exists
            if 'token_margin_list' in df.columns:
                for idx, row in df.iterrows():
                    symbol = row['symbol']
                    try:
                        # Handle different data types of token_margin_list
                        margin_list = row['token_margin_list']
                        # Check if it's a numpy array or pandas Series
                        if hasattr(margin_list, '__array__'):
                            # Convert to python list if it's a numpy array or pandas series
                            try:
                                # Try direct conversion to list
                                margin_list = margin_list.tolist()
                            except Exception as e_convert:
                                logger.debug(f"Could not convert token_margin_list directly to list: {e_convert}")
                                try:
                                    # If direct conversion fails, check if it's a multidimensional array
                                    if hasattr(margin_list, 'shape') and len(margin_list.shape) > 1:
                                        # If it's 2D or higher, take the first element
                                        margin_list = margin_list[0]
                                        # Then try to convert again
                                        if hasattr(margin_list, 'tolist'):
                                            margin_list = margin_list.tolist()
                                    # If it's 1D but conversion failed for some reason
                                    elif hasattr(margin_list, 'shape') and len(margin_list) > 0:
                                        # Try converting elements individually
                                        margin_list = [item for item in margin_list]
                                    else:
                                        margin_list = []
                                except Exception as e_shape:
                                    logger.warning(f"Could not process multidimensional token_margin_list for {symbol}: {e_shape}")
                                    margin_list = []

                        if isinstance(margin_list, list):
                            for item in margin_list:
                                if isinstance(item, dict):
                                    item_dict = item.copy()
                                    item_dict['symbol'] = symbol
                                    item_dict['margin_type'] = 'token'
                                    normalized_data.append(item_dict)
                        elif isinstance(margin_list, dict):
                            item_dict = margin_list.copy()
                            item_dict['symbol'] = symbol
                            item_dict['margin_type'] = 'token'
                            normalized_data.append(item_dict)
                        elif pd.isna(margin_list):
                            # Skip NaN values
                            pass
                        else:
                            logger.warning(f"Unexpected token_margin_list type: {type(margin_list)}")
                    except Exception as e:
                        logger.error(f"Error processing token_margin_list for symbol {symbol}: {e}")

            # Create a new DataFrame from the normalized data
            if normalized_data:
                normalized_df = pd.DataFrame(normalized_data)
                # Rename 'exchange' to 'exchange_name' for consistency
                if 'exchange' in normalized_df.columns and 'exchange_name' not in normalized_df.columns:
                    normalized_df = normalized_df.rename(columns={'exchange': 'exchange_name'})
                logger.info(f"Successfully normalized exchange list data with columns: {list(normalized_df.columns)}")
                return normalized_df
            else:
                logger.warning("Failed to normalize exchange list data - empty result")
                return df
        except Exception as e:
            logger.error(f"Error processing exchange list format: {e}")
            return df

    # Make a copy to avoid modifying the original
    result = df.copy()

    # Common timestamp column names to try if no specific column is provided
    common_timestamp_cols = ['time', 'timestamp', 'date', 'time_list', 'timeStamp', 'created_at']

    # If no timestamp column is specified, try to find one
    if timestamp_col is None:
        for col in common_timestamp_cols:
            if col in result.columns:
                timestamp_col = col
                logger.info(f"Using detected timestamp column: {timestamp_col}")
                break

    # If we still don't have a timestamp column, return the original DataFrame
    if timestamp_col is None or timestamp_col not in result.columns:
        logger.warning(f"No suitable timestamp column found in DataFrame with columns: {list(result.columns)}")
        return df

    try:
        # Try various timestamp conversion strategies
        if pd.api.types.is_integer_dtype(result[timestamp_col]):
            # For integer timestamps, try milliseconds first (most common)
            if result[timestamp_col].max() > 1e10:
                result['datetime'] = pd.to_datetime(result[timestamp_col], unit='ms')
            # Otherwise try seconds
            else:
                result['datetime'] = pd.to_datetime(result[timestamp_col], unit='s')
        else:
            # For string timestamps, let pandas infer the format
            result['datetime'] = pd.to_datetime(result[timestamp_col], errors='coerce')

            # Check if conversion was successful
            if result['datetime'].isna().all():
                logger.warning(f"Failed to convert {timestamp_col} to datetime")
                return df

        # Drop original timestamp column if not keeping it
        if not keep_original:
            result = result.drop(columns=[timestamp_col])

        return result
    except Exception as e:
        logger.error(f"Error processing timestamps: {e}")
        # If conversion fails, return the original DataFrame
        return df

def get_available_files(category, subcategory=None, data_dir=None):
    """
    Get a list of available data files for a category.
    
    Parameters:
    -----------
    category : str
        Data category (e.g., 'etf', 'futures')
    subcategory : str, optional
        Subcategory (e.g., 'funding_rate', 'liquidation')
    data_dir : str, optional
        Specific data directory to use. If None, the latest directory is used.
        
    Returns:
    --------
    list
        List of available file names without extension
    """
    # Determine base path
    if data_dir is None:
        data_dir = get_latest_data_directory()
        if not data_dir:
            return []
    
    base_path = os.path.join(DATA_BASE_PATH, data_dir)
    
    # Build search path
    if subcategory:
        search_path = os.path.join(base_path, category, subcategory)
    else:
        search_path = os.path.join(base_path, category)
    
    # Check if directory exists
    if not os.path.isdir(search_path):
        return []
    
    # Find files
    files = glob.glob(os.path.join(search_path, "*.parquet"))
    
    # Extract file names without extension
    return [os.path.basename(f).replace('.parquet', '') for f in files]

def get_available_assets_for_category(category, subcategory=None, data_dir=None):
    """
    Get a list of assets that have data for a specific category.

    Parameters:
    -----------
    category : str
        Data category (e.g., 'futures')
    subcategory : str, optional
        Subcategory (e.g., 'funding_rate')
    data_dir : str, optional
        Specific data directory to use. If None, the latest directory is used.

    Returns:
    --------
    list
        List of assets that have data for this category
    """
    files = get_available_files(category, subcategory, data_dir)
    assets = []

    # First look for specific asset matches in filenames
    for asset in SUPPORTED_ASSETS:
        # Check if any files contain this asset in a more structured way
        if any(asset.lower() in file.lower() for file in files):
            assets.append(asset)

    # If no assets were found but files exist, try a more lenient approach
    if not assets and files:
        logger.info(f"No specific asset matches found, but {len(files)} files exist. Using default assets.")
        # Just return all supported assets if there are files in this category
        return SUPPORTED_ASSETS

    # If still no assets but we know we have the directories, just use default assets
    if not assets:
        data_dir = get_latest_data_directory()
        if data_dir:
            base_path = os.path.join(DATA_BASE_PATH, data_dir)
            if subcategory:
                path = os.path.join(base_path, category, subcategory)
            else:
                path = os.path.join(base_path, category)

            if os.path.exists(path):
                logger.info(f"Directory exists at {path} but no asset matches found. Using default assets.")
                return SUPPORTED_ASSETS

    return assets

def calculate_metrics(df, category, subcategory=None):
    """
    Calculate relevant metrics based on data category.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to calculate metrics from
    category : str
        Data category (e.g., 'etf', 'futures')
    subcategory : str, optional
        Subcategory (e.g., 'funding_rate', 'liquidation')
        
    Returns:
    --------
    dict
        Dictionary of calculated metrics
    """
    metrics = {}
    
    if df.empty:
        return metrics
    
    # ETF metrics
    if category == 'etf':
        if 'aum_usd' in df.columns:
            metrics['Total AUM'] = pd.to_numeric(df['aum_usd'], errors='coerce').sum()
        if 'fund_flow_usd' in df.columns:
            metrics['Net Flow (24h)'] = pd.to_numeric(df['fund_flow_usd'], errors='coerce').sum()
    
    # Futures liquidation metrics
    elif category == 'futures' and subcategory == 'liquidation':
        if 'aggregated_long_liquidation_usd' in df.columns and 'aggregated_short_liquidation_usd' in df.columns:
            metrics['Total Liquidations'] = pd.to_numeric(df['aggregated_long_liquidation_usd'], errors='coerce').sum() + \
                                           pd.to_numeric(df['aggregated_short_liquidation_usd'], errors='coerce').sum()
            metrics['Long Liquidations'] = pd.to_numeric(df['aggregated_long_liquidation_usd'], errors='coerce').sum()
            metrics['Short Liquidations'] = pd.to_numeric(df['aggregated_short_liquidation_usd'], errors='coerce').sum()
    
    # Futures open interest metrics
    elif category == 'futures' and subcategory == 'open_interest':
        if 'open_interest_usd' in df.columns:
            metrics['Total Open Interest'] = pd.to_numeric(df['open_interest_usd'], errors='coerce').sum()
        elif 'close' in df.columns:  # For OHLC data
            metrics['Current Open Interest'] = pd.to_numeric(df['close'], errors='coerce').iloc[-1] if len(df) > 0 else 0
    
    # Futures funding rate metrics
    elif category == 'futures' and subcategory == 'funding_rate':
        if 'funding_rate' in df.columns:
            metrics['Average Funding Rate'] = pd.to_numeric(df['funding_rate'], errors='coerce').mean()
            metrics['Max Funding Rate'] = pd.to_numeric(df['funding_rate'], errors='coerce').max()
            metrics['Min Funding Rate'] = pd.to_numeric(df['funding_rate'], errors='coerce').min()
    
    return metrics

def get_historical_comparison(df, timestamp_col='time', value_col=None, periods=None):
    """
    Calculate historical comparisons for a time series.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame with time series data
    timestamp_col : str
        The name of the timestamp column
    value_col : str
        The name of the value column to compare
    periods : dict, optional
        Dictionary of periods to compare (e.g., {'1d': 1, '7d': 7, '30d': 30})
        
    Returns:
    --------
    dict
        Dictionary of comparison metrics
    """
    if df.empty or timestamp_col not in df.columns or value_col is None or value_col not in df.columns:
        return {}
    
    # Process timestamps if not already done
    if 'datetime' not in df.columns:
        df = process_timestamps(df, timestamp_col)
    
    # Default periods if not provided
    if periods is None:
        periods = {'1d': 1, '7d': 7, '30d': 30}
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Get the latest value
    latest_value = pd.to_numeric(df[value_col], errors='coerce').iloc[-1] if len(df) > 0 else None
    
    comparison = {'latest': latest_value}
    
    # Calculate changes for each period
    for period_name, days in periods.items():
        cutoff_date = df['datetime'].max() - timedelta(days=days)
        historical_df = df[df['datetime'] <= cutoff_date]
        
        if not historical_df.empty:
            historical_value = pd.to_numeric(historical_df[value_col], errors='coerce').iloc[-1]
            absolute_change = latest_value - historical_value
            percentage_change = (absolute_change / historical_value) * 100 if historical_value != 0 else None
            
            comparison[f'{period_name}_value'] = historical_value
            comparison[f'{period_name}_abs_change'] = absolute_change
            comparison[f'{period_name}_pct_change'] = percentage_change
    
    return comparison

def convert_df_to_csv(df):
    """
    Convert a DataFrame to a CSV string for download.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to convert
        
    Returns:
    --------
    str
        CSV string
    """
    return df.to_csv(index=False).encode('utf-8')