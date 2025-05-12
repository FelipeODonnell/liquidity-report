# Izun Liquidity Report - User Guide

This user guide provides comprehensive instructions on how to use the Izun Liquidity Report tool, which collects cryptocurrency market data from the CoinGlass API and presents it in an organized format.

## Project Overview

The Izun Liquidity Report tool:
- Fetches cryptocurrency data from CoinGlass API
- Organizes data by date in a structured folder system
- Saves data in Parquet format for efficient storage and querying
- Provides a Streamlit interface for data visualization
- Focuses on BTC, ETH, XRP, and SOL cryptocurrencies

## Table of Contents

1. [Setup](#setup)
2. [Data Collection Process](#data-collection-process)
3. [Running the Application](#running-the-application)
4. [Troubleshooting](#troubleshooting)

## Setup

### Prerequisites

- Python 3.8 or higher
- Git (for version control)
- Internet connection (for API access)

### Creating a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate.bat

# Deactivate when done
deactivate
```

### Install Dependencies

```bash
# Install requirements
pip install -e .
```

## Data Collection Process

The data collection process involves three main steps:

1. **Update API Time Parameters** - Update API URLs with current timestamps
2. **Run the Data Collection Tool** - Collect data from CoinGlass API
3. **Launch the Visualization Interface** - View and analyze the collected data

### Step 1: Update API Time Parameters

Before collecting data, you should update the API time parameters to ensure you're getting the most recent 6-month data window. You have two options:

#### Option 1: Full Parameter Update (First-time setup)

```bash
# Run the edit_time.py script to update all API parameters
python edit_time.py
```

This script:
- Calculates the current timestamp and the timestamp from 6 months ago
- Updates all API URLs in the project to use these timestamps
- Sets the data granularity to 4-hour intervals
- Ensures data is returned in USD

Use this script for initial setup or if you want to reset all URL parameters to default values.

#### Option 2: Refresh Time Parameters Only (Regular updates)

```bash
# Run the refresh_time.py script to update only time parameters
python refresh_time.py
```

This script:
- Calculates the current timestamp and the timestamp from 6 months ago
- Updates **only** the start_time and end_time parameters in API URLs
- Preserves all other parameters (interval, limit, unit, etc.)
- Creates backups with .time.bak extension

Use this script for regular updates to keep your data window current without changing other parameters you may have customized.

**Important:** Run one of these scripts periodically (weekly or monthly) to ensure you're always collecting the most recent 6 months of data.

### Step 2: Run the Data Collection Tool

After updating the time parameters, run the data collection tool:

```bash
# Run the report.py tool to collect data from all API endpoints
python report.py
```

This script:
- Calls all the API endpoints defined in the coinglass-api directory
- Organizes responses by date in the data directory (YYYYMMDD format)
- Saves data in Parquet format
- Implements rate limiting to prevent API throttling (29 requests/minute)

You can use additional options with the report.py tool:

```bash
# Discover and update the list of API files
python report.py --discover

# Specify custom rate limit (default is 29 requests/minute)
python report.py --max-rate 15
```

The data will be saved to folders based on the current date:
```
data/
└── 20250512/  # Date format: YYYYMMDD
    ├── etf/
    │   └── api_etf_bitcoin_list.parquet
    ├── futures/
    │   └── ...
    └── ...
```

## Running the Application

After collecting the data, you can launch the Streamlit application to visualize and analyze it:

```bash
# Run the main Streamlit application
streamlit run main.py
```

This will start the Streamlit server and open the application in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

Alternatively, you can run the main script directly:

```bash
# Run the main application script
python main.py
```

## Complete Workflow

Here's the complete workflow for running the Izun Liquidity Report tool:

```bash
# 1. Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate.bat  # On Windows

# 2. Update API time parameters (do this weekly or monthly)
python refresh_time.py  # Only updates time parameters, preserves other settings
# or
python edit_time.py  # Full update of all parameters (use for initial setup)

# 3. Collect data from the CoinGlass API
python report.py

# 4. Launch the Streamlit application
streamlit run main.py

# 5. Deactivate the virtual environment when done
deactivate
```

## Data Files

The project generates and uses the following key data files:

- **Raw API Data**: Stored in `data/YYYYMMDD/` folders in Parquet format
- **Symbol-Specific Files**: Files with _BTC, _ETH, _XRP, and _SOL suffixes contain data for those specific cryptocurrencies

## Troubleshooting

### API Rate Limiting

If you encounter rate limiting issues, you can reduce the API request rate:

```bash
python report.py --max-rate 15  # Reduce to 15 requests per minute
```

### Restore Original API Files

If you need to restore the original API files without the time parameters:

```bash
# Find all backup files and restore them
find coinglass-api -name "*.bak" -exec bash -c 'cp "{}" "${0%.bak}"' {} \;
```

### Check for Missing Data

If you suspect some data is missing, you can:

1. Run the report.py script with the --discover flag to ensure all API files are included
2. Check the data directory for missing or empty files
3. Run report.py again to collect any missing data

## Additional Tools

The project includes several utility scripts:

- `adjust_time.py`: Displays current timestamp and the timestamp from 6 months ago
- `edit_time.py`: Updates all API URL parameters including timestamps, interval (4h), and unit (usd)
- `refresh_time.py`: Updates only start_time and end_time parameters, preserving all other settings
- `limit-change.py`: Adjusts API request limits to 4500 where applicable
- `remove_duplicates.py`: Removes base files that have symbol-specific versions
- `parquet.py`: Converts CSV data storage to Parquet format
- `create_target.py`: Filters exchange data for target cryptocurrencies
- `clean_backups.py`: Removes backup files from the project

## Best Practices

1. **Regular Updates**: Run `refresh_time.py` at least monthly to keep your data window current
2. **Parameter Management**: Use `refresh_time.py` for routine updates to preserve your customized settings
3. **API Key Management**: Keep your CoinGlass API key secure
4. **Data Validation**: Periodically check the collected data for completeness and accuracy
5. **Backup Important Data**: Create backups of important data directories

## Conclusion

By following this guide, you should be able to effectively collect, store, and visualize cryptocurrency market data using the Izun Liquidity Report tool. Remember to regularly update your time parameters to ensure you're working with the most recent data.

For more information, refer to the README.md file or contact the project maintainer.