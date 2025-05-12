# Izun Liquidity Report

Cryptocurrency liquidity reporting tool using the CoinGlass API.

## Project Setup

### Creating a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

### Install Dependencies

```bash
# Install requirements
pip install -e .
```

## Running the Application

```bash
# Run the main application
python main.py

# Run with Streamlit (if implemented)
streamlit run main.py
```

## Data Collection Tools

This repository includes tools to collect and store data from the CoinGlass API:

### Modify API Files

Use the `modify_api_files.py` script to update all the API files to save responses as CSV:

```bash
# Modify all API files to save responses as CSV
python modify_api_files.py

# Dry run (don't make actual changes)
python modify_api_files.py --dry-run

# Create backups before modifying
python modify_api_files.py --backup

# Modify a single file
python modify_api_files.py --single coinglass-api/etf/api_etf_bitcoin_list.py
```

### Run Reports

Use the `report.py` tool to run all API files and save data to a date-based folder structure:

```bash
# First, discover and list all API files
python report.py --discover

# Run all non-commented API files
python report.py

# Specify custom rate limit (defaults to 29 requests/minute)
python report.py --max-rate 15
```

The `report.py` file will automatically generate a list of all API files in the `coinglass-api` directory. To exclude specific files from running, simply add a `#` at the beginning of the line in the `report.py` file.

Features:
- Rate limiting to prevent API rate limit errors (default: 29 requests/minute)
- Date-based organization of collected data
- Ability to selectively run API endpoints

Data will be saved to folders based on the current date:
```
data/
└── 20250512/  # Date format: YYYYMMDD
    ├── etf/
    │   └── api_etf_bitcoin_list.csv
    ├── futures/
    │   └── ...
    └── ...
```

### Data Processing Scripts

#### Convert to Parquet

Use the `parquet.py` script to convert data storage from CSV to Parquet format:

```bash
# Convert all API files to save as parquet
python parquet.py

# Dry run (don't make actual changes)
python parquet.py --dry-run

# Create backups before modifying
python parquet.py --backup

# Modify a single file
python parquet.py --single coinglass-api/etf/api_etf_bitcoin_list.py
```

Parquet files provide better compression and query performance than CSV files.

#### Filter Exchange Data

Use the `create_target.py` script to filter the exchanges.csv file:

```bash
# Filter exchanges.csv to only include BTC, ETH, XRP, and SOL
python create_target.py
```

This script creates a new file called `target_data.csv` that contains only the rows from `exchanges.csv` where the base asset is one of the target cryptocurrencies (BTC, ETH, XRP, or SOL).

#### Clean Backup Files

Use the `clean_backups.py` script to remove backup files:

```bash
# Remove all .bak files from the coinglass-api directory
python clean_backups.py
```

This script removes all backup files (with .bak extension) created during the modification process, keeping your project directory clean.

## Project Structure

- `coinglass-api/`: Python scripts to access CoinGlass API endpoints
- `data/YYYYMMDD/`: Data files with saved API responses organized by date
- `main.py`: Entry point for the application
- `report.py`: Tool to run all API endpoints and save data by date
- `modify_api_files.py`: Script to update API files to save CSV data
- `parquet.py`: Script to convert data storage from CSV to Parquet format
- `create_target.py`: Script to filter exchange data for target cryptocurrencies
- `clean_backups.py`: Script to remove backup files from the project

## API Categories

- **ETF Data**: Bitcoin and Ethereum ETF metrics (flows, assets, etc.)
- **Futures Market Data**: Funding rates, liquidations, open interest, etc.
- **Market Indicators**: Various market indicators like Fear & Greed index
- **On-chain Data**: Exchange balances and transactions
- **Options Data**: Options exchange information
- **Spot Market Data**: Order books and market data

## Target Cryptocurrencies

Initial tracking focused on the top cryptocurrencies:
- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- Ripple (XRP)

## Git Commands

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/FelipeODonnell/streamlit-stablecoin-dashboard.git
```