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

### Run API Endpoints

Use the `report.py` tool to select and run specific API endpoints:

```bash
# Interactive menu
python report.py

# Run all API endpoints
python report.py --run-all

# Run all endpoints in a specific category
python report.py --category etf

# List all available endpoints
python report.py --list
```

## Project Structure

- `coinglass-api/`: Python scripts to access CoinGlass API endpoints
- `data/coinglass/`: CSV files with saved API responses
- `main.py`: Entry point for the application
- `report.py`: Tool to run selected API endpoints
- `modify_api_files.py`: Script to update API files to save CSV data

## API Categories

- **ETF Data**: Bitcoin and Ethereum ETF metrics (flows, assets, etc.)
- **Futures Market Data**: Funding rates, liquidations, open interest, etc.
- **Market Indicators**: Various market indicators like Fear & Greed index
- **On-chain Data**: Exchange balances and transactions
- **Options Data**: Options exchange information
- **Spot Market Data**: Order books and market data

## Git Commands

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/FelipeODonnell/streamlit-stablecoin-dashboard.git
```