# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a liquidity reporting tool that fetches and analyzes cryptocurrency market data from the Coinglass API. The project is set up to collect data on various metrics including:

- ETF information (Bitcoin and Ethereum ETFs, flows, assets)
- Futures market data (funding rates, liquidations, open interest)
- Market indicators
- On-chain analytics
- Options market data
- Spot market data

The project is structured to mirror the API endpoints for easy reference and organization.

## Development Environment Setup

### Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

### Install Dependencies

Dependencies are managed using pyproject.toml:

```bash
# Install dependencies using pip
pip install -e .
```

## Running the Application

To run the main application:

```bash
# Run with Python
python main.py

# Run with Streamlit (if implemented)
streamlit run main.py
```

## Project Structure

- `coinglass-api/`: Python scripts that directly access Coinglass API endpoints, organized by category (futures, spot, etc.)
- `data/`: Likely stores or processes data retrieved from the API
- `main.py`: Entry point for the application

## API Usage Notes

- The project uses the Coinglass API v4
- API calls require headers with a CG-API-KEY
- Be careful not to expose or commit API keys

## Data Categories

- **ETF Data**: Bitcoin and Ethereum ETF metrics, AUM, flows, price histories
- **Futures Market Data**: 
  - Funding rates
  - Liquidations
  - Long/short ratios
  - Market data
  - Open interest
  - Order book data
  - Taker buy/sell volumes
  - Whale positions
- **Market Indicators**: Various market indicators like Fear & Greed index, RSI, moving averages
- **On-chain Data**: Exchange balances, transactions, assets
- **Options Data**: Options exchange information, open interest, volumes
- **Spot Market Data**: Order books, market data, taker volumes


# All supported crypto coins:
["BTC","ETH","SOL","XRP","DOGE","SUI","ADA","HYPE","BNB","TRUMP","LINK","PEPE","LTC","FARTCOIN","1000PEPE","AVAX","WIF","ENA","ONDO","DOT","UNI","AAVE","1000BONK","BCH","TRX","HBAR","NEAR","SHIB","PNUT","WLD","MOODENG","POPCAT","TAO","FIL","kPEPE","ZKJ","KAS","VIRTUAL","ARB","APT","TON","TIA","OP","CRV","XLM","EOS","INIT","ETC","GOAT","ALCH","ORDI","SEI","OM","AI16Z","GALA","RENDER","LDO","ATOM","S","INJ","FET","KAITO","JUP","BGB","PENGU","BERA","ETHFI","MOVE","NEIROETH","POL","LAYER","MKR","BRETT","ALGO","SAND","VET","KAVA","RUNE","GRASS","APE","ICP","STX","ENS","STRK","AIXBT","1000SHIB","BOME","TURBO","PAXG","NIL","DYDX","EIGEN","ZRO","PENDLE","kBONK","ZEREBRO","PEOPLE","PYTH","FORM","NEO","MEW","IP","VINE","NOT","MANA","1000CHEEMS","HMSTR","CAKE","ACT","GRT","CHILLGUY","MUBARAK","TRB","ARKM","PARTI","IMX","ARC","DEEP","BIGTIME","WCT","GMT","ZK","USUAL","THETA","JASMY","MELANIA","AR","UXLINK","NEIRO","XMR","AXS","GRIFFAIN","1000FLOKI","BABY","IOTA","AUCTION","SPX","JTO","TRUMPOFFICIAL","IO","FLOKI","SWARMS","W","TST","DOG","CFX","SUSHI","SOLV","PLUME","SIGN","MASK","ATH","RSR","REZ","PROMPT","PIPPIN","MEME","DOGS","CATI","JELLYJELLY","XAUT","CRO","RAY","CHZ","COMP","SHIB1000","MAGIC","LPT","HIPPO","SONIC","BONK","DEXE","BIO","KOMA","HIFI","VANA","SAGA","WAL","API3","VVV","OBOL","USDC","EGLD","ME","XAI","MNT","1000RATS","BANANAS31","BLUR","SXT","HAEDAL","ZETA","CORE","VOXEL","AERO","BSV","STO","1000SATS","MINA","GUN","COOKIE","XTZ","GORK","TUT","MOCA","SHELL","CELO","GAS","RARE","FLM","1000000MOG","PUNDIX","AVAAI","ORCA","FLOW","SNX","ANIME","AERGO","SSV","CETUS","AKT","ACH","DOOD","SIREN","1MBABYDOGE","CKB","ZEC","KSM","BEAM","TSTBSC","BANANA","EPIC","1000CAT","DYM","SOLAYER","SCR","SXP","SUNDOG","AEVO","1INCH","ROSE","ZEN","YFI","YGG","MYRO","PONKE","METIS","AIOT","MEMEFI","ZIL","COW","QTUM","DRIFT","SATS","ONE","ENJ","COTI","BMT","STORJ","SUPER","KERNEL","FHE","SAFE","QNT","IOTX","FXS","AGLD","PIXEL","MAVIA","WOO","LUNC","MANTA","SYRUP","FIDA","ARK","MORPHO","AXL","BTCDOM","HOOK","RED","ID","DF","BGSC","GMX","LRC","VTHO","KAIA","ACE","BROCCOLI","TOKEN","FWOG","ANKR","VANRY","KDA","NEIROCTO","BR","HYPER","TRU","ETHW","BANK","ZRX","SLERF","ALT","PORTAL","BB","ETHBTC","GLM","XCN","CYBER","PUFFER","TONCOIN","BAN","HOT","BEAMX","ICX","USTC","COIN50","TAI","UMA","MILK","EPT","AI","DEGEN","BLAST","STG","PEAQ","AIOZ","ZORA","HIGH","DASH","TNSR","TWT","ALPHA","RVN","FILECOIN","KMNO","THE","T","CTC","BROCCOLIF3B","LISTA","CGPT","SFP","NKN","SWELL","BSW","TOSHI","RFC","BAKE","HEI","PURR","POLYX","1000000BABYDOGE","SUN","GIGA","C98","X","XVG","ASTR","SERAPH","CVX","PRCL","LOOKS","BEL","WAVES","JOE","HIVE","JST","CHR","STEEM","RDNT","SYS","MOVR","NMR","LQTY","DEGO","RAYSOL","1000LUNC","PERP","LUNA2","EDU","BROCCOLI714","BAT","ONT","PHB","ATA","SKL","RPL","HNT","LEVER","PUMP","STPT","PHA","AGI","LUMIA","SYN","ALICE","IOST","KNC","ILV","BAND","FIO","VELO","OL","XDC","OMNI","TLM","GPS","NFP","ARPA","CTSI","NC","DARK","AVA","1000TURBO","1000NEIROCTO","SPELL","NTRN","CELR","DENT","ALPINE","BNT","DUCK","RONIN","SCRT","RATS","ACX","MBOX","FORTH","POWR","OGN","CAT","MAV","FLUX","1000X","kNEIRO","B3","LSK","MERL","MTL","GTC","FUN","RAYDIUM","FOXY","CSPR","ASR","RLC","J","BICO","XVS","CARV","PROM","CTK","10000WHY","WAXP","BID","REX","CHESS","XCH","G","COS","1000TOSHI","OG","LUNA","TAIKO","FUEL","CVC","OXT","FIS","QUICK","VELODROME","VRA","FLR","ONG","10000LADYS","B2","DUSK","ALPACA","10000WEN","ELX","GHST","XION","10000SATS","1000XEC","kSHIB","SLP","PRIME","MAJOR","ZEUS","ALEO","HFT","HOUSE","FTT","ZBCN","kFLOKI","SANTOS","RIF","ROAM","ORDER","DIA","CLOUD","MLN","LOKA","BAL","ZRC","CPOOL","1000WHY","AVL","AUDIO","SNT","DGB","DOLO","RON","10000ELON","BADGER","BUZZ","D","ULTI","BABYDOGE","GNO","QUBIC","1000MUMU","NS","MICHI","1000BTT","AIDOGE","SC","PAWS","ORBS","RAD","REI","VIC","TRUMPSOL","ZKSYNC","GT","DBR","GODS","NPC","XEM","FLOCK","LUNANEW","FTN","SWEAT","1000APU","XTER","LAI","SPEC","SUPRA","GLMR","AVAIL","DODO","1000000CHEEMS","CHEEMS","F","MASA","GROK","DSYNC","SCA","DODOX","L3","PYR","KILO","1000000PEIPEI","MYRIA","BOBA","HPOS10I","OSMO","ARCSOL","FB","PAAL","A8","SOLO","10000COQ","RSS3","LUCE","AMP","USDE","1MCHEEMS","ZIG","KEKIUS","OBT","ALU","XRD","SHM","OMG","100000AIDOGE","10000000AIDOGE","MVL","MOG","MBL","SEND","VR","SLF","TROLLSOL","REQ","ALTCOIN","REDSTONE","TOMI","PUMPBTC","ZENT","HYPERLANE","DEFI","FUNTOKEN","MOBILE","MBABYDOGE","ICE","PSG","MDT","TRY","RAI","APEX","SD","CLANKER","XNO","GHIBLI","RACA","10000QUBIC","ELON","VISTA","PIN","MANEKI","COOK","ALPH","BLZ","NAKA","DONKEY","IDEX","QI","DAG","ORAI","kLUNC","MIGGLES","TEL","JAGER","OMNI1","FRED","NAVX","WHY","AZERO","NFT","MUBARAKAH","BERT","HSK","BOOP","SNEK","LOOM","UFD","CLORE","VIDT","EUR","ULTIMA","WEMIX","DEAI","FDUSD","TITCOIN","HARRY","ALCX","1000CATS","BUTTHOLE","SQD","BUBB","OMNINETWORK","MUBI","DATA","DOGEGOV","DEVVE","BRL","BAIDOGE","SWFTC","GORILLABSC","LMWR","RATO","ZERO","G7","WIN","NEXO","VVAIFU","EDGE","GPU","PUMPAI","ACA","GHX","MOONPIG","NOS","GOMINING","BTT","SZN","BAR","HOSICO","CULT","PORTO","MAGA","CATS","POPE","USELESS","HOLD","GM","CITY","SWAN","FARM","SPA","POLS","AIC","PIVX","FARTBOY","WIZZ","JPY","ANON","STONKS","TLOS","PWEASE","ACS","LAZIO","DRB","FAI","GM1","OMI","TIBBIR","POND","DOGINME","WHITE","LAVA","VON","CAD","JUV","STNK","RBNT","TFUEL","TORN","WILD","OIK","MDOGS","BFTOKEN","GFM","STRAX","REEF","AUD","RIZ","YZYSOL","APX","MOODENGETH","CAPTAINBNB","SEED","CHF","CBK","WEN","MYX","DADDY","SKYAI","XPR","LADYS","SHX","SLING","WMTX","RIFSOL","AMI","EURC","KMD","GST","BROCCOLIF2B","XEC","LUNAI","SAROS","NEON","STMX","AGIXT","MAGATRUMP","AO","IQ","JAILSTOOL","WING","OZK","PAIN","BRISE","LAT","AMB","1DOLLAR","K","USDT","BOND","kDOGS","TEVA","APU","WAN","MAX","REN","NULS","DCR","LINA","CHEX","TREAT","USA","HAPPY","PEPU","CATTON","UNFI","IMT","100SATS","HASHAI","AURORA","GEAR","WAXL","PELL","BITCOIN","100BONK","NRN","BMEX","RADAR","ELIZA","PEIPEI","KEY","ZCX","QKC","TAT","BENQI","BNBCARD","DHX","GBP","SCROLL","GME","MOON"]

