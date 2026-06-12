# Macro-Technical Swing Trading Dashboard

Streamlit dashboard overlaying COT positioning, yield spreads, and price action for macro swing trading signals.

## Quickstart

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
bash setup.sh        # installs tvdatafeed from GitHub

# 3. Fetch data (~2 min, produces data/*.parquet)
python data_pipeline.py

# 4. Launch dashboard
streamlit run app.py
```

## How it works

The dashboard computes two 3-year rolling COT metrics on leveraged money (speculator) net positions:

- **Range Score**: min-max normalization to [0, 100] — *(Current − 3yr Min) / (3yr Max − 3yr Min)*
- **Percentile**: rank-based — what fraction of past 3 years the current reading exceeds

Both use a period-based window (156 weeks) since COT data is released weekly.

## Interpreting the chart

| COT bar color | Meaning | Swing trading signal |
|---|---|---|
| Red (≥80) | Speculators max-long — crowded trade | Contrarian bias to **sell** |
| Green (≤20) | Speculators max-short — extreme fear | Contrarian bias to **buy** |
| Grey | Neutral range | No extreme signal |

The overlay (orange line) is the US 10Y-2Y Treasury yield spread — a macro risk barometer. VIX (purple) toggles optionally.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New App" → select your repo → point to `app.py`
4. Streamlit reads `requirements.txt` and runs `setup.sh` automatically

**Note**: `tvdatafeed` uses a free TradingView session (no login required). Data may be limited — see [the repo](https://github.com/rongardF/tvdatafeed) for details on providing credentials.

## Project structure

```
data_pipeline.py   — fetches COT (CFTC), yields/VIX (TradingView), FedWatch (CME)
app.py             — Streamlit dashboard with Plotly subplot
requirements.txt   — PyPI dependencies
setup.sh           — post-install hook for tvdatafeed (Streamlit Cloud)
data/              — cached Parquet files (.gitignored)
```

## Data sources

- **COT**: CFTC Commitments of Traders, TFF Futures+Options report via `cot-reports` library
- **Yields/VIX/FX**: TradingView via `tvdatafeed` (daily bars, no login required)
- **FedWatch**: CME endpoint requires $25/mo API subscription — currently disabled
