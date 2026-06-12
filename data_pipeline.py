"""
data_pipeline.py - Data ingestion for Macro-Technical Swing Trading Dashboard
Fetches COT, yield/VIX, and Fed probabilities data, normalizes positioning,
and saves as compressed Parquet files.
"""
import os
import logging
from datetime import datetime, timedelta

import pandas as pd
from cot_reports import cot_reports
from tvDatafeed import TvDatafeed, Interval

log = logging.getLogger(__name__)

CACHE_FRESHNESS_SECONDS = 12 * 3600

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# TFF report: financial futures (currencies, equities, rates, VIX, crypto).
# Disaggregated report: commodities (Gold, Silver, Crude Oil, etc.).
# Each source maps its own report type and speculator column names.
COT_SOURCES = [
    {
        "name": "financials",
        "report_type": "traders_in_financial_futures_futopt",
        "spec_long_col": "Lev_Money_Positions_Long_All",
        "spec_short_col": "Lev_Money_Positions_Short_All",
        "date_col": "Report_Date_as_YYYY-MM-DD",
        "contracts": {
            "Euro FX": "099741",
            "Japanese Yen": "097741",
            "British Pound": "096742",
            "Swiss Franc": "092741",
            "Canadian Dollar": "090741",
            "Australian Dollar": "232741",
            "Mexican Peso": "095741",
            "S&P 500": "13874A",
            "Nasdaq 100": "209747",
            "Dow Jones": "124603",
            "10Y T-Notes": "042601",
            "5Y T-Notes": "044601",
            "2Y T-Notes": "043602",
            "30Y T-Bonds": "020601",
            "VIX": "1170E1",
            "Bitcoin": "133741",
        },
    },
    {
        "name": "commodities",
        "report_type": "disaggregated_futopt",
        # M_Money = Managed Money = speculative community in Disaggregated report
        "spec_long_col": "M_Money_Positions_Long_All",
        "spec_short_col": "M_Money_Positions_Short_All",
        "date_col": "Report_Date_as_YYYY-MM-DD",
        "contracts": {
            "Gold": "088691",
            "Silver": "084691",
            "Copper": "085692",
            "Crude Oil": "067651",
        },
    },
]


COT_REPORT_TYPE = "traders_in_financial_futures_futopt"

DATE_COL = "Report_Date_as_YYYY-MM-DD"

SPECULATOR_COLS = {
    "Long": "Lev_Money_Positions_Long_All",
    "Short": "Lev_Money_Positions_Short_All",
}


def fetch_cot_data(source, years=5):
    """Fetch COT data for one source config's contracts."""
    log.info("Fetching %s COT data (report_type=%s)...", source["name"], source["report_type"])
    end_year = datetime.now().year
    start_year = end_year - years + 1

    dfs = []
    for y in range(start_year, end_year + 1):
        try:
            yr_df = cot_reports.cot_year(y, source["report_type"], store_txt=False, verbose=False)
            if yr_df is not None and not yr_df.empty:
                dfs.append(yr_df)
                log.info("  Year %d: %d rows", y, len(yr_df))
        except Exception as e:
            log.warning("  Year %d skipped: %s", y, e)

    if not dfs:
        return pd.DataFrame(), source

    df = pd.concat(dfs, ignore_index=True)
    date_col = source["date_col"]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    log.info("  Total: %d rows, date range: %s to %s", len(df), df[date_col].min(), df[date_col].max())
    return df, source


COT_PERIODS_1YR = 52  # COT releases weekly (one report per Tuesday)
COT_MIN_PERIODS = 52   # require at least 1 year of history before computing


def _rank_pctile(window_series):
    """Fraction of values in the window ≤ the most recent value, as percent."""
    n = len(window_series)
    return (window_series <= window_series.iloc[-1]).sum() / n * 100 if n else 50.0


def compute_cot_metrics(cot_df, source, window_years=3):
    """Compute 3-year range score + rank-based percentile for one source."""
    log.info("Computing COT metrics for %s (window=%dyr, %d periods)...",
             source["name"], window_years, COT_PERIODS_1YR * window_years)
    results = []
    contracts = source["contracts"]
    total = len(contracts)
    found = 0
    window = COT_PERIODS_1YR * window_years
    long_col = source["spec_long_col"]
    short_col = source["spec_short_col"]
    date_col = source["date_col"]

    for code, cid in contracts.items():
        contract = cot_df[cot_df["CFTC_Contract_Market_Code"] == cid].copy()
        if contract.empty:
            continue
        found += 1

        if long_col not in contract.columns or short_col not in contract.columns:
            avail = [c for c in contract.columns if "Long" in c or "Short" in c]
            log.debug("  %s: expected cols not found; available: %s", code, avail[:6])
            continue

        contract = contract.sort_values(date_col).copy()
        contract["Net"] = contract[long_col].fillna(0) - contract[short_col].fillna(0)

        roll_min = contract["Net"].rolling(window=window, min_periods=COT_MIN_PERIODS).min()
        roll_max = contract["Net"].rolling(window=window, min_periods=COT_MIN_PERIODS).max()
        contract["Range_Score"] = ((contract["Net"] - roll_min) / (roll_max - roll_min + 1e-10)).clip(0, 1) * 100

        contract["Percentile"] = (
            contract["Net"].rolling(window=window, min_periods=COT_MIN_PERIODS)
            .apply(_rank_pctile, raw=False)
        )

        contract["Asset"] = code
        contract["COT_Code"] = cid
        results.append(
            contract[[date_col, "Asset", "COT_Code", "Net", "Range_Score", "Percentile"]]
            .rename(columns={date_col: "Date"})
            .reset_index(drop=True)
        )

    result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    log.info("  %s metrics: %d rows across %d/%d assets",
             source["name"], len(result), result["Asset"].nunique() if not result.empty else 0, found)
    return result


def fetch_tv_data():
    """Fetch yield spreads, VIX, and key indices from TradingView."""
    log.info("Fetching market data from TradingView...")
    tv = TvDatafeed()

    tickers = [
        ("US02Y", "TVC", Interval.in_daily, "US02Y"),
        ("US10Y", "TVC", Interval.in_daily, "US10Y"),
        ("VIX", "TVC", Interval.in_daily, "VIX"),
        ("SPX", "TVC", Interval.in_daily, "SPX"),
        ("EURUSD", "FX_IDC", Interval.in_daily, "EURUSD"),
        ("GBPUSD", "FX_IDC", Interval.in_daily, "GBPUSD"),
        ("USDJPY", "FX_IDC", Interval.in_daily, "USDJPY"),
        ("XAUUSD", "FX_IDC", Interval.in_daily, "XAUUSD"),
        ("XAGUSD", "FX_IDC", Interval.in_daily, "XAGUSD"),
        ("USOIL", "TVC", Interval.in_daily, "USOIL"),
        ("COPPER", "TVC", Interval.in_daily, "COPPER"),
    ]

    frames = []
    for symbol, exchange, interval, label in tickers:
        try:
            df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=2000)
            if df is not None and not df.empty:
                df = df.reset_index()
                df["symbol"] = label
                frames.append(df)
                log.info("  TV %s: %d bars, %s to %s", label, len(df), df["datetime"].min(), df["datetime"].max())
            else:
                log.warning("  TV %s: no data returned", label)
        except Exception as e:
            log.warning("  TV %s: error - %s", label, e)

    if not frames:
        log.warning("No TradingView data received (may need login credentials)")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    return combined


def compute_spreads(tv_df):
    """Compute yield spreads from raw yield data.

    Dates are normalized to calendar days (timezone-aware timestamps
    are truncated to dates). This is intentional — yields, VIX, and FX
    trade in different timezones but daily OHLC bars are aligned by
    calendar day.
    """
    if tv_df.empty:
        return tv_df

    tv_df["date"] = tv_df["datetime"].dt.date

    pivoted = tv_df.pivot_table(index="date", columns="symbol",
                                 values=["close", "open", "high", "low"],
                                 aggfunc="first")

    pivoted.columns = [f"{col[1]}_{col[0]}" for col in pivoted.columns]
    result = pivoted.reset_index()
    result["date"] = pd.to_datetime(result["date"])

    if "US10Y_close" in result.columns and "US02Y_close" in result.columns:
        result["US10Y2Y_Spread"] = result["US10Y_close"] - result["US02Y_close"]

    return result


def _nth_weekday(year, month, weekday, n):
    """Return date of the nth weekday of a month (e.g., 1st Friday of month)."""
    from calendar import monthcalendar
    weeks = monthcalendar(year, month)
    day_weeks = [w[weekday] for w in weeks if w[weekday] != 0]
    return datetime(year, month, day_weeks[n - 1]) if n <= len(day_weeks) else None


# Known FOMC schedule — updated annually. Meetings are 8 per year.
_FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-07", "2025-12-17",
]
_FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]


def generate_calendar():
    """Generate upcoming high-impact economic events.

    Uses the known FOMC schedule, generates NFP (first Friday),
    and CPI (~13th of month) dates for the current year.
    Falls back entirely to pattern generation — no scraping.
    """
    now = datetime.now()
    year = now.year
    events = []

    fomc_dates = _FOMC_DATES_2026 if year == 2026 else _FOMC_DATES_2025

    for ds in fomc_dates:
        dt = datetime.strptime(ds, "%Y-%m-%d")
        if dt >= now - timedelta(days=7):
            events.append({"date": dt, "event": "FOMC Meeting", "impact": "high"})

    for m in range(1, 13):
        dt = _nth_weekday(year, m, 4, 1)  # Friday = 4
        if dt and dt >= now - timedelta(days=7):
            events.append({"date": dt, "event": "NFP / Employment", "impact": "high"})

        cpi_dt = datetime(year, m, 13)
        if cpi_dt >= now - timedelta(days=7):
            events.append({"date": cpi_dt, "event": "CPI", "impact": "high"})

    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values("date").drop_duplicates(subset=["date", "event"]).reset_index(drop=True)
    return df


def save_all():
    """Fetch, compute, and save all data as Parquet files."""
    cutoff_ts = datetime.now().timestamp() - CACHE_FRESHNESS_SECONDS

    cot_file = os.path.join(DATA_DIR, "cot_metrics.parquet")
    tv_file = os.path.join(DATA_DIR, "market_data.parquet")
    cal_file = os.path.join(DATA_DIR, "calendar.parquet")

    if os.path.exists(cot_file) and os.path.getmtime(cot_file) > cutoff_ts:
        log.info("SKIP  %s is < 12h old", cot_file)
    else:
        all_metrics = []
        for source in COT_SOURCES:
            raw, src = fetch_cot_data(source, years=5)
            if raw.empty:
                log.warning("  No data for source %s", source["name"])
                continue
            pct = compute_cot_metrics(raw, src)
            if not pct.empty:
                all_metrics.append(pct)
        if all_metrics:
            combined = pd.concat(all_metrics, ignore_index=True)
            combined.to_parquet(cot_file, index=False)
            log.info("SAVE  COT metrics -> %s (%d rows)", cot_file, len(combined))

    if os.path.exists(tv_file) and os.path.getmtime(tv_file) > cutoff_ts:
        log.info("SKIP  %s is < 12h old", tv_file)
    else:
        tv_raw = fetch_tv_data()
        if not tv_raw.empty:
            tv_processed = compute_spreads(tv_raw)
            tv_processed.to_parquet(tv_file, index=False)
            log.info("SAVE  Market data -> %s (%d rows)", tv_file, len(tv_processed))
        else:
            pd.DataFrame().to_parquet(tv_file, index=False)
            log.warning("SAVE  Empty market data -> %s", tv_file)

    cal_df = generate_calendar()
    cal_df.to_parquet(cal_file, index=False)
    log.info("SAVE  Economic calendar -> %s (%d events)", cal_file, len(cal_df))

    log.info("Pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    save_all()
