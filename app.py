"""
app.py - Macro-Technical Swing Trading Dashboard
Streamlit + Plotly dashboard with candlestick charts, yield spread overlay,
and COT positioning metrics (3Y range score + percentile).
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Macro-Technical Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Data Loading (with Streamlit caching)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=43200)
def load_cot_data():
    path = os.path.join(DATA_DIR, "cot_metrics.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(ttl=43200)
def load_market_data():
    path = os.path.join(DATA_DIR, "market_data.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=43200)
def load_calendar():
    path = os.path.join(DATA_DIR, "calendar.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
ASSET_COT_MAP = {
    "EURUSD": "Euro FX",
    "GBPUSD": "British Pound",
    "USDJPY": "Japanese Yen",
    "AUDUSD": "Australian Dollar",
    "USDCAD": "Canadian Dollar",
    "USDCHF": "Swiss Franc",
    "USDMXN": "Mexican Peso",
    "SPX": "S&P 500",
    "VIX": "VIX",
    "XAUUSD": "Gold",
    "XAGUSD": "Silver",
    "USOIL": "Crude Oil",
    "COPPER": "Copper",
}

ASSET_TV_MAP = {
    "EURUSD": ("EURUSD", "FX_IDC"),
    "GBPUSD": ("GBPUSD", "FX_IDC"),
    "USDJPY": ("USDJPY", "FX_IDC"),
    "AUDUSD": ("AUDUSD", "FX_IDC"),
    "USDCAD": ("USDCAD", "FX_IDC"),
    "USDCHF": ("USDCHF", "FX_IDC"),
    "USDMXN": ("USDMXN", "FX_IDC"),
    "SPX": ("SPX", "TVC"),
    "VIX": ("VIX", "TVC"),
    "XAUUSD": ("XAUUSD", "FX_IDC"),
    "XAGUSD": ("XAGUSD", "FX_IDC"),
    "USOIL": ("USOIL", "TVC"),
    "COPPER": ("COPPER", "TVC"),
}

COT_METRICS = {
    "Range Score (min-max)": "Range_Score",
    "Percentile (rank-based)": "Percentile",
}


def get_asset_list(cot_df):
    available = []
    if not cot_df.empty:
        cot_assets = set(cot_df["Asset"].unique())
        for tv_name, cot_name in ASSET_COT_MAP.items():
            if cot_name in cot_assets:
                available.append(tv_name)
    if not available:
        available = list(ASSET_TV_MAP.keys())
    return sorted(available)


# ---------------------------------------------------------------------------
# Trade Signal Engine
# ---------------------------------------------------------------------------
def compute_signal(cot_df, tv_df, selected_asset):
    """Compute a swing-trading signal for the next week based on COT + yield spread.

    Returns a dict: {signal, action, cot_score, spread_score, cot_rs, spread_val,
                     confidence, color, details}
    """
    result = {
        "signal": "NO DATA",
        "action": "—",
        "cot_score": 0,
        "spread_score": 0,
        "cot_rs": None,
        "spread_val": None,
        "confidence": 0,
        "color": "grey",
        "details": "",
    }

    cot_name = ASSET_COT_MAP.get(selected_asset)
    if cot_name is None:
        result["details"] = "No COT mapping for this asset"
        return result

    asset_cot = cot_df[cot_df["Asset"] == cot_name]
    if asset_cot.empty:
        result["details"] = "No COT data available"
        return result

    latest_cot = asset_cot.sort_values("Date").iloc[-1]
    cot_rs = latest_cot.get("Range_Score")
    if pd.isna(cot_rs):
        result["details"] = "COT metric is NaN (insufficient history)"
        return result

    result["cot_rs"] = cot_rs

    if cot_rs <= 20:
        cot_score = 3
    elif cot_rs <= 35:
        cot_score = 2
    elif cot_rs <= 50:
        cot_score = 1
    elif cot_rs >= 80:
        cot_score = -3
    elif cot_rs >= 65:
        cot_score = -2
    else:
        cot_score = 0

    result["cot_score"] = cot_score

    spread_score = 0
    if not tv_df.empty and "US10Y2Y_Spread" in tv_df.columns:
        spread_vals = tv_df[["date", "US10Y2Y_Spread"]].dropna()
        if not spread_vals.empty:
            latest_spread = spread_vals.sort_values("date").iloc[-1]["US10Y2Y_Spread"]
            result["spread_val"] = latest_spread
            if latest_spread > 0.3:
                spread_score = 1
            elif latest_spread < 0:
                spread_score = -1

    result["spread_score"] = spread_score

    total = cot_score + spread_score
    confidence = min(abs(total) / 4.0 * 100, 95)

    if total >= 3:
        sig, act, col = "STRONG BUY", "Buy", "green"
    elif total >= 2:
        sig, act, col = "BUY", "Buy", "lightgreen"
    elif total >= 1:
        sig, act, col = "LEAN BUY", "Watch for dip to buy", "palegreen"
    elif total <= -3:
        sig, act, col = "STRONG SELL", "Sell / Short", "red"
    elif total <= -2:
        sig, act, col = "SELL", "Sell / Reduce", "salmon"
    elif total <= -1:
        sig, act, col = "LEAN SELL", "Watch for rally to sell", "lightcoral"
    else:
        sig, act, col = "HOLD", "Stay flat, no edge", "grey"

    result["signal"] = sig
    result["action"] = act
    result["color"] = col
    result["confidence"] = round(confidence)
    result["details"] = (
        f"COT Range Score: {cot_rs:.1f} | "
        f"10Y-2Y Spread: {result['spread_val']:.2f}" if result["spread_val"] else "10Y-2Y: n/a"
    )
    return result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.sidebar.title("Macro-Technical Dashboard")
st.sidebar.markdown("Swing trading signals from COT + macro data")

cot_df = load_cot_data()
tv_df = load_market_data()

# First-launch: no parquet files exist yet — run the pipeline automatically
if cot_df.empty and tv_df.empty:
    st.sidebar.warning("No local data — fetching from sources...")
    status = st.info(
        "⏳ **First launch detected.**\n\n"
        "Downloading COT reports and market data from CFTC and TradingView. "
        "This takes ~2 minutes. The page will reload automatically when done."
    )
    try:
        from data_pipeline import save_all
        save_all()
        st.cache_data.clear()
        status.empty()
        st.rerun()
    except Exception as e:
        status.empty()
        st.error(f"Pipeline failed: {e}")
        st.info("Run `python data_pipeline.py` locally, then push the generated "
                "`data/*.parquet` files to the repo.")

if not cot_df.empty:
    age = (datetime.now() - cot_df["Date"].max()).days
    st.sidebar.success(f"COT: {cot_df['Asset'].nunique()} assets, last data {age}d ago")
if not tv_df.empty:
    st.sidebar.info(f"Market: {len(tv_df)} rows")

asset_list = get_asset_list(cot_df)
selected = st.sidebar.selectbox("Select Asset", asset_list, index=0 if asset_list else 0)

lookback = st.sidebar.selectbox("Lookback", ["6M", "1Y", "2Y", "5Y"], index=1)
lookback_days = {"6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825}[lookback]

metric_key = st.sidebar.selectbox("COT Metric", list(COT_METRICS.keys()), index=0)
metric_col = COT_METRICS[metric_key]

show_spread = st.sidebar.checkbox("Show Yield Spread (UST 10Y-2Y)", value=True)
show_vix = st.sidebar.checkbox("Show VIX", value=False)

st.sidebar.divider()
st.sidebar.markdown("**Interpreting COT bars**")
st.sidebar.markdown("- Red (≥80): Speculators max-long → crowded trade, **contrarian bias to sell**")
st.sidebar.markdown("- Green (≤20): Speculators max-short → extreme fear, **contrarian bias to buy**")
st.sidebar.markdown("- Grey: Neutral range — no extreme signal")
st.sidebar.caption("High/lows reflect the speculator (leveraged money) community. "
                    "Extremes are contrarian signals — the crowd is often wrong at turning points.")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# --- Trade Signal ---
signal = compute_signal(cot_df, tv_df, selected)
st.sidebar.divider()
st.sidebar.subheader("Next 7-Day Signal")

sig_color = signal["color"]
st.sidebar.markdown(
    f"<h3 style='text-align:center; color:{sig_color}; margin:0;'>{signal['signal']}</h3>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    f"<p style='text-align:center; font-size:1.1em;'>{signal['action']}</p>",
    unsafe_allow_html=True,
)

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Confidence", f"{signal['confidence']}%")
with col2:
    cot_display = f"{signal['cot_rs']:.0f}" if signal["cot_rs"] is not None else "n/a"
    st.metric("COT Score", cot_display)

if signal["spread_val"] is not None:
    spread_display = f"{signal['spread_val']:.3f}"
    if signal["spread_val"] < 0:
        spread_display += " ⚠️"
    st.sidebar.metric("10Y-2Y Spread", spread_display)

st.sidebar.caption(signal["details"])


# ---------------------------------------------------------------------------
# Build Chart
# ---------------------------------------------------------------------------
def build_chart(selected_asset, cot_df, tv_df, lookback_days,
                metric_col="Range_Score",
                show_spread=True, show_vix=False,
                cal_df=None):
    cutoff = datetime.now() - timedelta(days=lookback_days)

    price_data = pd.DataFrame()
    cot_data = pd.DataFrame()
    sym = None

    if selected_asset in ASSET_TV_MAP:
        sym, exch = ASSET_TV_MAP[selected_asset]
        if not tv_df.empty:
            price_cols = [f"{sym}_close", f"{sym}_open", f"{sym}_high", f"{sym}_low"]
            if all(c in tv_df.columns for c in price_cols):
                price_data = tv_df[["date"] + price_cols].dropna().copy()
                price_data = price_data[price_data["date"] >= cutoff].sort_values("date")

    if not cot_df.empty:
        cot_name = ASSET_COT_MAP.get(selected_asset, selected_asset)
        cot_data = cot_df[cot_df["Asset"] == cot_name].copy()
        if not cot_data.empty:
            cot_data = cot_data[cot_data["Date"] >= cutoff].sort_values("Date")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    # --- Top pane: Candlestick ---
    if sym is not None and not price_data.empty and len(price_data) > 5:
        fig.add_trace(
            go.Candlestick(
                x=price_data["date"],
                open=price_data[f"{sym}_open"],
                high=price_data[f"{sym}_high"],
                low=price_data[f"{sym}_low"],
                close=price_data[f"{sym}_close"],
                name=selected_asset,
            ),
            row=1, col=1,
        )
    else:
        fig.add_annotation(
            text="Price data not available — run data_pipeline.py first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            row=1, col=1,
        )

    # --- Overlay: Yield Spread ---
    if show_spread and not tv_df.empty and "US10Y2Y_Spread" in tv_df.columns:
        spread_data = tv_df[["date", "US10Y2Y_Spread"]].dropna()
        spread_data = spread_data[spread_data["date"] >= cutoff]
        if not spread_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=spread_data["date"],
                    y=spread_data["US10Y2Y_Spread"],
                    mode="lines",
                    name="US 10Y-2Y Spread",
                    line=dict(color="orange", width=1.5),
                    yaxis="y2",
                ),
                row=1, col=1, secondary_y=True,
            )

    if show_vix and not tv_df.empty and "VIX_close" in tv_df.columns:
        vix_data = tv_df[["date", "VIX_close"]].dropna()
        vix_data = vix_data[vix_data["date"] >= cutoff]
        if not vix_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=vix_data["date"],
                    y=vix_data["VIX_close"],
                    mode="lines",
                    name="VIX",
                    line=dict(color="purple", width=1.5, dash="dot"),
                    yaxis="y2",
                ),
                row=1, col=1, secondary_y=True,
            )

    # --- Bottom pane: COT bars ---
    if not cot_data.empty and metric_col in cot_data.columns:
        vals = cot_data[metric_col]
        colors = ["red" if v >= 80 else "green" if v <= 20 else "lightgrey" for v in vals]

        fig.add_trace(
            go.Bar(
                x=cot_data["Date"],
                y=vals,
                name=f"COT {metric_col} ({cot_data['Asset'].iloc[0]})",
                marker_color=colors,
                opacity=0.85,
                showlegend=True,
            ),
            row=2, col=1,
        )

        fig.add_hline(y=80, line_dash="dash", line_color="red",
                      annotation_text=">=80", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green",
                      annotation_text="<=20", row=2, col=1)
        fig.update_yaxes(range=[-5, 105], row=2, col=1,
                         title_text=f"COT {metric_key}")

    # --- Economic Event Markers (top pane) ---
    if cal_df is not None and not cal_df.empty:
        event_lines = cal_df[
            (cal_df["date"] >= cutoff) &
            (cal_df["date"] <= datetime.now() + timedelta(days=30))
        ]
        for _, row in event_lines.iterrows():
            fig.add_vline(
                x=row["date"],
                line_dash="dash",
                line_color="white",
                line_width=0.8,
                opacity=0.4,
                row=1, col=1,
            )
            fig.add_annotation(
                x=row["date"],
                y=1.02,
                yref="paper",
                xref="x",
                text=row["event"],
                showarrow=False,
                font=dict(size=8, color="white"),
                opacity=0.6,
            )

    # --- Layout ---
    fig.update_layout(
        title=f"{selected_asset} — Macro-Technical Dashboard",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=800,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Spread / VIX", row=1, col=1, secondary_y=True,
                     showgrid=False)

    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
cal_df = load_calendar()
fig = build_chart(
    selected, cot_df, tv_df, lookback_days,
    metric_col=metric_col,
    show_spread=show_spread, show_vix=show_vix,
    cal_df=cal_df,
)
st.plotly_chart(fig, width="stretch")

# Show COT as-of date for the selected asset
cot_name = ASSET_COT_MAP.get(selected, selected)
cot_selected = cot_df[cot_df["Asset"] == cot_name]
cot_as_of = cot_selected["Date"].max() if not cot_selected.empty else None
cot_lag_str = ""
if cot_as_of:
    published = cot_as_of + timedelta(days=3)
    cot_lag_str = f"COT for {selected}: as-of {cot_as_of.date()} (published {published.date()})"

tv_max_date = tv_df["date"].max() if not tv_df.empty else None
st.caption(
    f"{cot_lag_str} | "
    f"{'Market: ' + str(tv_max_date.date()) if tv_max_date is not None else ''} | "
    f"Data: {DATA_DIR}"
)
