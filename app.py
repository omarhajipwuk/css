# app.py
import os
import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
from zoneinfo import ZoneInfo

# =========================
# Streamlit page setup
# =========================
st.set_page_config(
    page_title="Clean Spark + Borders",
    page_icon="⚡",
    layout="wide",
    menu_items={"About": "CSV dashboard + Border Prices (Catalyst API) — login required in-app"},
)

# =========================
# Catalyst API endpoints / config
# =========================
BASE_URL_UK      = "https://forecast.catalystcommodities.com/data/api/v3/daauction/uk"
BASE_URL_BORDERS = "https://forecast.catalystcommodities.com/data/api/v1/contiprices"
LOCAL_TZ         = "Europe/London"
CUTOFF_LOCAL     = "11:00"

BORDER_REGIONS = ["UK", "FR", "NL", "BE", "NO2", "DE", "DK1", "IE"]  # UK is synthesized from EPEX/N2EX

# =========================
# Sidebar nav
# =========================
st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to", ["Clean Spark", "Border Prices"], index=0)

# =========================
# Helpers / time
# =========================
def now_local() -> dt.datetime:
    return dt.datetime.now(ZoneInfo(LOCAL_TZ))

def before_cutoff() -> bool:
    hh, mm = map(int, CUTOFF_LOCAL.split(":"))
    today = now_local().date()
    cutoff = dt.datetime(today.year, today.month, today.day, hh, mm, tzinfo=ZoneInfo(LOCAL_TZ))
    return now_local() < cutoff

# =========================
# Catalyst API — auth + fetch (no token caching across users)
# =========================
def get_catalyst_token(username: str, password: str) -> str:
    url = "https://forecast.catalystcommodities.com/data/token"
    headers = {"content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "password", "username": username, "password": password}
    r = requests.post(url, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

def _extract_price_data(payload: dict, wanted_type: str) -> list[dict]:
    wanted = wanted_type.upper()
    for entry in payload.get("price_data", []) or []:
        if (entry.get("data_type") or "").upper() == wanted:
            return entry.get("data") or []
    return []

def _request_uk_series(token: str, data_type: str, from_date: str, to_date: str) -> pd.DataFrame:
    params  = {"data_type": data_type, "from_date": from_date, "to_date": to_date}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    r = requests.get(BASE_URL_UK, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    records = _extract_price_data(payload, data_type)
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["from_utc", "to_utc", "value", "market"])
    df["from_utc"] = pd.to_datetime(df["from_utc"], utc=True, errors="coerce")
    df["to_utc"]   = pd.to_datetime(df["to_utc"],   utc=True, errors="coerce")
    df["value"]    = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["from_utc"]).sort_values("from_utc").reset_index(drop=True)
    return df

def get_epex_df(token: str, from_date: str, to_date: str) -> pd.DataFrame:
    df = _request_uk_series(token, "EPEX-DAA-1", from_date, to_date)
    if not df.empty:
        df["market"] = "EPEX"
    return df

def get_n2ex_df(token: str, from_date: str, to_date: str) -> pd.DataFrame:
    df = _request_uk_series(token, "N2EX-DAA", from_date, to_date)
    if not df.empty:
        df["market"] = "N2EX"
    return df

def fetch_border_region(token: str, region: str, from_date: str, to_date: str, currency: str = "GBP") -> pd.DataFrame:
    url = f"{BASE_URL_BORDERS}/{region}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"from_date": from_date, "to_date": to_date, "currency": currency}
    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    items = payload.get("data", [])
    if not isinstance(items, list):
        return pd.DataFrame(columns=["utc_datetime","auction_price","forecast_price","region"])

    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=["utc_datetime","auction_price","forecast_price","region"])

    df["utc_datetime"]   = pd.to_datetime(df["utc_datetime"], utc=True, errors="coerce")
    df["auction_price"]  = pd.to_numeric(df.get("price_act"),  errors="coerce")
    df["forecast_price"] = pd.to_numeric(df.get("price_fcst"), errors="coerce")
    df["region"]         = region
    df = df.dropna(subset=["utc_datetime"]).sort_values("utc_datetime").reset_index(drop=True)
    return df[["utc_datetime","auction_price","forecast_price","region"]]

def fetch_all_borders(token: str, regions: list[str], from_date: str, to_date: str, currency: str = "GBP") -> pd.DataFrame:
    frames = []
    for r in regions:
        if r == "UK":   # UK synthesized separately
            continue
        part = fetch_border_region(token, r, from_date, to_date, currency)
        if not part.empty:
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["utc_datetime","auction_price","forecast_price","region"])

# =========================
# CSV loader for Clean Spark
# =========================
ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]

@st.cache_data(show_spinner=False)
def _read_with_encodings_path(path: str) -> tuple[pd.DataFrame, str]:
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            return df, enc
        except Exception as e:
            last_err = e
    raise last_err or UnicodeDecodeError("utf-8", b"", 0, 1, "encoding detection failed")

@st.cache_data(show_spinner=False)
def _read_with_encodings_bytes(b: bytes) -> tuple[pd.DataFrame, str]:
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(io.BytesIO(b), encoding=enc, on_bad_lines="skip")
            return df, enc
        except Exception as e:
            last_err = e
    raise last_err or UnicodeDecodeError("utf-8", b"", 0, 1, "encoding detection failed")

def _normalize_csv_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    if "D" not in df.columns and "D(CS £)" in df.columns:
        df = df.rename(columns={"D(CS £)": "D"})
    if "D" not in df.columns and "D(CS �)" in df.columns:
        df = df.rename(columns={"D(CS �)": "D"})
    if "Date" not in df.columns:
        raise ValueError("CSV must include a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False, infer_datetime_format=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_csv_path(path: str) -> tuple[pd.DataFrame | None, str | None]:
    if not os.path.exists(path):
        return None, None
    df_raw, enc = _read_with_encodings_path(path)
    return _normalize_csv_df(df_raw), enc

@st.cache_data(show_spinner=False)
def load_csv_bytes(b: bytes) -> tuple[pd.DataFrame, str]:
    df_raw, enc = _read_with_encodings_bytes(b)
    return _normalize_csv_df(df_raw), enc

def melt_for_plot(df: pd.DataFrame, chosen_cols: list[str]) -> pd.DataFrame:
    return (
        df.melt(id_vars="Date", value_vars=chosen_cols, var_name="Series", value_name="Value")
          .dropna(subset=["Value"])
    )

# =========================
# Clean Spark page
# =========================
def render_clean_spark():
    st.markdown(
        """
        <style>
        .spark-hero{padding:1.25rem 1.5rem;border-radius:1rem;
        background:radial-gradient(1200px 600px at 10% -20%, rgba(255,230,120,.25), transparent),
                   linear-gradient(135deg,#fff,#f7fafc);
        border:1px solid rgba(0,0,0,.05);}
        .spark-badge{display:inline-block;padding:.25rem .6rem;border-radius:999px;
        background:#fff7cc;border:1px solid #ffec80;font-size:.80rem;font-weight:600;color:#7a5b00;}
        .spark-subtle{color:#666;font-size:.95rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


    with st.sidebar:
        st.subheader("CSV Source")
        default_choice = "css.csv (same folder)" if os.path.exists("css.csv") else "Upload CSV"
        source = st.radio("Choose source", ["css.csv (same folder)", "Upload CSV"],
                          index=0 if default_choice == "css.csv (same folder)" else 1)
        uploaded = st.file_uploader("Upload CSV", type=["csv"]) if source == "Upload CSV" else None

        st.subheader("Options")
        drop_all_nan = st.checkbox("Drop rows where all series are empty", value=True)
        show_markers = st.checkbox("Show markers", value=False)
        show_table = st.checkbox("Show data table", value=False)
        x_label = st.text_input("X axis label", value="Date")
        y_label = st.text_input("Y axis label", value="Value")
        legend_title = st.text_input("Legend title", value="Series")

    df = None
    used_enc = None
    try:
        if source == "css.csv (same folder)":
            df, used_enc = load_csv_path("css.csv")
            if df is None:
                st.warning("No css.csv found. Switch to 'Upload CSV' or place css.csv next to app.py.")
        else:
            if uploaded is not None:
                df, used_enc = load_csv_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

    if df is None or df.empty:
        st.info("Load a CSV to see the chart here.")
        return

    numeric_cols = [c for c in df.columns if c != "Date"]
    if drop_all_nan and numeric_cols:
        df = df.loc[~df[numeric_cols].isna().all(axis=1)].reset_index(drop=True)

   #st.success(f"CSV loaded{f' (encoding: {used_enc})' if used_enc else ''}.")
    chosen = st.multiselect("Series to plot", options=numeric_cols, default=numeric_cols)
    if not chosen:
        st.info("Select at least one series to plot.")
        return

    plot_df = melt_for_plot(df, chosen)
    fig = px.line(
        plot_df, x="Date", y="Value", color="Series", template="simple_white",
        title="Border Prices",
        labels={"Date": x_label, "Value": y_label, "Series": legend_title},
    )
    fig.update_traces(line=dict(width=2.0), mode="lines+markers" if show_markers else "lines")
    fig.update_layout(margin=dict(t=60, r=20, l=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if show_table:
        st.dataframe(df, use_container_width=True)

# =========================
# Default page
# =========================

# =========================
# Border Prices page (login required via inputs)
# =========================
def _to_display_zone(ts: pd.Series, tz_option: str) -> pd.Series:
    # incoming timestamps are UTC-aware
    if tz_option == "Europe/London":
        return ts.dt.tz_convert("Europe/London")
    return ts  # UTC

def render_border_prices():
    st.title("🌍 Border Prices (Catalyst)")

    # ---- Login section (stored per user session) ----
    if "catalyst_token" not in st.session_state:
        st.session_state["catalyst_token"] = None
        st.session_state["catalyst_user"] = None
        st.session_state["token_time"] = None

    with st.sidebar:
        st.subheader("Catalyst login (required)")
        with st.form("login_form", clear_on_submit=True):
            user_in = st.text_input("Username", value="")
            pass_in = st.text_input("Password", type="password", value="")
            login_btn = st.form_submit_button("Login")

        if login_btn:
            if not user_in or not pass_in:
                st.error("Enter both username and password.")
            else:
                try:
                    token = get_catalyst_token(user_in, pass_in)
                    st.session_state["catalyst_token"] = token
                    st.session_state["catalyst_user"] = user_in
                    st.session_state["token_time"] = dt.datetime.utcnow()
                    st.success("Logged in.")
                except Exception as e:
                    st.session_state["catalyst_token"] = None
                    st.session_state["catalyst_user"] = None
                    st.error(f"Login failed: {e}")

        if st.session_state["catalyst_token"]:
            st.caption(f"Logged in as **{st.session_state['catalyst_user']}**")
            if st.button("Log out"):
                st.session_state["catalyst_token"] = None
                st.session_state["catalyst_user"] = None
                st.session_state["token_time"] = None
                st.experimental_rerun()

    token = st.session_state.get("catalyst_token")
    if not token:
        st.info("Please log in with your Catalyst credentials in the sidebar to fetch data.")
        return

    # ---- Query controls ----
    with st.sidebar:
        st.subheader("Query")
        date_sel = st.date_input("Delivery date (UTC day)", value=dt.date.today())
        currency = st.selectbox("Currency", ["GBP", "EUR"], index=0)
        include_uk = st.checkbox("Include UK (Auction=N2EX, Forecast=EPEX)", value=True)
        regions = st.multiselect(
            "Regions",
            options=[r for r in BORDER_REGIONS if r != "UK"],
            default=[r for r in BORDER_REGIONS if r not in ("UK",)],
        )
        tz_option = st.selectbox("Timestamp zone", ["UTC", "Europe/London"], index=0)
        show_table = st.checkbox("Show data table", value=False)

        st.subheader("Chart options")
        show_auction = st.checkbox("Show Auction", value=True)
        show_forecast = st.checkbox("Show Forecast", value=True)
        show_markers = st.checkbox("Show markers", value=False)

    # ---- Fetch / build data ----
    day_str = date_sel.strftime("%Y-%m-%d")

    borders_today = fetch_all_borders(token, regions, from_date=day_str, to_date=day_str, currency=currency)

    # Synthesize UK from separate endpoints
    uk_df = pd.DataFrame(columns=["utc_datetime","auction_price","forecast_price","region"])
    if include_uk:
        epex = get_epex_df(token, from_date=day_str, to_date=day_str)    # Forecast
        n2ex = get_n2ex_df(token, from_date=day_str, to_date=day_str)    # Auction
        if not n2ex.empty:
            tmp = pd.DataFrame({
                "utc_datetime": pd.to_datetime(n2ex["from_utc"], utc=True, errors="coerce"),
                "auction_price": pd.to_numeric(n2ex["value"], errors="coerce"),
            })
            uk_df = tmp
        if not epex.empty:
            tmp2 = pd.DataFrame({
                "utc_datetime": pd.to_datetime(epex["from_utc"], utc=True, errors="coerce"),
                "forecast_price": pd.to_numeric(epex["value"], errors="coerce"),
            })
            if uk_df.empty:
                uk_df = tmp2
            else:
                uk_df = uk_df.merge(tmp2, on="utc_datetime", how="outer")
        if not uk_df.empty:
            uk_df["region"] = "UK"
            uk_df = uk_df.sort_values("utc_datetime").reset_index(drop=True)

    combined = borders_today
    if not uk_df.empty:
        combined = pd.concat([combined, uk_df[combined.columns]], ignore_index=True) if not combined.empty else uk_df.copy()

    if combined.empty:
        st.warning("No data returned for the selected date/regions.")
        return

    combined = combined.copy()
    combined["utc_datetime"] = pd.to_datetime(combined["utc_datetime"], utc=True, errors="coerce")
    combined["display_time"] = _to_display_zone(combined["utc_datetime"], tz_option)

    # Long-form for plotting
    use_cols = []
    if show_auction:
        use_cols.append("auction_price")
    if show_forecast:
        use_cols.append("forecast_price")
    if not use_cols:
        st.info("Select at least one of Auction / Forecast to plot.")
        return

    long_df = combined[["display_time", "region"] + use_cols].copy()
    long_df = long_df.melt(id_vars=["display_time", "region"], var_name="kind", value_name="price")
    long_df["kind"] = long_df["kind"].map({"auction_price": "Auction", "forecast_price": "Forecast"})
    long_df = long_df.dropna(subset=["price"])

    # ---- Chart ----
    title = f"Border Prices — {day_str} ({tz_option})"
    fig = px.line(
        long_df,
        x="display_time", y="price",
        color="region", line_dash="kind",
        template="simple_white",
        title=title,
        labels={"display_time": "Time", "price": f"Price ({currency}/MWh)", "region": "Region", "kind": "Type"},
    )
    fig.update_traces(line=dict(width=2.0), mode="lines+markers" if show_markers else "lines")
    fig.update_layout(margin=dict(t=60, r=20, l=10, b=10), legend_title_text="Region / Type")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Optional table ----
    if show_table:
        out = combined.sort_values(["region", "utc_datetime"]).reset_index(drop=True)
        st.dataframe(out, use_container_width=True)

    st.caption("UK: Auction=N2EX, Forecast=EPEX. Times shown in the selected zone for display; underlying data are hourly UTC.")

# =========================
# Router
# =========================
if view == "Clean Spark":
    render_clean_spark()
else:
    render_border_prices()

# =========================
# Footer
# =========================
st.write("---")
st.caption(" note to self -- fix table switching errors.")






