# app.py
import os, io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Clean Spark Demo", page_icon="⚡", layout="wide")

# ---------- Sidebar navigation ----------
st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to", ["Clean Spark", "Default"], index=0)

# ---------- Robust CSV loading ----------
ENCODINGS_TO_TRY = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
    "iso-8859-1",
    "utf-16",
    "utf-16le",
    "utf-16be",
]

def _read_with_encodings(source, is_bytes=False):
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            if is_bytes:
                buf = io.BytesIO(source)
                df = pd.read_csv(buf, encoding=enc, on_bad_lines="skip")
            else:
                df = pd.read_csv(source, encoding=enc, on_bad_lines="skip")
            return df, enc
        except Exception as e:
            last_err = e
            continue
    raise last_err or UnicodeDecodeError("utf-8", b"", 0, 1, "encoding detection failed")

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    # Accept any of these as the primary series name
    if "D(CS �)" in df.columns and "D" not in df.columns:
        df = df.rename(columns={"D(CS �)": "D"})
    if "D(CS £)" in df.columns and "D" not in df.columns:
        df = df.rename(columns={"D(CS £)": "D"})
    # Require Date
    if "Date" not in df.columns:
        raise ValueError("CSV must include a 'Date' column.")
    # Parse ISO dates (your sample is YYYY-MM-DD)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    # Coerce numerics
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str):
    df_raw, enc = _read_with_encodings(path, is_bytes=False)
    return _normalize_df(df_raw), enc

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(b: bytes):
    df_raw, enc = _read_with_encodings(b, is_bytes=True)
    return _normalize_df(df_raw), enc

def melt_for_plot(df: pd.DataFrame, chosen_cols: list[str]) -> pd.DataFrame:
    return (
        df.melt(id_vars="Date", value_vars=chosen_cols, var_name="Series", value_name="Value")
          .dropna(subset=["Value"])
    )

# ---------- Clean Spark ----------
def render_clean_spark():
    st.markdown(
        """
        <style>
        .spark-hero{padding:1.25rem 1.5rem;border-radius:1rem;
        background:radial-gradient(1200px 600px at 10% -20%, rgba(255,230,120,.25), transparent),
                   linear-gradient(135deg,#fff 0%,#f7fafc 100%);
        border:1px solid rgba(0,0,0,.05);}
        .spark-badge{display:inline-block;padding:.25rem .6rem;border-radius:999px;
        background:#fff7cc;border:1px solid #ffec80;font-size:.80rem;font-weight:600;color:#7a5b00;}
        .spark-subtle{color:#666;font-size:.95rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    #st.markdown(
     #   """
      #  <div class="spark-hero">
       #   <span class="spark-badge">Clean Spark</span>
        #  <h1 style="margin:.4rem 0 0 0;">⚡ Plot your css.csv by date</h1>
         # <p class="spark-subtle">Handles non-UTF-8 CSV encodings and missing cells.</p>
        #</div>
        #""",
     #   unsafe_allow_html=True,
    #)

    with st.sidebar:
        st.subheader("Data Source")
        default_choice = "css.csv (same folder)" if os.path.exists("css.csv") else "Upload CSV"
        source = st.radio("Choose source", ["css.csv (same folder)", "Upload CSV"],
                          index=0 if default_choice == "css.csv (same folder)" else 1)
        uploaded = None
        if source == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=["csv"])

        st.subheader("Options")
        drop_all_nan = st.checkbox("Drop rows where all series are empty", value=True)
        show_markers = st.checkbox("Show markers", value=False)
        show_table = st.checkbox("Show data table", value=False)

    df = None
    used_encoding = None
    try:
        if source == "css.csv (same folder)":
            if os.path.exists("css.csv"):
                df, used_encoding = load_csv_from_path("css.csv")
            else:
                st.warning("No css.csv found. Switch to 'Upload CSV' or place css.csv next to app.py.")
        else:
            if uploaded is not None:
                df, used_encoding = load_csv_from_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

    if df is None or df.empty:
        st.info("Load a CSV to see the chart here.")
        st.caption("Tip: If you can, save the file as UTF-8. This app auto-tries common encodings anyway.")
        return

    numeric_cols = [c for c in df.columns if c != "Date"]
    if drop_all_nan and numeric_cols:
        df = df.loc[~df[numeric_cols].isna().all(axis=1)].reset_index(drop=True)

    st.success(f"CSV loaded (encoding: {used_encoding}).")

    # Choose series to plot + primary highlight
    chosen = st.multiselect("Series to plot", options=numeric_cols, default=numeric_cols)
    primary_guess = "D" if "D" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
    primary = st.selectbox("Highlight series", options=chosen if chosen else numeric_cols, index=0 if primary_guess in chosen else 0) if numeric_cols else None

    if not chosen:
        st.info("Select at least one series to plot.")
        return

    plot_df = melt_for_plot(df, chosen)
    fig = px.line(plot_df, x="Date", y="Value", color="Series", template="simple_white",
                  title="Clean Spark", labels = {"Date": "Date", "Value": "£MWh", "Series": "Days"})
    fig.update_traces(line=dict(width=2.0), mode="lines+markers" if show_markers else "lines")
    if primary:
        for tr in fig.data:
            if tr.name == primary:
                tr.update(line=dict(width=3.2))
    fig.update_layout(margin=dict(t=60, r=20, l=10, b=10), legend_title_text="Series")
    st.plotly_chart(fig, use_container_width=True)

    # KPIs for primary
    if primary in df.columns:
        series = df[primary].dropna()
        latest = series.iloc[-1] if not series.empty else np.nan
        prev = series.iloc[-2] if len(series) > 1 else latest
        delta = latest - prev if pd.notna(latest) and pd.notna(prev) else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Latest {primary}", f"{latest:.3f}" if pd.notna(latest) else "—",
                  f"{delta:+.3f}" if pd.notna(delta) else None)
        c2.metric(f"Mean {primary}", f"{df[primary].mean():.3f}" if df[primary].notna().any() else "—")
        c3.metric(f"Std Dev {primary}", f"{df[primary].std():.3f}" if df[primary].notna().any() else "—")

    if show_table:
        st.dataframe(df, use_container_width=True)

    st.caption("Fix the table ---- missing cells will be ignored for now.")

# ---------- Default ----------
def render_default():
    st.title("Default View")
    st.write("A simple placeholder page.")
    with st.sidebar:
        st.subheader("Default Controls")
        n = st.slider("Sample size", 100, 5000, 1000, 100)
        bins = st.slider("Histogram bins", 5, 100, 30, 5)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, n)})
    st.subheader("Quick Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{df['x'].mean():.3f}")
    c2.metric("Std Dev", f"{df['x'].std():.3f}")
    c3.metric("Samples", f"{len(df)}")
    st.subheader("Distribution")
    hist = px.histogram(df, x="x", nbins=bins, template="simple_white", title="Normal(0, 1) Sample")
    st.plotly_chart(hist, use_container_width=True)
    with st.expander("Peek at the data"):
        st.dataframe(df.head(20), use_container_width=True)

# ---------- Router ----------
if view == "Clean Spark":
    render_clean_spark()
else:
    render_default()

st.write("---")
st.caption(" Switch views from the sidebar.")
