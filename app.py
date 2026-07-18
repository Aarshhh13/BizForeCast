"""
app.py - BizForeCast Streamlit Dashboard (UPGRADED)
=====================================================
Changes in this version:
  - User CSV upload with format validation & preview
  - Manual data entry form (add sales rows one by one)
  - Custom .streamlit/config.toml for professional theme
  - Polished UI with metric cards, badges, dividers
  - Stock alert system
  - Multi-product comparison chart
  - Download forecast as CSV
  - Annotated forecast chart with confidence band
  - Fully responsive sidebar controls
"""

import pickle, os, io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="BizForeCast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — clean, professional look ─────────────────────────────────────
st.markdown("""
<style>
/* Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default Streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* Custom top banner */
.top-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 0 0 16px 16px;
    margin: -1rem -1rem 2rem -1rem;
}
.banner-title {
    font-size: 2rem; font-weight: 700;
    color: #ffffff; margin: 0; letter-spacing: -0.5px;
}
.banner-sub {
    font-size: 0.9rem; color: #a0aec0; margin-top: 4px;
}
.banner-badges { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
.badge {
    font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; letter-spacing: 0.03em;
}
.badge-blue  { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.badge-green { background: rgba(72,187,120,0.15); color: #48bb78; border: 1px solid rgba(72,187,120,0.3); }
.badge-purple{ background: rgba(159,122,234,0.15); color: #9f7aea; border: 1px solid rgba(159,122,234,0.3); }

/* KPI metric cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1.5rem 0; }
.kpi-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-label { font-size: 11px; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #1a202c; line-height: 1; }
.kpi-delta { font-size: 12px; margin-top: 5px; }
.kpi-delta.up   { color: #38a169; }
.kpi-delta.down { color: #e53e3e; }
.kpi-delta.neu  { color: #718096; }

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 2rem 0 1rem;
}
.section-title { font-size: 1.05rem; font-weight: 600; color: #1a202c; }
.section-line  { flex: 1; height: 1px; background: #e2e8f0; }

/* Alert boxes */
.alert {
    border-radius: 10px; padding: 12px 16px;
    font-size: 13px; font-weight: 500; margin: 6px 0;
    display: flex; align-items: center; gap: 10px;
}
.alert-red    { background: #fff5f5; border: 1px solid #fed7d7; color: #c53030; }
.alert-yellow { background: #fffff0; border: 1px solid #faf089; color: #975a16; }
.alert-green  { background: #f0fff4; border: 1px solid #c6f6d5; color: #276749; }

/* Upload zone */
.upload-info {
    background: #f7fafc; border: 2px dashed #cbd5e0; border-radius: 12px;
    padding: 1.25rem; text-align: center; color: #718096; font-size: 13px; margin: 8px 0;
}

/* Table style */
.styled-table { width:100%; border-collapse:collapse; font-size:13px; }
.styled-table th { background:#f7fafc; color:#4a5568; font-weight:600; padding:8px 12px; border-bottom:2px solid #e2e8f0; text-align:left; }
.styled-table td { padding:8px 12px; border-bottom:1px solid #f0f0f0; color:#2d3748; }
.styled-table tr:hover td { background:#f7fafc; }

/* Step pills */
.steps-row { display:flex; gap:12px; margin:12px 0; flex-wrap:wrap; }
.step-pill {
    display:flex; align-items:center; gap:8px;
    background:#ebf8ff; border:1px solid #bee3f8;
    border-radius:8px; padding:8px 14px;
    font-size:12px; color:#2b6cb0; font-weight:500;
}
.step-num {
    width:20px; height:20px; border-radius:50%; background:#3182ce;
    color:#fff; font-size:11px; font-weight:700;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_model_data():
    if not os.path.exists("models/trained_model.pkl"):
        return None
    with open("models/trained_model.pkl", "rb") as f:
        return pickle.load(f)


def validate_csv(df):
    """
    Check that uploaded CSV has the required columns.
    Returns (is_valid: bool, message: str)
    """
    required = {"date", "product_id", "sales"}
    missing  = required - set(df.columns.str.lower())
    if missing:
        return False, f"Missing columns: {', '.join(missing)}. Your CSV needs: date, product_id, sales"
    return True, "OK"


def preprocess_user_df(df):
    """Clean and validate user-uploaded or manually entered data."""
    df.columns = df.columns.str.lower().str.strip()
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])
    df["sales"] = df["sales"].clip(lower=0)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return df


def simple_forecast(history_df, n_days=30):
    """
    Lightweight forecast that works WITHOUT a pre-trained model.
    Uses 30-day rolling average + day-of-week factors.
    Perfect for user-uploaded data where we haven't retrained.
    """
    results = []
    for pid, grp in history_df.groupby("product_id"):
        grp = grp.sort_values("date")
        base = grp["sales"].tail(30).mean()
        dow_avg = grp.copy()
        dow_avg["dow"] = dow_avg["date"].dt.dayofweek
        dow_factors = dow_avg.groupby("dow")["sales"].mean()
        global_mean = dow_avg["sales"].mean() or 1

        last_date = grp["date"].max()
        for i in range(1, n_days + 1):
            future_date = last_date + timedelta(days=i)
            dow = future_date.weekday()
            factor = dow_factors.get(dow, global_mean) / global_mean if global_mean > 0 else 1
            noise  = np.random.normal(0, base * 0.05)
            pred   = max(0, base * factor + noise)
            results.append({
                "date": future_date,
                "product_id": pid,
                "predicted_sales": round(pred, 1)
            })
    return pd.DataFrame(results)


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# TOP BANNER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="top-banner">
  <div class="banner-title">📈 BizForeCast</div>
  <div class="banner-sub">Small Business Sales Demand Forecasting System</div>
  <div class="banner-badges">
    <span class="badge badge-blue">XGBoost</span>
    <span class="badge badge-blue">LightGBM</span>
    <span class="badge badge-green">Live Forecasting</span>
    <span class="badge badge-purple">ML-Powered</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")

    # ── DATA SOURCE SELECTOR ─────────────────────────────────────────────────
    st.markdown("**Data source**")
    data_source = st.radio(
        "Choose how to load data",
        ["📁 Upload my CSV", "✏️ Enter data manually", "🎲 Use sample data"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    # ── USER CSV UPLOAD ──────────────────────────────────────────────────────
    user_df = None

    if data_source == "📁 Upload my CSV":
        st.markdown("**Upload sales CSV**")
        st.markdown("""
        <div class="upload-info">
          Your CSV must have these 3 columns:<br>
          <b>date</b> &nbsp;|&nbsp; <b>product_id</b> &nbsp;|&nbsp; <b>sales</b>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop CSV here", type=["csv"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                valid, msg = validate_csv(raw_df)
                if not valid:
                    st.error(f"❌ {msg}")
                else:
                    user_df = preprocess_user_df(raw_df)
                    st.success(f"✅ {len(user_df):,} rows loaded!")
                    with st.expander("Preview your data"):
                        st.dataframe(user_df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.info("Waiting for file upload…")

    # ── MANUAL DATA ENTRY ────────────────────────────────────────────────────
    elif data_source == "✏️ Enter data manually":
        st.markdown("**Add sales records**")

        if "manual_rows" not in st.session_state:
            st.session_state.manual_rows = []

        with st.form("add_row_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                entry_date = st.date_input("Date", value=datetime.today())
                product_name = st.text_input("Product name", placeholder="e.g. Product_A")
            with col2:
                sales_qty = st.number_input("Sales (units)", min_value=0.0, step=1.0)

            submitted = st.form_submit_button("➕ Add row", use_container_width=True)
            if submitted:
                if product_name.strip():
                    st.session_state.manual_rows.append({
                        "date": str(entry_date),
                        "product_id": product_name.strip(),
                        "sales": sales_qty
                    })
                    st.success(f"Added: {product_name} | {entry_date} | {sales_qty} units")
                else:
                    st.warning("Please enter a product name.")

        if st.session_state.manual_rows:
            user_df = preprocess_user_df(
                pd.DataFrame(st.session_state.manual_rows)
            )
            st.info(f"📋 {len(user_df)} rows entered")

            if st.button("🗑️ Clear all rows"):
                st.session_state.manual_rows = []
                st.rerun()

            with st.expander("View entered data"):
                st.dataframe(user_df, use_container_width=True)
        else:
            st.caption("No rows yet — use the form above.")

    st.markdown("---")

    # ── FORECAST CONTROLS ────────────────────────────────────────────────────
    st.markdown("**Forecast settings**")
    forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30, step=7)

    # Stock alert threshold
    st.markdown("---")
    st.markdown("**Stock alert threshold**")
    stock_threshold = st.number_input(
        "Flag if 7-day demand exceeds (units)",
        min_value=0, value=700, step=50,
        help="Alert fires when predicted 7-day demand > this number"
    )

    st.markdown("---")
    st.caption("BizForeCast · XGBoost + LightGBM")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

model_data   = load_model_data()
using_model  = (model_data is not None)
using_upload = (user_df is not None)

# Decide which dataframe to use
if using_upload:
    all_history = user_df
    source_label = "your uploaded data"
elif using_model:
    train_df = model_data["train_df"]
    test_df  = model_data["test_df"]
    all_history  = pd.concat([train_df, test_df]).sort_values(["product_id","date"])
    source_label = "sample data"
else:
    st.warning("⚠️ No trained model found AND no data uploaded. Run `python train_model.py` first, or upload a CSV.")
    st.stop()

products = sorted(all_history["product_id"].unique())

# Sidebar product selectors (below data is ready)
with st.sidebar:
    st.markdown("**Products**")
    selected_product = st.selectbox("Primary product", products)
    compare_products = st.multiselect(
        "Compare products", products,
        default=products[:min(2, len(products))],
        help="Select multiple products to compare forecasts"
    )


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE FORECAST
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)

if using_model and not using_upload:
    # Use the trained ML model
    from forecast import forecast_all_products
    @st.cache_data
    def get_ml_forecast(n):
        return forecast_all_products(n_days=n)
    forecast_df = get_ml_forecast(forecast_days)
else:
    # Use the lightweight rolling-average forecast for user data
    @st.cache_data
    def get_simple_forecast(df_json, n):
        df = pd.read_json(io.StringIO(df_json))
        df["date"] = pd.to_datetime(df["date"])
        return simple_forecast(df, n_days=n)
    forecast_df = get_simple_forecast(all_history.to_json(), forecast_days)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

product_history = all_history[all_history["product_id"] == selected_product].copy()
product_forecast = forecast_df[forecast_df["product_id"] == selected_product].copy()

total_units   = int(all_history["sales"].sum())
avg_daily     = all_history["sales"].mean()
best_product  = all_history.groupby("product_id")["sales"].sum().idxmax()
pred_next7    = product_forecast.head(7)["predicted_sales"].sum()
hist_last7    = product_history["sales"].tail(7).sum()
delta_pct     = ((pred_next7 - hist_last7) / hist_last7 * 100) if hist_last7 > 0 else 0
model_label   = model_data["model_name"] if using_model else "Rolling Avg"
accuracy_val  = f"{100 - model_data['metrics']['best']['mape']:.1f}%" if using_model else "N/A"

st.markdown(f"""
<div class="section-header">
  <span class="section-title">📊 Overview</span>
  <span class="section-line"></span>
  <span style="font-size:12px;color:#718096;">Source: {source_label} &nbsp;|&nbsp; Model: {model_label}</span>
</div>
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Total units sold</div>
    <div class="kpi-value">{total_units:,}</div>
    <div class="kpi-delta neu">All products · all time</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg daily sales</div>
    <div class="kpi-value">{avg_daily:.0f}</div>
    <div class="kpi-delta neu">units / day</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">7-day forecast ({selected_product})</div>
    <div class="kpi-value">{pred_next7:.0f}</div>
    <div class="kpi-delta {'up' if delta_pct>=0 else 'down'}">
      {'▲' if delta_pct>=0 else '▼'} {abs(delta_pct):.1f}% vs last 7 days
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Model accuracy</div>
    <div class="kpi-value">{accuracy_val}</div>
    <div class="kpi-delta neu">by MAPE on test set</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STOCK ALERTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="section-header">
  <span class="section-title">🚨 Stock Alerts</span>
  <span class="section-line"></span>
</div>
""", unsafe_allow_html=True)

alert_cols = st.columns(len(products))
for i, pid in enumerate(products):
    pf7 = forecast_df[forecast_df["product_id"] == pid].head(7)["predicted_sales"].sum()
    with alert_cols[i % len(alert_cols)]:
        if pf7 > stock_threshold:
            st.markdown(f"""
            <div class="alert alert-red">
              🔴 <b>{pid}</b><br>
              7-day demand: <b>{pf7:.0f}</b> units<br>
              Exceeds threshold of {stock_threshold}
            </div>""", unsafe_allow_html=True)
        elif pf7 > stock_threshold * 0.8:
            st.markdown(f"""
            <div class="alert alert-yellow">
              🟡 <b>{pid}</b><br>
              7-day demand: <b>{pf7:.0f}</b> units<br>
              Approaching threshold ({stock_threshold})
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert alert-green">
              🟢 <b>{pid}</b><br>
              7-day demand: <b>{pf7:.0f}</b> units<br>
              Stock levels OK
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SALES TREND
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="section-header">
  <span class="section-title">📅 Sales Trend — {selected_product}</span>
  <span class="section-line"></span>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    ph = product_history.copy()
    ph["7-day avg"]  = ph["sales"].rolling(7,  min_periods=1).mean()
    ph["30-day avg"] = ph["sales"].rolling(30, min_periods=1).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=ph["date"], y=ph["sales"],
        name="Daily sales", mode="lines",
        line=dict(color="#cbd5e0", width=1), opacity=0.7
    ))
    fig_trend.add_trace(go.Scatter(
        x=ph["date"], y=ph["7-day avg"],
        name="7-day avg", line=dict(color="#4299e1", width=2)
    ))
    fig_trend.add_trace(go.Scatter(
        x=ph["date"], y=ph["30-day avg"],
        name="30-day avg", line=dict(color="#ed8936", width=2, dash="dash")
    ))
    fig_trend.update_layout(
        title=None, hovermode="x unified", height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(showgrid=False, color="#718096"),
        yaxis=dict(gridcolor="#f0f0f0", color="#718096"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    # Day-of-week bar chart
    ph2 = product_history.copy()
    ph2["day"] = pd.to_datetime(ph2["date"]).dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_data  = ph2.groupby("day")["sales"].mean().reindex(dow_order).reset_index()
    dow_data.columns = ["day","avg_sales"]
    max_day = dow_data.loc[dow_data["avg_sales"].idxmax(), "day"]
    colors  = ["#4299e1" if d == max_day else "#bee3f8" for d in dow_data["day"]]

    fig_dow = go.Figure(go.Bar(
        x=dow_data["day"].str[:3], y=dow_data["avg_sales"],
        marker_color=colors,
        text=dow_data["avg_sales"].round(0).astype(int),
        textposition="outside"
    ))
    fig_dow.update_layout(
        title=dict(text="Avg by day", font=dict(size=13)),
        height=300, margin=dict(l=0,r=0,t=30,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(showgrid=False, visible=False),
        xaxis=dict(showgrid=False, color="#718096"),
        showlegend=False,
    )
    st.plotly_chart(fig_dow, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="section-header">
  <span class="section-title">🔮 {forecast_days}-Day Demand Forecast</span>
  <span class="section-line"></span>
</div>
""", unsafe_allow_html=True)

recent_hist = product_history.tail(60)

fig_fc = go.Figure()

# Confidence band
fig_fc.add_trace(go.Scatter(
    x=list(product_forecast["date"]) + list(product_forecast["date"][::-1]),
    y=list(product_forecast["predicted_sales"] * 1.15) +
      list(product_forecast["predicted_sales"][::-1] * 0.85),
    fill="toself", fillcolor="rgba(66,153,225,0.08)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Confidence range", hoverinfo="skip"
))
# Historical line
fig_fc.add_trace(go.Scatter(
    x=recent_hist["date"], y=recent_hist["sales"],
    name="Historical", line=dict(color="#a0aec0", width=2), mode="lines"
))
# Forecast line
fig_fc.add_trace(go.Scatter(
    x=product_forecast["date"], y=product_forecast["predicted_sales"],
    name="Forecast", line=dict(color="#4299e1", width=2.5, dash="dash"),
    mode="lines+markers", marker=dict(size=4)
))
# Vertical "today" line
today = all_history["date"].max()
fig_fc.add_vline(x=today, line_dash="dot", line_color="#e53e3e",
                  annotation_text="Last data", annotation_position="top right",
                  annotation_font_size=11, annotation_font_color="#e53e3e")

fig_fc.update_layout(
    hovermode="x unified", height=340,
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=-0.15),
    xaxis=dict(showgrid=False, color="#718096"),
    yaxis=dict(gridcolor="#f0f0f0", color="#718096", title="Units"),
)
st.plotly_chart(fig_fc, use_container_width=True)

# Forecast summary row
fc_total = product_forecast["predicted_sales"].sum()
fc_avg   = product_forecast["predicted_sales"].mean()
fc_peak  = product_forecast["predicted_sales"].max()
fc_peak_date = product_forecast.loc[product_forecast["predicted_sales"].idxmax(), "date"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total predicted units", f"{fc_total:,.0f}")
m2.metric("Daily average", f"{fc_avg:.1f}")
m3.metric("Peak day", f"{fc_peak:.0f} units")
m4.metric("Peak date", pd.to_datetime(fc_peak_date).strftime("%b %d"))

# Download button
csv_data = convert_df_to_csv(product_forecast.rename(columns={
    "date":"Date","predicted_sales":"Predicted Sales","product_id":"Product"
}))
st.download_button(
    label="⬇️ Download forecast as CSV",
    data=csv_data,
    file_name=f"{selected_product}_forecast_{forecast_days}d.csv",
    mime="text/csv",
)

with st.expander("View full forecast table"):
    display_fc = product_forecast.copy()
    display_fc["date"] = display_fc["date"].dt.strftime("%Y-%m-%d")
    display_fc["predicted_sales"] = display_fc["predicted_sales"].round(1)
    display_fc.columns = ["Date", "Predicted Sales", "Product"]
    st.dataframe(display_fc[["Date","Predicted Sales"]], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MULTI-PRODUCT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

if compare_products:
    st.markdown("""
    <div class="section-header">
      <span class="section-title">🔄 Product Comparison</span>
      <span class="section-line"></span>
    </div>
    """, unsafe_allow_html=True)

    palette = ["#4299e1","#48bb78","#ed8936","#9f7aea","#f56565","#38b2ac"]
    fig_cmp = go.Figure()
    for i, pid in enumerate(compare_products):
        pf_c = forecast_df[forecast_df["product_id"] == pid]
        fig_cmp.add_trace(go.Scatter(
            x=pf_c["date"], y=pf_c["predicted_sales"],
            name=pid, mode="lines",
            line=dict(color=palette[i % len(palette)], width=2.5)
        ))
    fig_cmp.update_layout(
        hovermode="x unified", height=320,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=-0.18),
        xaxis=dict(showgrid=False, color="#718096"),
        yaxis=dict(gridcolor="#f0f0f0", color="#718096", title="Predicted units"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Comparison summary table
    cmp_rows = []
    for pid in compare_products:
        pf_c = forecast_df[forecast_df["product_id"] == pid]
        cmp_rows.append({
            "Product": pid,
            "7-day total": f"{pf_c.head(7)['predicted_sales'].sum():,.0f}",
            "30-day total": f"{pf_c.head(30)['predicted_sales'].sum():,.0f}",
            "Daily avg": f"{pf_c['predicted_sales'].mean():.1f}",
            "Peak demand": f"{pf_c['predicted_sales'].max():.0f}",
        })
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL PERFORMANCE (only if ML model is loaded)
# ══════════════════════════════════════════════════════════════════════════════

if using_model and not using_upload:
    st.markdown("""
    <div class="section-header">
      <span class="section-title">🎯 Model Performance</span>
      <span class="section-line"></span>
    </div>
    """, unsafe_allow_html=True)

    test_results = model_data["test_results"]
    perf_df = pd.DataFrame({
        "date":       pd.to_datetime(test_results["dates"]),
        "actual":     test_results["actual"],
        "predicted":  test_results["best_pred"],
        "product_id": test_results["product_id"],
    })
    prod_perf = perf_df[perf_df["product_id"] == selected_product]

    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        if len(prod_perf) > 0:
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=prod_perf["date"], y=prod_perf["actual"],
                name="Actual", line=dict(color="#4299e1", width=2)
            ))
            fig_perf.add_trace(go.Scatter(
                x=prod_perf["date"], y=prod_perf["predicted"],
                name="Predicted", line=dict(color="#ed8936", width=2, dash="dot")
            ))
            fig_perf.update_layout(
                hovermode="x unified", height=280,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(orientation="h", y=-0.2),
                xaxis=dict(showgrid=False, color="#718096"),
                yaxis=dict(gridcolor="#f0f0f0", color="#718096"),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

    with col_p2:
        metrics = model_data["metrics"]
        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:10px">
          <div class="kpi-label">XGBoost RMSE</div>
          <div class="kpi-value" style="font-size:1.3rem">{metrics['xgboost']['rmse']:.1f}</div>
        </div>
        <div class="kpi-card" style="margin-bottom:10px">
          <div class="kpi-label">LightGBM RMSE</div>
          <div class="kpi-value" style="font-size:1.3rem">{metrics['lightgbm']['rmse']:.1f}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Best model</div>
          <div class="kpi-value" style="font-size:1.1rem">{metrics['best_model']}</div>
          <div class="kpi-delta green">MAPE: {metrics['best']['mape']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance
    fi = model_data.get("feature_importance", {})
    if fi:
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"])
        fi_df  = fi_df.sort_values("Importance", ascending=True).tail(12)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker_color=["#4299e1" if i >= len(fi_df)-3 else "#bee3f8"
                          for i in range(len(fi_df))]
        ))
        fig_fi.update_layout(
            title=dict(text=f"Feature importance — {model_data['model_name']}", font=dict(size=13)),
            height=340, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=0,r=0,t=35,b=0),
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, color="#4a5568", tickfont=dict(size=12)),
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#a0aec0;font-size:12px'>"
    "BizForeCast &nbsp;·&nbsp; Built with Python, XGBoost, LightGBM & Streamlit"
    "</p>",
    unsafe_allow_html=True
)