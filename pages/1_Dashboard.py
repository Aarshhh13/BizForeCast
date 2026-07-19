"""
pages/1_Dashboard.py  —  BizForeCast main dashboard
"""
import io, os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import inject_css, navbar, section, load_model, validate_csv, clean_df, simple_forecast, to_csv_bytes, C

st.set_page_config(page_title="Dashboard · BizForeCast", page_icon="📊", layout="wide")
inject_css()
navbar("Dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — data source + controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Data Source")
    source = st.radio("", ["📁 Upload CSV","✏️ Manual entry","🎲 Sample data"], label_visibility="collapsed")
    st.markdown("---")

    user_df = None

    # ── CSV upload ─────────────────────────────────────────────────────────
    if source == "📁 Upload CSV":
        st.markdown("**Your CSV must have:** `date`, `product_id`, `sales`")
        f = st.file_uploader("Drop file here", type=["csv"], label_visibility="collapsed")
        if f:
            try:
                raw = pd.read_csv(f)
                ok, msg = validate_csv(raw)
                if not ok:
                    st.error(f"❌ {msg}")
                else:
                    user_df = clean_df(raw)
                    st.success(f"✅ {len(user_df):,} rows loaded")
                    with st.expander("Preview"):
                        st.dataframe(user_df.head(8), use_container_width=True)
            except Exception as e:
                st.error(f"Read error: {e}")
        else:
            st.markdown("""
            <div style="border:2px dashed #cbd5e0;border-radius:10px;padding:1rem;
                        text-align:center;color:#718096;font-size:12px;">
              Drag &amp; drop a CSV file here
            </div>""", unsafe_allow_html=True)

    # ── Manual entry ───────────────────────────────────────────────────────
    elif source == "✏️ Manual entry":
        if "rows" not in st.session_state:
            st.session_state.rows = []
        with st.form("rf", clear_on_submit=True):
            d  = st.date_input("Date", value=datetime.today())
            p  = st.text_input("Product name", placeholder="e.g. Widget_A")
            s  = st.number_input("Sales (units)", min_value=0.0, step=1.0)
            if st.form_submit_button("➕ Add row", use_container_width=True):
                if p.strip():
                    st.session_state.rows.append({"date":str(d),"product_id":p.strip(),"sales":s})
                else:
                    st.warning("Enter a product name")
        if st.session_state.rows:
            user_df = clean_df(pd.DataFrame(st.session_state.rows))
            st.info(f"{len(user_df)} rows")
            if st.button("🗑️ Clear all"):
                st.session_state.rows = []
                st.rerun()

    st.markdown("---")
    st.markdown("### 📅 Forecast settings")
    forecast_days   = st.slider("Horizon (days)", 7, 90, 30, step=7)
    stock_threshold = st.number_input("Stock alert if 7-day demand >", min_value=0, value=700, step=50)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
model_data   = load_model()
using_model  = model_data is not None
using_upload = user_df is not None

if using_upload:
    all_hist = user_df
    src_label = "your data"
elif using_model:
    all_hist  = pd.concat([model_data["train_df"], model_data["test_df"]]).sort_values(["product_id","date"])
    src_label = "sample data"
else:
    st.error("No model found and no data uploaded. Run `python train_model.py` first, or upload a CSV.")
    st.stop()

products = sorted(all_hist["product_id"].unique())

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 Products")
    sel  = st.selectbox("Primary product", products)
    cmp  = st.multiselect("Compare products", products, default=products[:min(2,len(products))])

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
if using_model and not using_upload:
    try:
        from forecast import forecast_all_products
        @st.cache_data
        def _ml_fc(n): return forecast_all_products(n_days=n)
        fc_df = _ml_fc(forecast_days)
    except Exception:
        fc_df = simple_forecast(all_hist, forecast_days)
else:
    @st.cache_data
    def _sf(j, n):
        df = pd.read_json(io.StringIO(j))
        df["date"] = pd.to_datetime(df["date"])
        return simple_forecast(df, n)
    fc_df = _sf(all_hist.to_json(), forecast_days)

prod_hist = all_hist[all_hist["product_id"]==sel].copy()
prod_fc   = fc_df[fc_df["product_id"]==sel].copy()

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
section(f"📊 Overview — {sel}", badge=src_label)

total   = int(all_hist["sales"].sum())
avg_d   = all_hist["sales"].mean()
fc7     = prod_fc.head(7)["predicted_sales"].sum()
hist7   = prod_hist["sales"].tail(7).sum()
delta   = (fc7-hist7)/hist7*100 if hist7 else 0
acc     = f"{100-model_data['metrics']['best']['mape']:.1f}%" if using_model else "N/A"
ml_name = model_data["model_name"] if using_model else "Rolling Avg"

arrow   = "▲" if delta>=0 else "▼"
dcls    = "ku" if delta>=0 else "kd"

st.markdown(f"""
<div class="kg">
  <div class="kc">
    <div class="kl">Total units sold</div>
    <div class="kv">{total:,}</div>
    <div class="ks">All products · all time</div>
  </div>
  <div class="kc">
    <div class="kl">Avg daily sales</div>
    <div class="kv">{avg_d:.0f}</div>
    <div class="ks">units per day</div>
  </div>
  <div class="kc">
    <div class="kl">7-day forecast</div>
    <div class="kv">{fc7:.0f}</div>
    <div class="ks {dcls}">{arrow} {abs(delta):.1f}% vs last 7 days</div>
  </div>
  <div class="kc">
    <div class="kl">Model accuracy</div>
    <div class="kv">{acc}</div>
    <div class="ks">Active: {ml_name}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STOCK ALERTS
# ══════════════════════════════════════════════════════════════════════════════
section("🚨 Stock Alerts")
acols = st.columns(len(products))
for i, pid in enumerate(products):
    d7 = fc_df[fc_df["product_id"]==pid].head(7)["predicted_sales"].sum()
    if d7 > stock_threshold:
        cls, sym = "alr", "🔴"
        msg = f"Exceeds threshold of {stock_threshold:,}"
    elif d7 > stock_threshold * .8:
        cls, sym = "aly", "🟡"
        msg = f"Approaching threshold ({stock_threshold:,})"
    else:
        cls, sym = "alg", "🟢"
        msg = "Stock levels sufficient"
    with acols[i % len(acols)]:
        st.markdown(f"""
        <div class="al {cls}">
          {sym} <strong>{pid}</strong><br>
          7-day demand: <strong>{d7:.0f}</strong> units<br>
          {msg}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SALES TREND
# ══════════════════════════════════════════════════════════════════════════════
section(f"📅 Sales Trend — {sel}")
cl, cr = st.columns([2,1])

with cl:
    ph = prod_hist.copy()
    ph["7d"]  = ph["sales"].rolling(7,  min_periods=1).mean()
    ph["30d"] = ph["sales"].rolling(30, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ph["date"], y=ph["sales"],    name="Daily sales",
                              line=dict(color="#cbd5e0",width=1), opacity=.7))
    fig.add_trace(go.Scatter(x=ph["date"], y=ph["7d"],       name="7-day avg",
                              line=dict(color="#4361ee",width=2)))
    fig.add_trace(go.Scatter(x=ph["date"], y=ph["30d"],      name="30-day avg",
                              line=dict(color="#f72585",width=2,dash="dash")))
    fig.update_layout(hovermode="x unified",height=290,plot_bgcolor="#fff",paper_bgcolor="#fff",
                       margin=dict(l=0,r=0,t=6,b=0),
                       legend=dict(orientation="h",y=-0.22,font=dict(size=11)),
                       xaxis=dict(showgrid=False,color="#718096"),
                       yaxis=dict(gridcolor="#f0f0f0",color="#718096"))
    st.plotly_chart(fig, use_container_width=True)

with cr:
    ph2 = prod_hist.copy()
    ph2["day"] = pd.to_datetime(ph2["date"]).dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow   = ph2.groupby("day")["sales"].mean().reindex(order).reset_index()
    mx    = dow.loc[dow["sales"].idxmax(),"day"]
    colors= ["#4361ee" if d==mx else "#c3d0ff" for d in dow["day"]]
    fig2  = go.Figure(go.Bar(x=dow["day"].str[:3], y=dow["sales"],
                              marker_color=colors,
                              text=dow["sales"].round(0).astype(int),
                              textposition="outside"))
    fig2.update_layout(title="Avg by weekday",height=290,plot_bgcolor="#fff",paper_bgcolor="#fff",
                        margin=dict(l=0,r=0,t=30,b=0),showlegend=False,
                        xaxis=dict(showgrid=False,color="#718096"),
                        yaxis=dict(visible=False))
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════════════════
section(f"🔮 {forecast_days}-Day Demand Forecast")

rec = prod_hist.tail(60)
fig3 = go.Figure()
# Confidence band
fig3.add_trace(go.Scatter(
    x=list(prod_fc["date"])+list(prod_fc["date"][::-1]),
    y=list(prod_fc["predicted_sales"]*1.15)+list(prod_fc["predicted_sales"][::-1]*.85),
    fill="toself", fillcolor="rgba(67,97,238,.08)", line=dict(color="rgba(0,0,0,0)"),
    name="Confidence range", hoverinfo="skip"))
fig3.add_trace(go.Scatter(x=rec["date"], y=rec["sales"], name="Historical",
                           line=dict(color="#a0aec0",width=2)))
fig3.add_trace(go.Scatter(x=prod_fc["date"], y=prod_fc["predicted_sales"], name="Forecast",
                           line=dict(color="#4361ee",width=2.5,dash="dash"),
                           mode="lines+markers", marker=dict(size=4)))
fig3.add_vline(x=all_hist["date"].max(), line_dash="dot", line_color="#ef233c",
                annotation_text="Last data", annotation_font_size=11,
                annotation_font_color="#ef233c", annotation_position="top right")
fig3.update_layout(hovermode="x unified",height=340,plot_bgcolor="#fff",paper_bgcolor="#fff",
                    margin=dict(l=0,r=0,t=6,b=0),
                    legend=dict(orientation="h",y=-0.15,font=dict(size=11)),
                    xaxis=dict(showgrid=False,color="#718096"),
                    yaxis=dict(gridcolor="#f0f0f0",color="#718096",title="Units"))
st.plotly_chart(fig3, use_container_width=True)

# Summary + download
fc_tot   = prod_fc["predicted_sales"].sum()
fc_avg   = prod_fc["predicted_sales"].mean()
fc_peak  = prod_fc["predicted_sales"].max()
peak_dt  = pd.to_datetime(prod_fc.loc[prod_fc["predicted_sales"].idxmax(),"date"]).strftime("%b %d")

m1,m2,m3,m4 = st.columns(4)
m1.metric("Total predicted", f"{fc_tot:,.0f} units")
m2.metric("Daily average",   f"{fc_avg:.1f} units")
m3.metric("Peak demand",     f"{fc_peak:.0f} units")
m4.metric("Peak date",       peak_dt)

st.download_button("⬇️ Download forecast CSV",
                    data=to_csv_bytes(prod_fc.rename(columns={"date":"Date","predicted_sales":"Predicted Sales","product_id":"Product"})),
                    file_name=f"{sel}_forecast_{forecast_days}d.csv", mime="text/csv")

with st.expander("View full forecast table"):
    t = prod_fc.copy()
    t["date"] = t["date"].dt.strftime("%Y-%m-%d")
    t.columns = ["Date","Product","Predicted Sales"]
    st.dataframe(t[["Date","Predicted Sales"]], use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
if cmp:
    section("🔄 Product Comparison")
    pal = C["chart"]
    fig4 = go.Figure()
    for i, pid in enumerate(cmp):
        pf = fc_df[fc_df["product_id"]==pid]
        fig4.add_trace(go.Scatter(x=pf["date"], y=pf["predicted_sales"],
                                   name=pid, mode="lines",
                                   line=dict(color=pal[i%len(pal)],width=2.5)))
    fig4.update_layout(hovermode="x unified",height=300,plot_bgcolor="#fff",paper_bgcolor="#fff",
                        margin=dict(l=0,r=0,t=6,b=0),
                        legend=dict(orientation="h",y=-0.18,font=dict(size=11)),
                        xaxis=dict(showgrid=False,color="#718096"),
                        yaxis=dict(gridcolor="#f0f0f0",color="#718096",title="Predicted units"))
    st.plotly_chart(fig4, use_container_width=True)

    rows = []
    for pid in cmp:
        pf = fc_df[fc_df["product_id"]==pid]
        rows.append({"Product":pid,
                     "7-day":   f"{pf.head(7)['predicted_sales'].sum():,.0f}",
                     "30-day":  f"{pf.head(30)['predicted_sales'].sum():,.0f}",
                     "Daily avg":f"{pf['predicted_sales'].mean():.1f}",
                     "Peak":    f"{pf['predicted_sales'].max():.0f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("""<p style="text-align:center;color:#a0aec0;font-size:12px;margin-top:2rem;">
BizForeCast &nbsp;·&nbsp; Python · XGBoost · LightGBM · Streamlit</p>""",
unsafe_allow_html=True)