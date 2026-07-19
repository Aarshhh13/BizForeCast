"""
app.py  —  BizForeCast Landing Page
"""
import streamlit as st
from utils import inject_css, navbar, section, C

st.set_page_config(
    page_title="BizForeCast — ML Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()
navbar("Home")

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 55%,#0f3460 100%);
            border-radius:20px;padding:3.5rem 2rem 3rem;margin:0 0 2rem;text-align:center;">
  <div style="font-size:3rem;margin-bottom:.5rem;">📈</div>
  <h1 style="font-size:2.6rem;font-weight:800;color:#fff;margin:0;letter-spacing:-1px;">
    BizForeCast
  </h1>
  <p style="font-size:1.1rem;color:#a0aec0;margin:.75rem 0 0;line-height:1.7;">
    ML-powered demand forecasting for small businesses.<br>
    Know what you'll sell — <em style="color:#90cdf4;">before you run out.</em>
  </p>
  <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:1.4rem;">
    <span style="background:rgba(67,97,238,.2);color:#90cdf4;border:1px solid rgba(67,97,238,.4);
                 padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;">XGBoost</span>
    <span style="background:rgba(114,9,183,.2);color:#d6bcfa;border:1px solid rgba(114,9,183,.4);
                 padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;">LightGBM</span>
    <span style="background:rgba(6,214,160,.2);color:#9ae6b4;border:1px solid rgba(6,214,160,.4);
                 padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;">Pandas · NumPy</span>
    <span style="background:rgba(255,209,102,.2);color:#f6e05e;border:1px solid rgba(255,209,102,.4);
                 padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;">Streamlit</span>
  </div>
  <div style="margin-top:2rem;display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
    <a href="/Dashboard" target="_self"
       style="background:#4361ee;color:#fff;padding:12px 30px;border-radius:10px;
              font-weight:700;font-size:14px;text-decoration:none;display:inline-block;">
      Launch Dashboard →
    </a>
    <a href="https://github.com/your-username/BizForeCast" target="_blank"
       style="background:rgba(255,255,255,.08);color:#cbd5e0;padding:12px 24px;border-radius:10px;
              font-weight:600;font-size:14px;text-decoration:none;display:inline-block;
              border:1px solid rgba(255,255,255,.15);">
      GitHub ↗
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

# ── STATS STRIP ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="kg" style="grid-template-columns:repeat(4,1fr);">
  <div class="kc" style="text-align:center;">
    <div class="kl">Models trained</div>
    <div class="kv">2</div>
    <div class="ks">XGBoost + LightGBM</div>
  </div>
  <div class="kc" style="text-align:center;">
    <div class="kl">Features engineered</div>
    <div class="kv">18+</div>
    <div class="ks">Lag, rolling, seasonal, trend</div>
  </div>
  <div class="kc" style="text-align:center;">
    <div class="kl">Forecast horizon</div>
    <div class="kv">90</div>
    <div class="ks">days ahead per product</div>
  </div>
  <div class="kc" style="text-align:center;">
    <div class="kl">Model accuracy</div>
    <div class="kv">~87%</div>
    <div class="ks">MAPE on held-out test set</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── PROBLEM / SOLUTION ────────────────────────────────────────────────────────
section("The problem BizForeCast solves")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div style="background:#fff5f5;border:1px solid #fed7d7;border-radius:14px;padding:1.5rem;height:100%;">
      <div style="font-size:1.4rem;margin-bottom:8px;">❌ Without forecasting</div>
      <ul style="color:#744210;font-size:13px;line-height:2;padding-left:18px;margin:0;">
        <li>Guess how much stock to order each month</li>
        <li>Overstock — cash tied up in unsold goods</li>
        <li>Understock — customers leave empty-handed</li>
        <li>React to problems only <em>after</em> they happen</li>
      </ul>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="background:#f0fff4;border:1px solid #c6f6d5;border-radius:14px;padding:1.5rem;height:100%;">
      <div style="font-size:1.4rem;margin-bottom:8px;">✅ With BizForeCast</div>
      <ul style="color:#22543d;font-size:13px;line-height:2;padding-left:18px;margin:0;">
        <li>ML predicts demand 90 days ahead, per product</li>
        <li>Right stock levels — no waste, no shortages</li>
        <li>Automatic alerts when stock risk is detected</li>
        <li>Understand <em>why</em> sales go up or down</li>
      </ul>
    </div>""", unsafe_allow_html=True)

# ── HOW IT WORKS ──────────────────────────────────────────────────────────────
section("How it works")
steps = [
    ("#4361ee","1","Upload data",    "Drop in your CSV with date, product and sales columns — or explore with built-in sample data."),
    ("#7209b7","2","Feature engineering","Lag variables, rolling averages, seasonal signals and trend indicators are created automatically."),
    ("#06d6a0","3","Model training", "XGBoost and LightGBM train on your data. The better model is selected automatically by RMSE."),
    ("#f72585","4","Forecast",       "Day-by-day predictions up to 90 days ahead, with confidence bands and peak-day detection."),
    ("#ffd166","5","Actionable UI",  "Visualise trends, compare products, get stock alerts, and download forecasts as CSV."),
]
cols = st.columns(5)
for col, (color, num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style="text-align:center;padding:.75rem .25rem;">
          <div style="width:40px;height:40px;border-radius:50%;background:{color};color:#fff;
                      font-weight:800;font-size:1rem;display:flex;align-items:center;
                      justify-content:center;margin:0 auto 10px;">{num}</div>
          <div style="font-weight:600;font-size:13px;color:#1a1a2e;margin-bottom:5px;">{title}</div>
          <div style="font-size:12px;color:#718096;line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── FEATURE ENGINEERING CARDS ─────────────────────────────────────────────────
section("Features the model learns from", badge="Feature Engineering")

features = [
    ("⏪","Lag 1",          "lag_1",           "Yesterday's actual sales — the single strongest predictor. High yesterday means high today."),
    ("📅","Lag 7",          "lag_7",            "Same weekday last week. Captures the weekly cycle so Saturdays resemble Saturdays."),
    ("📦","Lag 30",         "lag_30",           "Sales from last month. Teaches the model about slow-moving monthly trends."),
    ("〰️","Rolling Avg 7", "rolling_mean_7",   "Average of the past 7 days. Smooths daily noise to reveal the true short-term trend."),
    ("📊","Rolling Avg 90", "rolling_mean_30",  "Average of the previous ~3 months. Captures longer demand momentum without spikes."),
    ("📐","Rolling Std",    "rolling_std_7",    "How much sales varied last week. High volatility → model forecasts more conservatively."),
    ("🌡️","Season",        "month, quarter",   "Month number (1–12) and quarter. Lets the model learn summer peaks and holiday spikes."),
    ("📆","Day of week",    "day_of_week",      "Monday=0 … Sunday=6. Every product has a weekly rhythm the model learns from."),
    ("📈","Trend",          "days_since_start", "Days since first sale — captures overall business growth so old patterns don't mislead."),
    ("🎯","Is Weekend",     "is_weekend",       "Binary 0/1. Simple but powerful — separates the weekend pattern from weekday instantly."),
]

st.markdown('<div class="fg">', unsafe_allow_html=True)
for icon, name, tech, desc in features:
    st.markdown(f"""
    <div class="fc">
      <div class="fi">{icon}</div>
      <div class="fn">{name}</div>
      <div class="ft">{tech}</div>
      <div class="fd">{desc}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── CTA ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;background:#f0f4ff;border:1px solid #c3d0ff;
            border-radius:16px;padding:2.5rem 2rem;margin:2rem 0 1rem;">
  <div style="font-size:1.4rem;font-weight:700;color:#1a1a2e;margin-bottom:6px;">
    Ready to forecast your sales?
  </div>
  <div style="font-size:14px;color:#718096;margin-bottom:1.4rem;">
    Upload your own CSV or explore with sample data — no setup needed.
  </div>
  <a href="/Dashboard" target="_self"
     style="background:#4361ee;color:#fff;padding:12px 34px;border-radius:10px;
            font-weight:700;font-size:14px;text-decoration:none;display:inline-block;">
    Open Dashboard →
  </a>
</div>
<p style="text-align:center;color:#a0aec0;font-size:12px;margin-top:1.5rem;">
  BizForeCast &nbsp;·&nbsp; Python · XGBoost · LightGBM · Streamlit &nbsp;·&nbsp;
  <a href="https://github.com/Aarshhh13/BizForeCast"
     style="color:#4361ee;text-decoration:none;">GitHub ↗</a>
</p>
""", unsafe_allow_html=True)
