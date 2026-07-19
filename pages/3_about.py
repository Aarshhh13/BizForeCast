"""
pages/3_About.py  —  About BizForeCast
"""
import streamlit as st
from utils import inject_css, navbar, section

st.set_page_config(page_title="About · BizForeCast", page_icon="ℹ️", layout="wide")
inject_css()
navbar("About")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="about-hero">
  <h1>📈 BizForeCast</h1>
  <p>
    A machine learning demand forecasting system built to help small businesses
    predict future product sales, optimise inventory, and make smarter ordering decisions.
  </p>
  <div style="margin-top:1rem;">
    <span class="tpill">Python 3.11</span>
    <span class="tpill">XGBoost</span>
    <span class="tpill">LightGBM</span>
    <span class="tpill">Pandas</span>
    <span class="tpill">NumPy</span>
    <span class="tpill">Streamlit</span>
    <span class="tpill">Plotly</span>
    <span class="tpill">Scikit-learn</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Quick links ───────────────────────────────────────────────────────────────
lc1, lc2, lc3 = st.columns(3)
with lc1:
    st.markdown("""
    <div class="kc" style="text-align:center;">
      <div style="font-size:1.5rem;margin-bottom:6px;">🌐</div>
      <div class="kl">Live App</div>
      <a href="https://bizforecastiq.streamlit.app" target="_blank"
         style="font-size:13px;color:#4361ee;text-decoration:none;font-weight:600;">
        bizforecastiq.streamlit.app ↗
      </a>
    </div>""", unsafe_allow_html=True)
with lc2:
    st.markdown("""
    <div class="kc" style="text-align:center;">
      <div style="font-size:1.5rem;margin-bottom:6px;">💻</div>
      <div class="kl">Source Code</div>
      <a href="https://github.com/Aarshhh13/BizForeCast" target="_blank"
         style="font-size:13px;color:#4361ee;text-decoration:none;font-weight:600;">
        github.com/Aarshhh13/BizForeCast ↗
      </a>
    </div>""", unsafe_allow_html=True)
with lc3:
    st.markdown("""
    <div class="kc" style="text-align:center;">
      <div style="font-size:1.5rem;margin-bottom:6px;">🔗</div>
      <div class="kl">Connect</div>
      <a href="https://www.linkedin.com/in/aarsh-shrivastava-95a3262a5/" target="_blank"
         style="font-size:13px;color:#4361ee;text-decoration:none;font-weight:600;">
        LinkedIn Profile ↗
      </a>
    </div>""", unsafe_allow_html=True)

# ── Why I built this ──────────────────────────────────────────────────────────
section("💡 Why I built this")
st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
            padding:1.5rem 1.75rem;font-size:14px;color:#4a5568;line-height:1.85;">
  Small businesses lose significant revenue every year from two inventory mistakes:
  <strong>overstock</strong> (cash locked in unsold goods) and <strong>understock</strong>
  (customers walking away empty-handed). Most solutions either require expensive ERP software
  or rely on manual spreadsheet guesswork.
  <br><br>
  BizForeCast was built to bridge that gap — a free, open-source ML forecasting tool that
  any small business can run with just a CSV of their sales history. It uses the same
  gradient boosting models that power demand forecasting at large retailers, packaged into
  an approachable dashboard with no data science background required.
</div>
""", unsafe_allow_html=True)

# ── How it was built ─────────────────────────────────────────────────────────
section("🔨 How it was built", badge="Technical Pipeline")
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("""
    <div class="tl">
      <div class="tli">
        <div class="tlt">Data Collection & Generation</div>
        <div class="tld">Built a synthetic data generator producing realistic multi-product sales
        with weekly cycles, seasonal peaks, trend growth, and random noise.
        Supports real CSV upload for any business.</div>
      </div>
      <div class="tli">
        <div class="tlt">Feature Engineering</div>
        <div class="tld">Created 18+ features per row: lag variables (1/7/14/30 days),
        rolling means and standard deviations (7/14/30-day windows),
        seasonal indicators (month, quarter, day-of-week), and trend features
        (days since start, cumulative sales).</div>
      </div>
      <div class="tli">
        <div class="tlt">Time-Based Train/Test Split</div>
        <div class="tld">Used a strict chronological 80/20 split — training on older data,
        testing on the most recent 20%. Random splits would cause temporal leakage
        in time series and give falsely optimistic metrics.</div>
      </div>
      <div class="tli">
        <div class="tlt">Model Training</div>
        <div class="tld">Trained XGBoost and LightGBM with regularisation (L1/L2),
        subsampling (0.8), and low learning rate (0.05) to prevent overfitting.
        Both models evaluated on RMSE, MAE, and MAPE.</div>
      </div>
      <div class="tli">
        <div class="tlt">Automated Model Selection</div>
        <div class="tld">The model with lower RMSE on the held-out test set is automatically
        saved as the active model. All metadata, test predictions, and feature
        importances are persisted to disk.</div>
      </div>
      <div class="tli">
        <div class="tlt">Streamlit Dashboard</div>
        <div class="tld">Multi-page app with a landing page, interactive dashboard,
        model performance analysis, and about page. CSS-responsive for mobile.
        Deployed free on Streamlit Cloud.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    # Tech stack breakdown
    section("🧰 Tech stack")
    stack = [
        ("Python 3.11",    "Core language"),
        ("XGBoost 2.x",    "Primary gradient boosting model"),
        ("LightGBM 4.x",   "Secondary gradient boosting model"),
        ("Pandas",         "Data loading, cleaning, feature engineering"),
        ("NumPy",          "Numerical operations and array handling"),
        ("Scikit-learn",   "RMSE / MAE / MAPE evaluation metrics"),
        ("Streamlit",      "Multi-page interactive web dashboard"),
        ("Plotly",         "Interactive charts (line, bar, scatter)"),
        ("Pickle",         "Model serialisation and persistence"),
        ("Git / GitHub",   "Version control and source hosting"),
    ]
    for tech, role in stack:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:baseline;
                    padding:7px 0;border-bottom:1px solid #f0f0f0;font-size:13px;">
          <span style="font-weight:600;color:#1a1a2e;">{tech}</span>
          <span style="color:#718096;">{role}</span>
        </div>""", unsafe_allow_html=True)

# ── Key design decisions ──────────────────────────────────────────────────────
section("⚙️ Key design decisions")
d1, d2, d3 = st.columns(3)
for col, title, body in [
    (d1, "Why gradient boosting?",
     "XGBoost and LightGBM handle structured tabular data natively with no normalisation needed, "
     "train in seconds, and expose feature importance — critical for explaining forecasts to "
     "non-technical business owners. Neural networks become worth it only at much larger scale."),
    (d2, "Why a global model?",
     "One model trained on all products simultaneously. New products with limited history "
     "benefit from patterns learned across all products. Alternative: one model per product "
     "(Prophet-style) — doesn't share knowledge and doesn't scale to 100+ SKUs."),
    (d3, "Why recursive forecasting?",
     "Predict one day, use that prediction as a lag feature for the next day. "
     "Simple and works well up to 30 days. Error accumulates at longer horizons — "
     "an acknowledged limitation. Direct multi-output forecasting would fix this at the "
     "cost of training N separate models."),
]:
    with col:
        st.markdown(f"""
        <div class="kc" style="height:100%;">
          <div class="kl">{title}</div>
          <div style="font-size:13px;color:#718096;line-height:1.7;margin-top:6px;">{body}</div>
        </div>""", unsafe_allow_html=True)

# ── What I'd improve ─────────────────────────────────────────────────────────
section("🚀 What I'd add with more time")
st.markdown("""
<div style="background:#f0f4ff;border:1px solid #c3d0ff;border-radius:14px;
            padding:1.25rem 1.5rem;">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;">
    <div>
      <div style="font-size:13px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">
        📊 SHAP Explainability
      </div>
      <div style="font-size:12px;color:#718096;line-height:1.6;">
        Per-prediction feature attribution showing exactly which signals
        pushed a specific forecast up or down.
      </div>
    </div>
    <div>
      <div style="font-size:13px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">
        🔧 Optuna Hyperparameter Search
      </div>
      <div style="font-size:12px;color:#718096;line-height:1.6;">
        Bayesian optimisation over 50+ trials instead of fixed parameters,
        reducing RMSE further.
      </div>
    </div>
    <div>
      <div style="font-size:13px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">
        🔄 TimeSeriesSplit CV
      </div>
      <div style="font-size:12px;color:#718096;line-height:1.6;">
        5-fold walk-forward cross-validation for a more reliable performance
        estimate than a single holdout.
      </div>
    </div>
    <div>
      <div style="font-size:13px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">
        🚀 FastAPI Backend
      </div>
      <div style="font-size:12px;color:#718096;line-height:1.6;">
        REST endpoint so any ERP or inventory system can call the forecast
        API programmatically.
      </div>
    </div>
    <div>
      <div style="font-size:13px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">
        🐳 Docker + CI/CD
      </div>
      <div style="font-size:12px;color:#718096;line-height:1.6;">
        Containerised deployment with GitHub Actions auto-retraining
        on new data every week.
      </div>
    </div>
    <div>
      <div style="font-size:13px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">
        🎯 Quantile Regression
      </div>
      <div style="font-size:12px;color:#718096;line-height:1.6;">
        Proper probabilistic confidence intervals (10th–90th percentile)
        instead of the current ±15% heuristic.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FAQ ───────────────────────────────────────────────────────────────────────
section("❓ FAQ")
faqs = [
    ("Do I need to know Python to use BizForeCast?",
     "No. Once deployed, the dashboard is fully point-and-click. Just upload your CSV and explore the forecasts. Python is only needed if you want to retrain the model on your own data."),
    ("What format does my CSV need to be in?",
     "Three columns: date (YYYY-MM-DD format), product_id (any text name), and sales (a number). That's it. You can have as many products and dates as you like."),
    ("Why does the forecast for my uploaded data look simpler?",
     "When you upload your own CSV, the app uses a rolling-average fallback forecast instead of the trained XGBoost/LightGBM model — because those models were trained on the sample data, not your data. To get ML-quality forecasts on your own data, download the code from GitHub and run train_model.py on your CSV."),
    ("What's the difference between XGBoost and LightGBM?",
     "Both are gradient boosting algorithms that build decision trees sequentially, each correcting the errors of the previous. The key difference: XGBoost grows trees level-by-level (breadth-first), while LightGBM grows trees leaf-by-leaf (best-first). LightGBM is typically 5–10× faster on large datasets but produces similar accuracy on smaller datasets like this one. BizForeCast trains both and picks the winner automatically."),
    ("What does MAPE mean and what's a good score?",
     "MAPE stands for Mean Absolute Percentage Error — the average percentage gap between predicted and actual sales. Lower is better. Under 10% is excellent. 10–20% is good for retail. Above 20% suggests the model needs more training data or better features."),
]
for q, a in faqs:
    with st.expander(q):
        st.markdown(f"<div style='font-size:13px;color:#4a5568;line-height:1.8;padding:.25rem 0'>{a}</div>",
                    unsafe_allow_html=True)

st.markdown("""<p style="text-align:center;color:#a0aec0;font-size:12px;margin-top:2rem;">
BizForeCast &nbsp;·&nbsp; Python · XGBoost · LightGBM · Streamlit &nbsp;·&nbsp;
<a href="https://github.com/Aarshhh13/BizForeCast" style="color:#4361ee;text-decoration:none;">GitHub ↗</a>
</p>""", unsafe_allow_html=True)
