"""
pages/2_Models.py  —  Model Performance & Comparison
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import inject_css, navbar, section, load_model, C

st.set_page_config(page_title="Models · BizForeCast", page_icon="🤖", layout="wide")
inject_css()
navbar("Models")

model_data = load_model()

if model_data is None:
    st.warning("No trained model found. Run `python train_model.py` first.")
    st.stop()

metrics = model_data["metrics"]
xgb_m   = metrics["xgboost"]
lgb_m   = metrics["lightgbm"]
best    = metrics["best_model"]
fi      = model_data.get("feature_importance", {})
tr      = model_data["test_results"]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON CARDS
# ══════════════════════════════════════════════════════════════════════════════
section("🤖 Model Comparison", badge="XGBoost vs LightGBM")

winner_xgb = best == "XGBoost"

def pct_diff(a, b):
    """Return how much better a is vs b as a % string."""
    if b == 0: return "—"
    d = (b - a) / b * 100
    return f"{d:+.1f}%"

st.markdown(f"""
<div class="mg">
  <!-- XGBoost -->
  <div class="mc {"win" if winner_xgb else ""}">
    <div class="mn">XGBoost</div>
    <span class="mb {"w" if winner_xgb else "a"}">{"✓ Best model" if winner_xgb else "Runner-up"}</span>
    <div class="mr"><span class="mk">RMSE</span><span class="mv">{xgb_m['rmse']:.2f}</span></div>
    <div class="mr"><span class="mk">MAE</span><span class="mv">{xgb_m['mae']:.2f}</span></div>
    <div class="mr"><span class="mk">MAPE</span><span class="mv">{xgb_m['mape']:.2f}%</span></div>
    <div class="mr"><span class="mk">Accuracy</span><span class="mv">{100-xgb_m['mape']:.1f}%</span></div>
    <div class="mr"><span class="mk">Tree growth</span><span class="mv">Level-wise</span></div>
    <div class="mr"><span class="mk">Best for</span><span class="mv">Balanced datasets</span></div>
  </div>
  <!-- LightGBM -->
  <div class="mc {"win" if not winner_xgb else ""}">
    <div class="mn">LightGBM</div>
    <span class="mb {"w" if not winner_xgb else "a"}">{"✓ Best model" if not winner_xgb else "Runner-up"}</span>
    <div class="mr"><span class="mk">RMSE</span><span class="mv">{lgb_m['rmse']:.2f}</span></div>
    <div class="mr"><span class="mk">MAE</span><span class="mv">{lgb_m['mae']:.2f}</span></div>
    <div class="mr"><span class="mk">MAPE</span><span class="mv">{lgb_m['mape']:.2f}%</span></div>
    <div class="mr"><span class="mk">Accuracy</span><span class="mv">{100-lgb_m['mape']:.1f}%</span></div>
    <div class="mr"><span class="mk">Tree growth</span><span class="mv">Leaf-wise (faster)</span></div>
    <div class="mr"><span class="mk">Best for</span><span class="mv">Large datasets</span></div>
  </div>
  <!-- What the metrics mean -->
  <div class="mc">
    <div class="mn">What the metrics mean</div>
    <span class="mb a">Reference</span>
    <div class="mr"><span class="mk">RMSE</span><span class="mv">Punishes big errors</span></div>
    <div class="mr"><span class="mk">MAE</span><span class="mv">Avg error in units</span></div>
    <div class="mr"><span class="mk">MAPE</span><span class="mv">% error (scale-free)</span></div>
    <div class="mr"><span class="mk">Accuracy</span><span class="mv">100% − MAPE</span></div>
    <div class="mr"><span class="mk">Winner picked by</span><span class="mv">Lowest RMSE</span></div>
    <div class="mr"><span class="mk">RMSE diff</span>
      <span class="mv">{pct_diff(min(xgb_m["rmse"],lgb_m["rmse"]),max(xgb_m["rmse"],lgb_m["rmse"]))}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Bar chart: side-by-side metric comparison ─────────────────────────────────
section("📊 Metric comparison", badge="lower is better")

metrics_df = pd.DataFrame({
    "Metric": ["RMSE","MAE","MAPE (%)"],
    "XGBoost":  [xgb_m["rmse"], xgb_m["mae"], xgb_m["mape"]],
    "LightGBM": [lgb_m["rmse"], lgb_m["mae"], lgb_m["mape"]],
})
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name="XGBoost",  x=metrics_df["Metric"], y=metrics_df["XGBoost"],
                          marker_color="#4361ee" if winner_xgb else "#c3d0ff"))
fig_bar.add_trace(go.Bar(name="LightGBM", x=metrics_df["Metric"], y=metrics_df["LightGBM"],
                          marker_color="#7209b7" if not winner_xgb else "#d4b8fa"))
fig_bar.update_layout(barmode="group", height=300, plot_bgcolor="#fff", paper_bgcolor="#fff",
                       margin=dict(l=0,r=0,t=10,b=0),
                       legend=dict(orientation="h", y=-0.2, font=dict(size=12)),
                       xaxis=dict(showgrid=False, color="#718096"),
                       yaxis=dict(gridcolor="#f0f0f0", color="#718096", title="Value (lower = better)"))
st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ACTUAL vs PREDICTED
# ══════════════════════════════════════════════════════════════════════════════
section("🎯 Actual vs Predicted on Test Set")

all_prods = sorted(set(tr["product_id"]))
sel_prod  = st.selectbox("Product", all_prods)

perf_df = pd.DataFrame({
    "date":       pd.to_datetime(tr["dates"]),
    "actual":     tr["actual"],
    "xgb_pred":   tr["xgb_pred"],
    "lgb_pred":   tr["lgb_pred"],
    "best_pred":  tr["best_pred"],
    "product_id": tr["product_id"],
})
pp = perf_df[perf_df["product_id"]==sel_prod]

if len(pp):
    fig_av = go.Figure()
    fig_av.add_trace(go.Scatter(x=pp["date"], y=pp["actual"],   name="Actual sales",
                                 line=dict(color="#1a1a2e",width=2.5)))
    fig_av.add_trace(go.Scatter(x=pp["date"], y=pp["xgb_pred"], name="XGBoost prediction",
                                 line=dict(color="#4361ee",width=1.8,dash="dot")))
    fig_av.add_trace(go.Scatter(x=pp["date"], y=pp["lgb_pred"], name="LightGBM prediction",
                                 line=dict(color="#7209b7",width=1.8,dash="dash")))
    fig_av.update_layout(hovermode="x unified",height=340,plot_bgcolor="#fff",paper_bgcolor="#fff",
                          margin=dict(l=0,r=0,t=6,b=0),
                          legend=dict(orientation="h",y=-0.18,font=dict(size=11)),
                          xaxis=dict(showgrid=False,color="#718096"),
                          yaxis=dict(gridcolor="#f0f0f0",color="#718096",title="Units sold"))
    st.plotly_chart(fig_av, use_container_width=True)

    # Residuals chart
    with st.expander("📉 Residuals (prediction errors)"):
        pp2 = pp.copy()
        pp2["error"] = pp2["best_pred"] - pp2["actual"]
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=pp2["date"], y=pp2["error"],
                                marker_color=["#ef233c" if e>0 else "#06d6a0" for e in pp2["error"]],
                                name="Error (predicted − actual)"))
        fig_r.add_hline(y=0, line_color="#718096", line_dash="dot")
        fig_r.update_layout(height=250, plot_bgcolor="#fff", paper_bgcolor="#fff",
                              margin=dict(l=0,r=0,t=6,b=0),
                              xaxis=dict(showgrid=False,color="#718096"),
                              yaxis=dict(gridcolor="#f0f0f0",color="#718096",title="Error (units)"))
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption("Red = over-prediction (ordered too much). Green = under-prediction (risk of stockout).")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
if fi:
    section(f"🔍 Feature Importance — {model_data['model_name']}", badge="what drives predictions")

    fi_df = (pd.DataFrame(list(fi.items()), columns=["Feature","Importance"])
               .sort_values("Importance", ascending=True)
               .tail(15))

    # Rename features to human-readable labels
    label_map = {
        "lag_1":           "Lag 1 · yesterday's sales",
        "lag_7":           "Lag 7 · same weekday last week",
        "lag_14":          "Lag 14 · two weeks ago",
        "lag_30":          "Lag 30 · last month's sales",
        "rolling_mean_7":  "Rolling avg 7d · short-term trend",
        "rolling_mean_14": "Rolling avg 14d",
        "rolling_mean_30": "Rolling avg 30d · 3-month trend",
        "rolling_std_7":   "Rolling std 7d · demand volatility",
        "rolling_std_14":  "Rolling std 14d",
        "rolling_std_30":  "Rolling std 30d",
        "day_of_week":     "Day of week · weekly pattern",
        "day_of_month":    "Day of month",
        "month":           "Month · seasonal signal",
        "quarter":         "Quarter · seasonal signal",
        "year":            "Year",
        "is_weekend":      "Is weekend · 0 or 1",
        "week_of_year":    "Week of year",
        "days_since_start":"Trend · days since start",
        "cumulative_sales":"Cumulative sales",
    }
    fi_df["Label"] = fi_df["Feature"].map(label_map).fillna(fi_df["Feature"])

    n = len(fi_df)
    bar_colors = ["#4361ee" if i>=n-3 else "#c3d0ff" for i in range(n)]

    fig_fi = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Label"],
        orientation="h", marker_color=bar_colors,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(height=420, plot_bgcolor="#fff", paper_bgcolor="#fff",
                          margin=dict(l=0,r=0,t=6,b=0),
                          xaxis=dict(showgrid=False,visible=False),
                          yaxis=dict(showgrid=False,color="#4a5568",tickfont=dict(size=11)))
    st.plotly_chart(fig_fi, use_container_width=True)

    # Plain-English explanation of top 3
    top3 = fi_df.tail(3)["Label"].tolist()[::-1]
    st.markdown(f"""
    <div style="background:#f0f4ff;border:1px solid #c3d0ff;border-radius:12px;padding:1rem 1.25rem;margin-top:.5rem;">
      <div style="font-size:13px;font-weight:600;color:#1a1a2e;margin-bottom:6px;">
        Top 3 features driving this forecast:
      </div>
      {"".join(f'<div style="font-size:13px;color:#3182ce;padding:2px 0;">✦ {f}</div>' for f in top3)}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHAT EACH METRIC MEANS — plain English explainer
# ══════════════════════════════════════════════════════════════════════════════
section("📖 What do these numbers mean?")
ec1, ec2, ec3 = st.columns(3)
with ec1:
    st.markdown("""
    <div class="kc">
      <div class="kl">RMSE — Root Mean Squared Error</div>
      <div style="font-size:13px;color:#718096;line-height:1.7;margin-top:6px;">
        Square each prediction error, average them, take the square root.
        Large errors are punished more than small ones.
        <br><br>
        <strong style="color:#1a1a2e;">RMSE of 45</strong> means predictions
        are roughly 45 units off — with big misses penalised heavily.
      </div>
    </div>""", unsafe_allow_html=True)
with ec2:
    st.markdown("""
    <div class="kc">
      <div class="kl">MAE — Mean Absolute Error</div>
      <div style="font-size:13px;color:#718096;line-height:1.7;margin-top:6px;">
        Average of absolute errors. All errors weighted equally —
        a 10-unit miss and a 100-unit miss count proportionally.
        <br><br>
        <strong style="color:#1a1a2e;">MAE of 32</strong> means on average
        the forecast is 32 units away from the truth.
      </div>
    </div>""", unsafe_allow_html=True)
with ec3:
    st.markdown("""
    <div class="kc">
      <div class="kl">MAPE — Mean Absolute Percentage Error</div>
      <div style="font-size:13px;color:#718096;line-height:1.7;margin-top:6px;">
        Percentage version of MAE. Scale-independent — easy to explain
        to stakeholders regardless of sales volume.
        <br><br>
        <strong style="color:#1a1a2e;">MAPE of 13%</strong> means the model
        is off by 13% on average → <strong>87% accuracy.</strong>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("""<p style="text-align:center;color:#a0aec0;font-size:12px;margin-top:2rem;">
BizForeCast &nbsp;·&nbsp; Python · XGBoost · LightGBM · Streamlit</p>""",
unsafe_allow_html=True)