"""
app.py - BizForeCast Streamlit Dashboard
Small Business Sales Demand Forecasting System

Run this file to launch the interactive dashboard:
    streamlit run app.py

Dashboard sections:
  1. Overview metrics (total sales, best product, model accuracy)
  2. Sales trend analysis (historical charts per product)
  3. Demand forecasting (future predictions chart)
  4. Model performance (actual vs predicted, error metrics)
  5. Feature importance (which features drive predictions)
"""

import pickle
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from forecast import forecast_all_products


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BizForeCast — Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

# ─── Load Data ────────────────────────────────────────────────────────────────

@st.cache_data
def load_model_data():
    """Load the trained model and all associated data (cached for speed)."""
    if not os.path.exists("models/trained_model.pkl"):
        return None
    with open("models/trained_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_forecast(n_days):
    """Generate and cache the forecast so it's not re-run every interaction."""
    return forecast_all_products(n_days=n_days)


# ─── App Layout ───────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("📈 BizForeCast")
    st.markdown("**Small Business Sales Demand Forecasting System** | Powered by XGBoost & LightGBM")
    st.divider()

    # Load model
    data = load_model_data()
    if data is None:
        st.error("⚠️ No trained model found. Please run `python train_model.py` first!")
        st.code("python train_model.py", language="bash")
        return

    # Extract data components
    model_name    = data["model_name"]
    metrics       = data["metrics"]
    test_results  = data["test_results"]
    train_df      = data["train_df"]
    test_df       = data["test_df"]
    feature_imp   = data.get("feature_importance", {})

    all_history = pd.concat([train_df, test_df]).sort_values(["product_id", "date"])
    products    = sorted(all_history["product_id"].unique())

    # ── Sidebar Controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")
        selected_product = st.selectbox("Select Product", products)
        forecast_days    = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=7)
        st.divider()
        st.markdown(f"**Active Model:** {model_name}")
        best = metrics["best"]
        st.metric("RMSE", f"{best['rmse']:.1f}")
        st.metric("MAE",  f"{best['mae']:.1f}")
        st.metric("MAPE", f"{best['mape']:.1f}%")

    # ── Section 1: Overview Metrics ──────────────────────────────────────────
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)

    total_sales   = all_history["sales"].sum()
    avg_daily     = all_history["sales"].mean()
    best_product  = all_history.groupby("product_id")["sales"].sum().idxmax()
    model_accuracy = max(0, 100 - metrics["best"]["mape"])

    col1.metric("Total Sales",     f"{total_sales:,.0f} units")
    col2.metric("Avg Daily Sales", f"{avg_daily:.1f} units")
    col3.metric("Top Product",     best_product)
    col4.metric("Model Accuracy",  f"{model_accuracy:.1f}%")

    st.divider()

    # ── Section 2: Sales Trend Analysis ─────────────────────────────────────
    st.subheader(f"📅 Sales Trend — {selected_product}")

    product_history = all_history[all_history["product_id"] == selected_product].copy()
    product_history["period"] = ["Training" if d <= train_df["date"].max()
                                  else "Testing" for d in product_history["date"]]

    fig_trend = px.line(
        product_history, x="date", y="sales",
        color="period",
        color_discrete_map={"Training": "#4C8BF5", "Testing": "#F5A623"},
        labels={"sales": "Units Sold", "date": "Date", "period": "Period"},
        title=f"Historical Sales for {selected_product}",
    )
    fig_trend.update_layout(hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Rolling average overlay
    product_history["7-Day Avg"]  = product_history["sales"].rolling(7,  min_periods=1).mean()
    product_history["30-Day Avg"] = product_history["sales"].rolling(30, min_periods=1).mean()

    col_a, col_b = st.columns(2)
    with col_a:
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=product_history["date"], y=product_history["sales"],
                                       name="Daily Sales", opacity=0.4, line=dict(color="#4C8BF5")))
        fig_roll.add_trace(go.Scatter(x=product_history["date"], y=product_history["7-Day Avg"],
                                       name="7-Day Avg", line=dict(color="#F5A623", width=2)))
        fig_roll.add_trace(go.Scatter(x=product_history["date"], y=product_history["30-Day Avg"],
                                       name="30-Day Avg", line=dict(color="#E74C3C", width=2)))
        fig_roll.update_layout(title="Sales with Rolling Averages", plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig_roll, use_container_width=True)

    with col_b:
        # Day of week analysis
        product_history["day_name"] = pd.to_datetime(product_history["date"]).dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_sales = product_history.groupby("day_name")["sales"].mean().reindex(dow_order).reset_index()
        fig_dow = px.bar(dow_sales, x="day_name", y="sales",
                          title="Average Sales by Day of Week",
                          labels={"sales": "Avg Units", "day_name": ""},
                          color="sales", color_continuous_scale="Blues")
        fig_dow.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dow, use_container_width=True)

    st.divider()

    # ── Section 3: Demand Forecast ───────────────────────────────────────────
    st.subheader(f"🔮 {forecast_days}-Day Demand Forecast")

    with st.spinner(f"Generating {forecast_days}-day forecast..."):
        forecast_df = get_forecast(forecast_days)

    product_forecast = forecast_df[forecast_df["product_id"] == selected_product]

    fig_forecast = go.Figure()
    # Historical sales (last 60 days for context)
    recent_history = product_history.tail(60)
    fig_forecast.add_trace(go.Scatter(
        x=recent_history["date"], y=recent_history["sales"],
        name="Historical Sales", line=dict(color="#4C8BF5", width=2), mode="lines"
    ))
    # Forecast line
    fig_forecast.add_trace(go.Scatter(
        x=product_forecast["date"], y=product_forecast["predicted_sales"],
        name="Forecast", line=dict(color="#2ECC71", width=2, dash="dash"), mode="lines+markers"
    ))
    # Confidence band (±15% uncertainty range)
    fig_forecast.add_trace(go.Scatter(
        x=list(product_forecast["date"]) + list(product_forecast["date"][::-1]),
        y=list(product_forecast["predicted_sales"] * 1.15) + list(product_forecast["predicted_sales"][::-1] * 0.85),
        fill="toself", fillcolor="rgba(46,204,113,0.1)", line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Range", showlegend=True
    ))
    fig_forecast.update_layout(title=f"Sales Forecast — {selected_product} (Next {forecast_days} Days)",
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                 hovermode="x unified")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Forecast summary table
    st.markdown("**Forecast Summary**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predicted Units", f"{product_forecast['predicted_sales'].sum():,.0f}")
    col2.metric("Daily Average",         f"{product_forecast['predicted_sales'].mean():.1f}")
    col3.metric("Peak Day Prediction",   f"{product_forecast['predicted_sales'].max():.0f}")

    with st.expander("View full forecast table"):
        display_df = product_forecast.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df.columns = ["Date", "Predicted Sales", "Product"]
        st.dataframe(display_df[["Date", "Predicted Sales"]], use_container_width=True)

    st.divider()

    # ── Section 4: Model Performance ─────────────────────────────────────────
    st.subheader("🎯 Model Performance — Actual vs Predicted")

    test_dates   = pd.to_datetime(test_results["dates"])
    actual_sales = test_results["actual"]
    best_preds   = test_results["best_pred"]
    product_ids  = test_results["product_id"]

    perf_df = pd.DataFrame({
        "date": test_dates, "actual": actual_sales,
        "predicted": best_preds, "product_id": product_ids,
    })
    prod_perf = perf_df[perf_df["product_id"] == selected_product]

    if len(prod_perf) > 0:
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=prod_perf["date"], y=prod_perf["actual"],
                                       name="Actual Sales", line=dict(color="#4C8BF5")))
        fig_perf.add_trace(go.Scatter(x=prod_perf["date"], y=prod_perf["predicted"],
                                       name="Predicted Sales", line=dict(color="#F5A623", dash="dot")))
        fig_perf.update_layout(title=f"Actual vs Predicted on Test Set — {selected_product}",
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                 hovermode="x unified")
        st.plotly_chart(fig_perf, use_container_width=True)

    # Model comparison table
    st.markdown("**Model Comparison**")
    comparison = pd.DataFrame({
        "Model":  ["XGBoost", "LightGBM"],
        "RMSE":   [f"{metrics['xgboost']['rmse']:.2f}", f"{metrics['lightgbm']['rmse']:.2f}"],
        "MAE":    [f"{metrics['xgboost']['mae']:.2f}",  f"{metrics['lightgbm']['mae']:.2f}"],
        "MAPE":   [f"{metrics['xgboost']['mape']:.2f}%", f"{metrics['lightgbm']['mape']:.2f}%"],
        "Winner": ["✅" if metrics["best_model"] == "XGBoost" else "",
                   "✅" if metrics["best_model"] == "LightGBM" else ""],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 5: Feature Importance ────────────────────────────────────────
    st.subheader("🔍 Feature Importance — What Drives Predictions?")

    if feature_imp:
        fi_df = pd.DataFrame(list(feature_imp.items()), columns=["Feature", "Importance"])
        fi_df  = fi_df.sort_values("Importance", ascending=True).tail(15)

        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         title=f"Top Features ({model_name})",
                         color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_fi, use_container_width=True)

    # Footer
    st.divider()
    st.markdown(
        "<small style='color:gray'>BizForeCast | Built with Python, XGBoost, LightGBM & Streamlit</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()