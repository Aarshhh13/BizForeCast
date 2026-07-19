# 📈 BizForeCast — ML Sales Demand Forecasting

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-bizforecastiq.streamlit.app-4361ee?style=for-the-badge&logo=streamlit&logoColor=white)](https://bizforecastiq.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

**ML-powered demand forecasting for small businesses.**  
Know what you'll sell — before you run out.

[🌐 Live App](https://bizforecastiq.streamlit.app) &nbsp;·&nbsp; [📖 Documentation](#-how-it-works) &nbsp;·&nbsp; [🚀 Quick Start](#-quick-start)

</div>

---

## 🎯 What problem does it solve?

Small businesses lose money from two inventory mistakes every month:

| Without BizForeCast | With BizForeCast |
|---|---|
| ❌ Guess how much stock to order | ✅ ML predicts demand 90 days ahead |
| ❌ Overstock — cash locked in unsold goods | ✅ Right stock level every time |
| ❌ Understock — customers leave empty-handed | ✅ Automatic low-stock alerts |
| ❌ React to problems after they happen | ✅ Spot seasonal peaks weeks in advance |

---

## ✨ Features

- **Landing page** — hero, feature cards, how-it-works walkthrough
- **Interactive dashboard** — upload your own CSV, or explore with sample data
- **Manual data entry** — add sales row-by-row directly in the browser
- **Demand forecast** — day-by-day predictions up to 90 days, with confidence bands
- **Stock alert system** — colour-coded 🔴🟡🟢 alerts per product
- **Product comparison** — overlay forecasts for multiple products on one chart
- **Model performance page** — actual vs predicted, residuals, RMSE/MAE/MAPE explained
- **Model comparison** — XGBoost vs LightGBM side-by-side metrics + bar chart
- **Feature importance** — human-readable chart showing what drives predictions
- **Download forecast CSV** — one-click export of any product forecast
- **About page** — project story, tech stack, design decisions, FAQ
- **Mobile responsive** — works on phones and tablets

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML Models | XGBoost 2.x, LightGBM 4.x |
| Data Processing | Pandas, NumPy |
| Evaluation | Scikit-learn (RMSE, MAE, MAPE) |
| Visualisation | Plotly |
| Dashboard | Streamlit (multi-page) |
| Model Storage | Pickle |
| Deployment | Streamlit Cloud (free) |

---

## 🔬 Feature Engineering

The system automatically creates **18+ features** from raw date + sales data:

| Feature | Column | What it captures |
|---|---|---|
| **Lag 1** | `lag_1` | Yesterday's sales — strongest single predictor |
| **Lag 7** | `lag_7` | Same weekday last week — weekly cycle |
| **Lag 14 / 30** | `lag_14`, `lag_30` | Sales from last month — medium-term trends |
| **Rolling Avg 7d** | `rolling_mean_7` | Short-term trend, smooths daily noise |
| **Rolling Avg 30d** | `rolling_mean_30` | ~3-month average — sustained momentum |
| **Rolling Std** | `rolling_std_7` | Demand volatility — how erratic sales are |
| **Season** | `month`, `quarter` | Monthly/quarterly patterns, holiday peaks |
| **Day of week** | `day_of_week` | Weekday vs weekend patterns |
| **Trend** | `days_since_start` | Overall business growth direction |
| **Is weekend** | `is_weekend` | Binary 0/1 — simple but powerful signal |

---

## 🤖 ML Models

### XGBoost
Extreme Gradient Boosting. Builds decision trees **level-by-level**, each correcting
the errors of the previous. Built-in L1/L2 regularisation prevents overfitting.

### LightGBM
Light Gradient Boosting Machine (Microsoft). Builds trees **leaf-by-leaf** — 5–10×
faster than XGBoost on large datasets. Histogram-based splitting for speed.

### Model selection
Both models train on the same data. The one with **lower RMSE** on the held-out test
set is automatically selected as the active model.

### Evaluation metrics
| Metric | Formula | What it means |
|---|---|---|
| RMSE | √(mean of squared errors) | Penalises large errors more — sensitive to spikes |
| MAE | mean of absolute errors | Average error in units — easy to interpret |
| MAPE | mean of \|actual−pred\|/actual × 100 | % error — scale-independent |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/BizForeCast.git
cd BizForeCast
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate sample data
```bash
python generate_sample_data.py
```

### 5. Train the model
```bash
python train_model.py
```

### 6. Launch the dashboard
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
BizForeCast/
├── app.py                    # Landing page (Page 1)
├── pages/
│   ├── 1_Dashboard.py        # Main dashboard — upload, forecast, alerts
│   ├── 2_Models.py           # Model performance & comparison
│   └── 3_About.py            # About page, FAQ, tech stack
├── utils.py                  # Shared CSS, helpers, data loaders
├── preprocess.py             # Feature engineering pipeline
├── train_model.py            # XGBoost + LightGBM training
├── forecast.py               # Future demand generation
├── generate_sample_data.py   # Synthetic data for testing
├── .streamlit/
│   └── config.toml           # Theme and server settings
├── requirements.txt
└── README.md
```

---

## 📊 Screenshots

> *Add screenshots here after running the app locally.*

| Page | Preview |
|---|---|
| Landing page | *(screenshot)* |
| Dashboard | *(screenshot)* |
| Model comparison | *(screenshot)* |
| About | *(screenshot)* |

---

## 🧠 Key Design Decisions

**Why gradient boosting over neural networks?**  
For structured tabular data at this scale, XGBoost/LightGBM outperforms neural networks:
faster training (seconds vs hours), no normalisation required, and native feature importance
for explainability. Neural networks become worthwhile only at hundreds of products and
multi-year history.

**Why a global model?**  
One model trained on all products simultaneously — new products with limited history benefit
from patterns learned across all products. Product-per-model approaches (e.g. Prophet) don't
share knowledge and don't scale to 100+ SKUs.

**Why time-based train/test split?**  
Random splits cause temporal leakage in time series. The model must train only on past data
and be evaluated on future data — exactly like production deployment.

**Why recursive forecasting?**  
Predict one day, use that prediction as a lag feature for the next. Works well up to 30 days.
Error accumulates at longer horizons (acknowledged limitation). Direct multi-output forecasting
would fix this at the cost of training N separate models.

---

## 🚧 Planned improvements

- [ ] SHAP per-prediction explainability
- [ ] Optuna hyperparameter optimisation
- [ ] TimeSeriesSplit cross-validation (5-fold)
- [ ] FastAPI REST endpoint for ERP integration
- [ ] Docker + GitHub Actions weekly retraining
- [ ] Quantile regression for proper confidence intervals

---

## 📄 CSV format

Your CSV needs exactly **3 columns**:

```csv
date,product_id,sales
2024-01-01,Widget_A,120
2024-01-02,Widget_A,135
2024-01-01,Widget_B,88
```

| Column | Type | Example |
|---|---|---|
| `date` | YYYY-MM-DD | `2024-03-15` |
| `product_id` | Any text | `Widget_A` |
| `sales` | Number | `142` |

---

## 👤 Author

Built as a portfolio project demonstrating applied ML for business forecasting.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077b5?style=flat&logo=linkedin)](https://www.linkedin.com/in/aarsh-shrivastava-95a3262a5/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Aarshhh13)

---

<div align="center">
  <sub>BizForeCast · Python · XGBoost · LightGBM · Streamlit · Open Source</sub>
</div>