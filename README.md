# 📈 BizForeCast — Small Business Sales Demand Forecasting System

A machine learning system that predicts future product demand using historical sales data, helping small businesses optimize inventory and reduce stockouts.

**[Live Demo →](https://your-name-bizforecast.streamlit.app)** | **[GitHub →](https://github.com/your-username/BizForeCast)**

---

## Features

- **Demand Forecasting** — Predict sales for the next 7–90 days per product using XGBoost and LightGBM
- **Feature Engineering** — Lag variables (1/7/14/30-day), rolling averages, seasonal indicators, trend features
- **Interactive Dashboard** — Streamlit web app with dynamic charts and product-level analysis
- **Model Comparison** — Automatically selects the best-performing model (XGBoost vs LightGBM) using RMSE
- **Sales Trend Analysis** — Weekly patterns, rolling averages, day-of-week breakdown
- **Feature Importance** — Visualize which signals drive predictions most

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| ML Models | XGBoost, LightGBM |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit |
| Model Evaluation | Scikit-learn (RMSE, MAE, MAPE) |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/BizForeCast.git
cd BizForeCast
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate sample data (or use your own CSV)
```bash
python generate_sample_data.py
```

Your CSV should have these columns: `date`, `product_id`, `sales`

### 5. Train the model
```bash
python train_model.py
```

### 6. Launch the dashboard
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## Project Structure

```
BizForeCast/
├── data/
│   └── sales_data.csv          # Historical sales data (CSV)
├── models/
│   └── trained_model.pkl       # Saved trained model
├── app.py                      # Streamlit dashboard (main UI)
├── train_model.py              # Model training pipeline
├── preprocess.py               # Data cleaning & feature engineering
├── forecast.py                 # Future demand generation
├── generate_sample_data.py     # Creates sample data for testing
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Feature Engineering Details

| Feature Type | Examples | Purpose |
|---|---|---|
| Lag features | `lag_1`, `lag_7`, `lag_14`, `lag_30` | Recent sales history |
| Rolling averages | `rolling_mean_7`, `rolling_mean_30` | Smooth trend signals |
| Rolling std deviation | `rolling_std_7`, `rolling_std_30` | Demand volatility |
| Seasonal | `month`, `quarter`, `day_of_week` | Cyclical patterns |
| Calendar | `is_weekend`, `week_of_year` | Weekly effects |
| Trend | `days_since_start`, `cumulative_sales` | Long-term growth |

---

## Model Performance

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error — penalizes large errors |
| MAE | Mean Absolute Error — average prediction error in units |
| MAPE | Mean Absolute Percentage Error — % accuracy |

---

## Screenshots

*Add your dashboard screenshots here after running the app*

---

## Author

Built as a portfolio project demonstrating applied machine learning for business forecasting.

**Connect:** [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)