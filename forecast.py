"""
forecast.py - Generate Future Sales Forecasts
BizForeCast: Small Business Sales Demand Forecasting System

This file uses the trained model to predict future demand.
It builds future feature rows step-by-step (each prediction
feeds into the next, simulating a real forecasting scenario).
"""

import pickle
import numpy as np
import pandas as pd


def load_model(model_path="models/trained_model.pkl"):
    """Load the saved trained model and its metadata."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data


def forecast_product(df_product, model, feature_cols, n_days=30):
    """
    Generate n_days of future sales forecasts for a single product.

    Strategy: iterative/recursive forecasting.
    - We predict day 1, then use that prediction as a lag feature for day 2, etc.
    - This is realistic: in the real world you don't know future sales, only past ones.
    """
    # Start from the last known date
    last_date = df_product["date"].max()
    history = df_product.copy()

    predictions = []
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    for future_date in future_dates:
        # Build a single row of features for this future date
        row = {}

        # Lag features from known or already-predicted history
        all_sales = list(history["sales"])
        for lag in [1, 7, 14, 30]:
            row[f"lag_{lag}"] = all_sales[-lag] if len(all_sales) >= lag else np.mean(all_sales)

        # Rolling averages
        for window in [7, 14, 30]:
            recent = all_sales[-window:] if len(all_sales) >= window else all_sales
            row[f"rolling_mean_{window}"] = np.mean(recent)
            row[f"rolling_std_{window}"] = np.std(recent) if len(recent) > 1 else 0

        # Date/seasonal features
        row["day_of_week"]     = future_date.dayofweek
        row["day_of_month"]    = future_date.day
        row["month"]           = future_date.month
        row["quarter"]         = future_date.quarter
        row["year"]            = future_date.year
        row["is_weekend"]      = int(future_date.dayofweek >= 5)
        row["week_of_year"]    = future_date.isocalendar()[1]
        row["days_since_start"] = (future_date - df_product["date"].min()).days
        row["cumulative_sales"] = sum(all_sales)

        # Only use features the model was trained on
        X = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])

        # Predict and ensure non-negative sales
        pred = max(0, float(model.predict(X)[0]))
        predictions.append(pred)

        # Append prediction to history so next iteration can use it as a lag feature
        new_row = df_product.iloc[-1:].copy()
        new_row["date"]  = future_date
        new_row["sales"] = pred
        history = pd.concat([history, new_row], ignore_index=True)

    return pd.DataFrame({
        "date": future_dates,
        "predicted_sales": np.round(predictions, 2)
    })


def forecast_all_products(model_path="models/trained_model.pkl", n_days=30):
    """
    Generate forecasts for ALL products in the dataset.
    Returns a DataFrame with columns: date, product_id, predicted_sales.
    """
    print(f"Generating {n_days}-day forecast for all products...")
    data = load_model(model_path)
    model       = data["model"]
    feature_cols = data["features"]
    train_df    = data["train_df"]
    test_df     = data["test_df"]

    # Use all historical data (train + test) as the base for forecasting
    all_history = pd.concat([train_df, test_df]).sort_values(["product_id", "date"])

    all_forecasts = []
    for product_id, group in all_history.groupby("product_id"):
        print(f"  Forecasting product: {product_id}")
        forecast_df = forecast_product(group, model, feature_cols, n_days=n_days)
        forecast_df["product_id"] = product_id
        all_forecasts.append(forecast_df)

    result = pd.concat(all_forecasts, ignore_index=True)
    print(f"\nForecast complete! {len(result)} rows generated.")
    return result


# Test the forecasting module directly
if __name__ == "__main__":
    forecast = forecast_all_products(n_days=30)
    print("\nSample forecast output:")
    print(forecast.head(10))
    print(f"\nTotal rows: {len(forecast)}")
    print(f"Products: {forecast['product_id'].unique()}")