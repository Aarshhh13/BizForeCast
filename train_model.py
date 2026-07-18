"""
train_model.py - Train XGBoost and LightGBM Forecasting Models
BizForeCast: Small Business Sales Demand Forecasting System

This file:
1. Loads and preprocesses the data
2. Splits data into train/test sets (time-based split, not random!)
3. Trains XGBoost and LightGBM models
4. Evaluates model performance (RMSE, MAE, MAPE)
5. Saves the best model to disk

Run this file FIRST before launching the dashboard:
    python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

from preprocess import preprocess


# ─── Configuration ───────────────────────────────────────────────────────────

# Columns used as model input features (everything except target & identifiers)
FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_std_14", "rolling_std_30",
    "day_of_week", "day_of_month", "month", "quarter", "year",
    "is_weekend", "week_of_year", "days_since_start",
]

TARGET_COL = "sales"
MODEL_PATH = "models/trained_model.pkl"

# ─── Helper Functions ─────────────────────────────────────────────────────────

def time_based_split(df, test_fraction=0.2):
    """
    Split data by time — NOT randomly.
    For time series, we must always train on past data and test on future data.
    Random splits would leak future information into training (cheating!).
    """
    split_date = df["date"].quantile(1 - test_fraction)
    train = df[df["date"] <= split_date]
    test  = df[df["date"] >  split_date]
    print(f"Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test)} rows  ({test['date'].min().date()} to {test['date'].max().date()})")
    return train, test


def evaluate(y_true, y_pred, model_name):
    """Print RMSE, MAE and MAPE for a model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # MAPE: avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    print(f"\n📊 {model_name} Performance:")
    print(f"   RMSE (Root Mean Squared Error): {rmse:.2f}  ← lower is better")
    print(f"   MAE  (Mean Absolute Error):     {mae:.2f}  ← lower is better")
    print(f"   MAPE (Mean Absolute % Error):   {mape:.2f}% ← lower is better")
    return {"rmse": rmse, "mae": mae, "mape": mape}


# ─── Model Training ───────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train an XGBoost regression model."""
    print("\n🚀 Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, "XGBoost")
    return model, y_pred, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train a LightGBM regression model."""
    print("\n🚀 Training LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, "LightGBM")
    return model, y_pred, metrics


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    # Step 1: Load and preprocess data
    print("=" * 55)
    print("  BizForeCast — Model Training")
    print("=" * 55)
    df = preprocess()

    # Step 2: Split into train/test
    print("\n📅 Splitting data by time...")
    train, test = time_based_split(df)

    # Only use columns that exist in the dataframe
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    X_train = train[available_features]
    y_train = train[TARGET_COL]
    X_test  = test[available_features]
    y_test  = test[TARGET_COL]

    # Step 3: Train both models
    xgb_model, xgb_preds, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model, lgb_preds, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)

    # Step 4: Pick the better model (lower RMSE wins)
    if xgb_metrics["rmse"] <= lgb_metrics["rmse"]:
        best_model_name = "XGBoost"
        best_model = xgb_model
        best_metrics = xgb_metrics
        best_preds = xgb_preds
    else:
        best_model_name = "LightGBM"
        best_model = lgb_model
        best_metrics = lgb_metrics
        best_preds = lgb_preds

    print(f"\n✅ Best model: {best_model_name} (RMSE: {best_metrics['rmse']:.2f})")

    # Step 5: Save the best model and metadata
    os.makedirs("models", exist_ok=True)
    save_data = {
        "model": best_model,
        "model_name": best_model_name,
        "features": available_features,
        "metrics": {
            "xgboost": xgb_metrics,
            "lightgbm": lgb_metrics,
            "best": best_metrics,
            "best_model": best_model_name,
        },
        "test_results": {
            "dates": test["date"].values,
            "actual": y_test.values,
            "xgb_pred": xgb_preds,
            "lgb_pred": lgb_preds,
            "best_pred": best_preds,
            "product_id": test["product_id"].values,
        },
        "train_df": train,
        "test_df": test,
        "feature_importance": dict(zip(available_features, best_model.feature_importances_)),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)

    print(f"\n💾 Model saved to '{MODEL_PATH}'")
    print("\n🎉 Training complete! Now run the dashboard with:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()