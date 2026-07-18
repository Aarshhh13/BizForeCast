"""
preprocess.py - Data Cleaning & Feature Engineering
BizForeCast: Small Business Sales Demand Forecasting System

This file reads raw sales data and creates useful features for the ML model.
Features created:
  - Lag variables (sales from past days/weeks)
  - Rolling averages (7-day, 14-day, 30-day moving averages)
  - Seasonal indicators (month, day of week, quarter)
  - Trend-based features (cumulative sales, days since start)
"""

import pandas as pd
import numpy as np


def load_data(filepath="data/sales_data.csv"):
    """
    Load sales data from a CSV file.
    Expected columns: date, product_id, sales
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows for {df['product_id'].nunique()} products.")
    return df


def add_lag_features(df, lags=[1, 7, 14, 30]):
    """
    Lag features: sales from N days ago.
    Example: lag_1 = yesterday's sales, lag_7 = sales from 7 days ago.
    These help the model learn from recent patterns.
    """
    print("Creating lag features...")
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("product_id")["sales"].shift(lag)
    return df


def add_rolling_features(df, windows=[7, 14, 30]):
    """
    Rolling averages: average sales over a window of past days.
    Example: rolling_7 = average of last 7 days.
    These smooth out daily noise and capture trends.
    """
    print("Creating rolling average features...")
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby("product_id")["sales"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("product_id")["sales"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std())
        )
    return df


def add_seasonal_features(df):
    """
    Seasonal & time-based features extracted from the date column.
    These help the model understand weekly/monthly/seasonal patterns.
    """
    print("Creating seasonal features...")
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Monday, 6=Sunday
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def add_trend_features(df):
    """
    Trend features to capture long-term sales direction.
    """
    print("Creating trend features...")
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    df["cumulative_sales"] = df.groupby("product_id")["sales"].cumsum().shift(1)
    return df


def preprocess(filepath="data/sales_data.csv"):
    """
    Main preprocessing pipeline — runs all steps in order.
    Returns a clean DataFrame ready for model training.
    """
    df = load_data(filepath)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_seasonal_features(df)
    df = add_trend_features(df)

    # Drop rows where lag features are NaN (first few rows per product)
    df = df.dropna()
    df = df.reset_index(drop=True)

    print(f"\nPreprocessing complete! Final shape: {df.shape}")
    print(f"Features created: {list(df.columns)}")
    return df


# Run this file directly to test preprocessing
if __name__ == "__main__":
    df = preprocess()
    print("\nSample of processed data:")
    print(df.head())git add 