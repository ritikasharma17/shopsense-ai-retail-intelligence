"""
Unit Tests for ShopSense Retail Demand Forecasting Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.forecasting import generate_forecast, calculate_metrics, prepare_timeseries

@pytest.fixture
def synthetic_timeseries_df():
    """Generates 90 days of daily sales for testing."""
    np.random.seed(42)
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(90)]
    rows = []
    for d in dates:
        # 10 units base + day of week lift + noise
        qty = int(max(0, 10 + (2 if d.weekday() in [4, 5, 6] else 0) + np.random.normal(0, 2)))
        rows.append({
            "date": d,
            "product_id": "SKU-001",
            "product_name": "Test Item",
            "category": "Test Category",
            "unit_price": 20.0,
            "unit_cost": 10.0,
            "quantity": qty,
            "revenue": qty * 20.0,
            "cost_of_goods": qty * 10.0,
            "gross_profit": qty * 10.0,
            "current_stock": 50,
            "lead_time_days": 10,
            "supplier_name": "Vendor A"
        })
    return pd.DataFrame(rows)

def test_time_ordered_split_strictly_preserved(synthetic_timeseries_df):
    res = generate_forecast(synthetic_timeseries_df, target="quantity", horizon_days=14, test_days=14)

    assert res.test_eval_df is not None
    # Verify train set strictly precedes test set
    train_end = res.historical_df["date"].iloc[:-14].max()
    test_start = res.test_eval_df["date"].min()
    assert train_end < test_start

def test_forecast_horizon_and_non_negativity(synthetic_timeseries_df):
    res = generate_forecast(synthetic_timeseries_df, target="quantity", horizon_days=30, test_days=14)

    assert len(res.forecast_df) == 30
    assert (res.forecast_df["forecast"] >= 0).all()
    assert (res.forecast_df["lower_80"] >= 0).all()
    assert (res.forecast_df["lower_95"] >= 0).all()

def test_prediction_intervals_monotonic_ordering(synthetic_timeseries_df):
    res = generate_forecast(synthetic_timeseries_df, target="quantity", horizon_days=21)

    f_df = res.forecast_df
    # lower_95 <= lower_80 <= forecast <= upper_80 <= upper_95
    assert (f_df["lower_95"] <= f_df["lower_80"] + 1e-5).all()
    assert (f_df["lower_80"] <= f_df["forecast"] + 1e-5).all()
    assert (f_df["forecast"] <= f_df["upper_80"] + 1e-5).all()
    assert (f_df["upper_80"] <= f_df["upper_95"] + 1e-5).all()

def test_calculate_metrics_correctness():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # Absolute errors: [2, 2, 3] -> sum = 7. MAE = 7/3 = 2.33
    # Squared errors: [4, 4, 9] -> sum = 17. RMSE = sqrt(17/3) = 2.38
    # WAPE = sum(|error|) / sum(y_true) = 7 / 60 = 11.67%
    m = calculate_metrics(y_true, y_pred, "Test Model")

    assert m.mae == pytest.approx(2.33, abs=0.01)
    assert m.rmse == pytest.approx(2.38, abs=0.01)
    assert m.wape_pct == pytest.approx(11.67, abs=0.01)

def test_short_history_warning(synthetic_timeseries_df):
    short_df = synthetic_timeseries_df.head(20) # 20 days only
    res = generate_forecast(short_df, target="quantity", horizon_days=7)
    assert len(res.limitations) > 0
    assert any("short" in lim.lower() for lim in res.limitations)
