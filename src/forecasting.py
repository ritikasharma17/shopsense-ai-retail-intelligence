"""
ShopSense - Retail Demand Forecasting Engine
Implements time-aware evaluation, baseline comparisons, and uncertainty bounds.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

@dataclass
class ForecastMetrics:
    model_name: str
    mae: float
    rmse: float
    wape_pct: float
    is_best: bool = False


@dataclass
class ForecastResult:
    historical_df: pd.DataFrame
    forecast_df: pd.DataFrame
    test_eval_df: Optional[pd.DataFrame]
    model_metrics: List[ForecastMetrics]
    selected_model: str
    target_variable: str  # 'revenue' or 'quantity'
    granularity: str      # 'Daily' or 'Weekly'
    forecast_horizon_days: int
    data_points_count: int
    train_points_count: int
    test_points_count: int
    limitations: List[str] = field(default_factory=list)


def prepare_timeseries(
    df: pd.DataFrame,
    target: str = "quantity",
    freq: str = "D",
    product_id: Optional[str] = None,
    category: Optional[str] = None
) -> pd.DataFrame:
    """
    Filters and aggregates dataset to a continuous timeseries without gaps.
    """
    sub = df.copy()
    if product_id and product_id != "All Products":
        sub = sub[sub["product_id"] == product_id]
    elif category and category != "All Categories":
        sub = sub[sub["category"] == category]

    if sub.empty:
        return pd.DataFrame(columns=["date", target])

    # Daily aggregation
    ts = sub.groupby(pd.Grouper(key="date", freq=freq))[target].sum().reset_index()

    # Reindex to full continuous date range to fill zero-sales days
    min_date = ts["date"].min()
    max_date = ts["date"].max()
    full_idx = pd.date_range(start=min_date, end=max_date, freq=freq)
    ts = ts.set_index("date").reindex(full_idx, fill_value=0.0).rename_axis("date").reset_index()

    return ts


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> ForecastMetrics:
    """
    Computes MAE, RMSE, and WAPE (Weighted Absolute Percentage Error).
    """
    # Enforce non-negativity on forecasts
    y_pred = np.clip(y_pred, 0.0, None)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sum_true = float(np.sum(y_true))
    wape = float((np.sum(np.abs(y_true - y_pred)) / sum_true * 100.0)) if sum_true > 0 else 0.0

    return ForecastMetrics(
        model_name=model_name,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        wape_pct=round(wape, 2)
    )


def build_features(df_ts: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Constructs calendar and autoregressive lag features for regression forecasting.
    """
    feat = df_ts.copy()
    feat["day_of_week"] = feat["date"].dt.dayofweek
    feat["day_of_year"] = feat["date"].dt.dayofyear
    feat["month"] = feat["date"].dt.month
    feat["is_weekend"] = feat["day_of_week"].isin([5, 6]).astype(int)

    # Autoregressive lags
    feat["lag_1"] = feat[target].shift(1)
    feat["lag_7"] = feat[target].shift(7)
    feat["lag_14"] = feat[target].shift(14)
    feat["rolling_mean_7"] = feat[target].shift(1).rolling(window=7, min_periods=1).mean()
    feat["rolling_std_7"] = feat[target].shift(1).rolling(window=7, min_periods=1).std().fillna(0.0)

    return feat


def generate_forecast(
    df: pd.DataFrame,
    target: str = "quantity",
    horizon_days: int = 30,
    test_days: int = 21,
    product_id: Optional[str] = None,
    category: Optional[str] = None,
    preferred_model: str = "Auto (Best Model)"
) -> ForecastResult:
    """
    Executes time-aware train/test evaluation across Baseline and ML models,
    selects the best model, and generates out-of-sample forecasts with confidence intervals.
    """
    ts = prepare_timeseries(df, target=target, freq="D", product_id=product_id, category=category)
    n_points = len(ts)

    limitations = []
    if n_points < 30:
        limitations.append("Time series history is very short (< 30 days). Uncertainty intervals are elevated.")
    if n_points < test_days * 2:
        test_days = max(7, n_points // 4)

    # 1. Time-Ordered Train / Test Split
    train_df = ts.iloc[:-test_days].copy()
    test_df = ts.iloc[-test_days:].copy()

    y_train = train_df[target].values
    y_test = test_df[target].values

    # Model 1: Baseline Moving Average (7-Day Rolling Mean)
    window = min(7, len(train_df))
    last_train_ma = train_df[target].iloc[-window:].mean()
    baseline_test_pred = np.full(shape=len(test_df), fill_value=last_train_ma)
    metrics_baseline = calculate_metrics(y_test, baseline_test_pred, "7-Day Moving Average (Baseline)")

    # Model 2: Holt's Linear Exponential Smoothing
    # Implementation using exponential decay smoothing
    alpha = 0.3
    beta = 0.1
    level = y_train[0] if len(y_train) > 0 else 0.0
    trend = (y_train[1] - y_train[0]) if len(y_train) > 1 else 0.0

    for val in y_train[1:]:
        prev_level = level
        level = alpha * val + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    exp_test_pred = np.array([max(0.0, level + (i + 1) * trend * 0.95) for i in range(len(test_df))])
    metrics_exp = calculate_metrics(y_test, exp_test_pred, "Holt Exponential Smoothing")

    # Model 3: Ridge Lag Regression
    feat_all = build_features(ts, target)
    feat_train = feat_all.iloc[:-test_days].dropna()
    feat_test = feat_all.iloc[-test_days:]

    feature_cols = ["day_of_week", "day_of_year", "month", "is_weekend", "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7"]

    ridge_test_pred = np.zeros(len(test_df))
    metrics_ridge = None

    if len(feat_train) > 14:
        X_tr = feat_train[feature_cols]
        y_tr = feat_train[target]
        model_ridge = Ridge(alpha=1.0)
        model_ridge.fit(X_tr, y_tr)

        # Recursive out-of-sample prediction for test set
        curr_ts = train_df.copy()
        preds = []
        for d in range(len(test_df)):
            curr_feat = build_features(curr_ts, target)
            last_row = curr_feat.iloc[-1:][feature_cols]
            p = float(model_ridge.predict(last_row)[0])
            p = max(0.0, p)
            preds.append(p)

            # Append predicted value to step forward
            next_date = curr_ts["date"].iloc[-1] + timedelta(days=1)
            curr_ts = pd.concat([curr_ts, pd.DataFrame([{"date": next_date, target: p}])], ignore_index=True)

        ridge_test_pred = np.array(preds)
        metrics_ridge = calculate_metrics(y_test, ridge_test_pred, "Ridge Lag Regression (ML)")

    # Collate metrics
    all_metrics = [metrics_baseline, metrics_exp]
    if metrics_ridge:
        all_metrics.append(metrics_ridge)

    # Determine best model based on MAE
    best_m = min(all_metrics, key=lambda m: m.mae)
    best_m.is_best = True

    chosen_model_name = best_m.model_name if preferred_model == "Auto (Best Model)" else preferred_model

    # Evaluation dataframe for the holdout period
    eval_df = test_df[["date", target]].rename(columns={target: "actual"}).copy()
    eval_df["baseline_pred"] = baseline_test_pred
    eval_df["exp_smoothing_pred"] = exp_test_pred
    if metrics_ridge:
        eval_df["ridge_pred"] = ridge_test_pred

    # 2. Out-of-Sample Final Forecasting (Next H Days)
    last_date = ts["date"].max()
    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]

    # Compute residual variance on test set to formulate confidence intervals
    if "Ridge" in chosen_model_name and metrics_ridge:
        test_residuals = y_test - ridge_test_pred
    elif "Exponential" in chosen_model_name:
        test_residuals = y_test - exp_test_pred
    else:
        test_residuals = y_test - baseline_test_pred

    residual_std = float(np.std(test_residuals)) if len(test_residuals) > 1 else max(1.0, float(np.mean(y_train) * 0.2))

    # Generate point forecasts for future horizon
    if "Ridge" in chosen_model_name and metrics_ridge:
        # Refit Ridge on 100% of available data
        feat_full = build_features(ts, target).dropna()
        X_full = feat_full[feature_cols]
        y_full = feat_full[target]
        model_final = Ridge(alpha=1.0)
        model_final.fit(X_full, y_full)

        sim_ts = ts.copy()
        future_preds = []
        for d in range(horizon_days):
            sim_feat = build_features(sim_ts, target)
            last_row = sim_feat.iloc[-1:][feature_cols]
            p = float(model_final.predict(last_row)[0])
            p = max(0.0, p)
            future_preds.append(p)
            sim_ts = pd.concat([sim_ts, pd.DataFrame([{"date": future_dates[d], target: p}])], ignore_index=True)
    elif "Exponential" in chosen_model_name:
        # Refit smoothing over all data
        level_f = ts[target].iloc[0]
        trend_f = (ts[target].iloc[1] - ts[target].iloc[0]) if len(ts) > 1 else 0.0
        for val in ts[target].iloc[1:]:
            p_lev = level_f
            level_f = alpha * val + (1 - alpha) * (p_lev + trend_f)
            trend_f = beta * (level_f - p_lev) + (1 - beta) * trend_f
        future_preds = [max(0.0, level_f + (i + 1) * trend_f * 0.95) for i in range(horizon_days)]
    else:
        # Moving average
        ma_val = float(ts[target].iloc[-7:].mean())
        # Slight day-of-week indexing
        dow_profile = ts.groupby(ts["date"].dt.dayofweek)[target].mean()
        mean_all = ts[target].mean() if ts[target].mean() > 0 else 1.0
        future_preds = []
        for d in future_dates:
            dow = d.dayofweek
            mult = (dow_profile.get(dow, mean_all) / mean_all) if mean_all > 0 else 1.0
            future_preds.append(max(0.0, ma_val * mult))

    # Construct parametric prediction intervals
    # Expanding uncertainty over time horizon (sqrt of horizon step)
    f_df = pd.DataFrame({"date": future_dates, "forecast": np.round(future_preds, 2)})
    step_mult = np.sqrt(np.arange(1, horizon_days + 1))
    f_df["lower_80"] = np.maximum(0.0, np.round(f_df["forecast"] - 1.28 * residual_std * (0.8 + 0.05 * step_mult), 2))
    f_df["upper_80"] = np.round(f_df["forecast"] + 1.28 * residual_std * (0.8 + 0.05 * step_mult), 2)
    f_df["lower_95"] = np.maximum(0.0, np.round(f_df["forecast"] - 1.96 * residual_std * (0.8 + 0.08 * step_mult), 2))
    f_df["upper_95"] = np.round(f_df["forecast"] + 1.96 * residual_std * (0.8 + 0.08 * step_mult), 2)

    limitations.append("Forecast assumes constant macroeconomic environment, typical supplier delivery, and baseline marketing.")
    limitations.append("Unplanned promotions, sudden competitor actions, or supplier stockouts will cause variance from projected demand.")

    return ForecastResult(
        historical_df=ts,
        forecast_df=f_df,
        test_eval_df=eval_df,
        model_metrics=all_metrics,
        selected_model=chosen_model_name,
        target_variable=target,
        granularity="Daily",
        forecast_horizon_days=horizon_days,
        data_points_count=n_points,
        train_points_count=len(train_df),
        test_points_count=len(test_df),
        limitations=limitations
    )
