"""
ShopSense - Executive KPI & Performance Metric Engine
Computes traceable financial, operational, and category metrics with period deltas.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class BusinessKPIs:
    total_revenue: float
    total_units_sold: int
    total_gross_profit: float
    gross_margin_pct: float
    average_order_value: float
    total_orders: int
    unique_skus_sold: int

    # Comparative Period Deltas (vs previous equal time window)
    revenue_delta_pct: Optional[float] = None
    units_delta_pct: Optional[float] = None
    profit_delta_pct: Optional[float] = None
    margin_delta_pts: Optional[float] = None
    aov_delta_pct: Optional[float] = None

    # Traceability Metadata
    date_start: str = ""
    date_end: str = ""
    total_days: int = 0


def calculate_core_kpis(df: pd.DataFrame, compare_prior_period: bool = True) -> BusinessKPIs:
    """
    Computes primary retail financial and operational metrics.
    If compare_prior_period is True and data spans >= 14 days,
    splits the timeframe in half and computes growth deltas.
    """
    if df.empty:
        return BusinessKPIs(
            total_revenue=0.0,
            total_units_sold=0,
            total_gross_profit=0.0,
            gross_margin_pct=0.0,
            average_order_value=0.0,
            total_orders=0,
            unique_skus_sold=0,
        )

    # Core aggregations
    rev = float(df["revenue"].sum())
    units = int(df["quantity"].sum())
    profit = float(df["gross_profit"].sum())
    margin = (profit / rev * 100.0) if rev > 0 else 0.0

    # AOV: Net revenue divided by distinct order IDs (or rows if orders not distinct)
    n_orders = int(df["order_id"].nunique()) if "order_id" in df.columns else len(df)
    aov = (rev / n_orders) if n_orders > 0 else 0.0
    skus = int(df["product_id"].nunique())

    min_date = df["date"].min()
    max_date = df["date"].max()
    span_days = (max_date - min_date).days + 1

    kpis = BusinessKPIs(
        total_revenue=round(rev, 2),
        total_units_sold=units,
        total_gross_profit=round(profit, 2),
        gross_margin_pct=round(margin, 2),
        average_order_value=round(aov, 2),
        total_orders=n_orders,
        unique_skus_sold=skus,
        date_start=min_date.strftime("%Y-%m-%d"),
        date_end=max_date.strftime("%Y-%m-%d"),
        total_days=span_days,
    )

    if compare_prior_period and span_days >= 14:
        # Split into two equal halves chronologically
        midpoint = min_date + pd.Timedelta(days=span_days // 2)
        df_prior = df[df["date"] < midpoint]
        df_curr = df[df["date"] >= midpoint]

        if not df_prior.empty and not df_curr.empty:
            p_rev = float(df_prior["revenue"].sum())
            c_rev = float(df_curr["revenue"].sum())
            kpis.revenue_delta_pct = round(((c_rev - p_rev) / p_rev * 100.0), 1) if p_rev > 0 else None

            p_units = int(df_prior["quantity"].sum())
            c_units = int(df_curr["quantity"].sum())
            kpis.units_delta_pct = round(((c_units - p_units) / p_units * 100.0), 1) if p_units > 0 else None

            p_prof = float(df_prior["gross_profit"].sum())
            c_prof = float(df_curr["gross_profit"].sum())
            kpis.profit_delta_pct = round(((c_prof - p_prof) / p_prof * 100.0), 1) if p_prof > 0 else None

            p_margin = (p_prof / p_rev * 100.0) if p_rev > 0 else 0.0
            c_margin = (c_prof / c_rev * 100.0) if c_rev > 0 else 0.0
            kpis.margin_delta_pts = round(c_margin - p_margin, 2)

            p_orders = int(df_prior["order_id"].nunique()) if "order_id" in df_prior.columns else len(df_prior)
            c_orders = int(df_curr["order_id"].nunique()) if "order_id" in df_curr.columns else len(df_curr)
            p_aov = (p_rev / p_orders) if p_orders > 0 else 0.0
            c_aov = (c_rev / c_orders) if c_orders > 0 else 0.0
            kpis.aov_delta_pct = round(((c_aov - p_aov) / p_aov * 100.0), 1) if p_aov > 0 else None

    return kpis


def get_timeseries_breakdown(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Aggregates transactions into continuous timeseries with rolling indicators.
    freq: 'D' (Daily), 'W' (Weekly), 'M' (Monthly).
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "revenue", "quantity", "gross_profit", "orders", "rolling_7d_rev"])

    grouped = df.groupby(pd.Grouper(key="date", freq=freq)).agg({
        "revenue": "sum",
        "quantity": "sum",
        "gross_profit": "sum",
        "order_id": "nunique"
    }).reset_index()

    grouped = grouped.rename(columns={"order_id": "orders"})
    grouped["gross_margin_pct"] = np.where(
        grouped["revenue"] > 0,
        (grouped["gross_profit"] / grouped["revenue"]) * 100.0,
        0.0
    )

    if freq == "D":
        grouped["rolling_7d_rev"] = grouped["revenue"].rolling(window=7, min_periods=1).mean()
        grouped["rolling_7d_profit"] = grouped["gross_profit"].rolling(window=7, min_periods=1).mean()

    return grouped


def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes category-level performance summary.
    """
    if df.empty:
        return pd.DataFrame()

    total_rev = df["revenue"].sum()

    cat_df = df.groupby("category").agg({
        "revenue": "sum",
        "gross_profit": "sum",
        "quantity": "sum",
        "product_id": "nunique"
    }).reset_index()

    cat_df = cat_df.rename(columns={"product_id": "sku_count", "quantity": "units_sold"})
    cat_df["revenue_share_pct"] = (cat_df["revenue"] / total_rev * 100.0).round(1) if total_rev > 0 else 0.0
    cat_df["gross_margin_pct"] = np.where(
        cat_df["revenue"] > 0,
        (cat_df["gross_profit"] / cat_df["revenue"] * 100.0).round(2),
        0.0
    )
    cat_df = cat_df.sort_values("revenue", ascending=False).reset_index(drop=True)
    return cat_df


def get_channel_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sales channel breakdown (In-Store vs Online vs Click & Collect).
    """
    if df.empty or "sales_channel" not in df.columns:
        return pd.DataFrame()

    total_rev = df["revenue"].sum()
    ch_df = df.groupby("sales_channel").agg({
        "revenue": "sum",
        "quantity": "sum",
        "order_id": "nunique"
    }).reset_index().rename(columns={"order_id": "orders", "quantity": "units_sold"})

    ch_df["revenue_share_pct"] = (ch_df["revenue"] / total_rev * 100.0).round(1) if total_rev > 0 else 0.0
    ch_df["aov"] = np.where(ch_df["orders"] > 0, (ch_df["revenue"] / ch_df["orders"]).round(2), 0.0)
    ch_df = ch_df.sort_values("revenue", ascending=False).reset_index(drop=True)
    return ch_df
