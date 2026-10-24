"""
Unit Tests for ShopSense Executive KPI Calculations (Verified with Hand-Calculated Ground Truths)
"""

import pytest
import pandas as pd
import numpy as np

from src.kpi_engine import (
    calculate_core_kpis,
    get_timeseries_breakdown,
    get_category_breakdown,
    get_channel_breakdown,
)

@pytest.fixture
def sample_ground_truth_df():
    """
    Carefully designed synthetic dataset with hand-calculable exact totals.
    """
    return pd.DataFrame([
        {
            "date": pd.Timestamp("2024-01-01"),
            "order_id": "ORD-001",
            "product_id": "SKU-A",
            "product_name": "Product A",
            "category": "Home",
            "unit_price": 20.0,
            "unit_cost": 10.0,
            "quantity": 2,
            "discount_amount": 0.0,
            "revenue": 40.0,
            "cost_of_goods": 20.0,
            "gross_profit": 20.0,
            "sales_channel": "In-Store"
        },
        {
            "date": pd.Timestamp("2024-01-02"),
            "order_id": "ORD-002",
            "product_id": "SKU-B",
            "product_name": "Product B",
            "category": "Apparel",
            "unit_price": 50.0,
            "unit_cost": 20.0,
            "quantity": 1,
            "discount_amount": 5.0,
            "revenue": 45.0,
            "cost_of_goods": 20.0,
            "gross_profit": 25.0,
            "sales_channel": "Online"
        },
        {
            "date": pd.Timestamp("2024-01-03"),
            "order_id": "ORD-003",
            "product_id": "SKU-C",
            "product_name": "Product C",
            "category": "Home",
            "unit_price": 10.0,
            "unit_cost": 4.0,
            "quantity": 3,
            "discount_amount": 0.0,
            "revenue": 30.0,
            "cost_of_goods": 12.0,
            "gross_profit": 18.0,
            "sales_channel": "In-Store"
        }
    ])

def test_kpi_hand_calculated_ground_truth(sample_ground_truth_df):
    kpis = calculate_core_kpis(sample_ground_truth_df, compare_prior_period=False)

    # Hand-calculated assertions:
    assert kpis.total_revenue == 115.00
    assert kpis.total_units_sold == 6
    assert kpis.total_gross_profit == 63.00
    assert kpis.gross_margin_pct == pytest.approx(54.78, abs=0.01)
    assert kpis.total_orders == 3
    assert kpis.average_order_value == pytest.approx(38.33, abs=0.01)
    assert kpis.unique_skus_sold == 3
    assert kpis.total_days == 3

def test_kpi_empty_dataframe():
    df_empty = pd.DataFrame()
    kpis = calculate_core_kpis(df_empty)
    assert kpis.total_revenue == 0.0
    assert kpis.total_units_sold == 0
    assert kpis.total_gross_profit == 0.0
    assert kpis.gross_margin_pct == 0.0
    assert kpis.average_order_value == 0.0
    assert kpis.total_orders == 0

def test_category_breakdown_calculations(sample_ground_truth_df):
    cat_df = get_category_breakdown(sample_ground_truth_df)

    # Home has SKU-A ($40) + SKU-C ($30) = $70 revenue, $38 gross profit
    home_row = cat_df[cat_df["category"] == "Home"].iloc[0]
    assert home_row["revenue"] == 70.0
    assert home_row["gross_profit"] == 38.0
    assert home_row["units_sold"] == 5
    assert home_row["sku_count"] == 2
    assert home_row["revenue_share_pct"] == pytest.approx(60.9, abs=0.1)

    # Apparel has SKU-B ($45 revenue, $25 profit)
    app_row = cat_df[cat_df["category"] == "Apparel"].iloc[0]
    assert app_row["revenue"] == 45.0
    assert app_row["gross_profit"] == 25.0
    assert app_row["units_sold"] == 1

def test_channel_breakdown(sample_ground_truth_df):
    ch_df = get_channel_breakdown(sample_ground_truth_df)
    # In-Store: ORD-001 ($40) + ORD-003 ($30) = $70 revenue, 2 orders, AOV $35.0
    instore_row = ch_df[ch_df["sales_channel"] == "In-Store"].iloc[0]
    assert instore_row["revenue"] == 70.0
    assert instore_row["orders"] == 2
    assert instore_row["aov"] == 35.0
