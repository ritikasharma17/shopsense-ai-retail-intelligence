"""
Unit Tests for ShopSense Inventory Intelligence & Replenishment Logic
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.inventory import analyze_inventory, inventory_to_dataframe

@pytest.fixture
def mock_inventory_sales_df():
    """Constructs 45 days of sales for 3 controlled inventory test items."""
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(45)]
    rows = []

    for d in dates:
        # SKU-CRIT: Constant 5 units/day sales. Lead time 10 days. Current stock 12 units.
        # Days of Inventory = 12 / 5 = 2.4 days. Will run out before 10d lead time -> CRITICAL.
        rows.append({
            "date": d,
            "product_id": "SKU-CRIT",
            "product_name": "Critical Item",
            "category": "Home",
            "unit_price": 20.0,
            "unit_cost": 10.0,
            "quantity": 5,
            "revenue": 100.0,
            "cost_of_goods": 50.0,
            "gross_profit": 50.0,
            "current_stock": 12,
            "lead_time_days": 10,
            "supplier_name": "Vendor A"
        })

        # SKU-SLOW: Zero sales for last 45 days. Current stock 80 units -> OVERSTOCKED / SLOW-MOVING.
        rows.append({
            "date": d,
            "product_id": "SKU-SLOW",
            "product_name": "Slow Planter",
            "category": "Home",
            "unit_price": 30.0,
            "unit_cost": 15.0,
            "quantity": 0,
            "revenue": 0.0,
            "cost_of_goods": 0.0,
            "gross_profit": 0.0,
            "current_stock": 80,
            "lead_time_days": 14,
            "supplier_name": "Vendor B"
        })

        # SKU-HEALTHY: 2 units/day sales. Lead time 7 days. Current stock 60 units -> ADEQUATE.
        rows.append({
            "date": d,
            "product_id": "SKU-HEALTHY",
            "product_name": "Healthy Soap",
            "category": "Wellness",
            "unit_price": 10.0,
            "unit_cost": 4.0,
            "quantity": 2,
            "revenue": 20.0,
            "cost_of_goods": 8.0,
            "gross_profit": 12.0,
            "current_stock": 60,
            "lead_time_days": 7,
            "supplier_name": "Vendor C"
        })

    return pd.DataFrame(rows)

def test_inventory_stockout_classification(mock_inventory_sales_df):
    report = analyze_inventory(mock_inventory_sales_df, trailing_days=45, service_level_z=1.65)

    item_map = {item.product_id: item for item in report.items}

    # SKU-CRIT assertions
    crit = item_map["SKU-CRIT"]
    assert crit.stockout_status == "CRITICAL"
    assert crit.days_of_inventory == pytest.approx(2.5, abs=0.2)
    assert crit.avg_daily_sales == pytest.approx(5.0, abs=0.2)
    assert crit.lead_time_days == 10
    assert crit.suggested_reorder_qty > 0
    assert "CRITICAL DEFICIT" in crit.alert_explanation

    # SKU-SLOW assertions
    slow = item_map["SKU-SLOW"]
    assert slow.stockout_status == "OVERSTOCKED"
    assert slow.is_slow_moving is True
    assert slow.avg_daily_sales == 0.0
    assert slow.capital_tied_up == 80 * 15.0 # $1,200

    # SKU-HEALTHY assertions
    healthy = item_map["SKU-HEALTHY"]
    assert healthy.stockout_status == "ADEQUATE"
    assert healthy.days_of_inventory == pytest.approx(30.7, abs=1.0)

def test_inventory_aggregate_counts(mock_inventory_sales_df):
    report = analyze_inventory(mock_inventory_sales_df, trailing_days=45)
    assert report.critical_count == 1
    assert report.slow_moving_count == 1
    assert report.dead_stock_capital == 1200.0

def test_inventory_to_dataframe_conversion(mock_inventory_sales_df):
    report = analyze_inventory(mock_inventory_sales_df, trailing_days=45)
    df_inv = inventory_to_dataframe(report)
    assert len(df_inv) == 3
    assert "SKU" in df_inv.columns
    assert "Days of Inventory" in df_inv.columns
    assert "Reorder Point" in df_inv.columns
