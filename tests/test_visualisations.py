"""
Unit Tests for ShopSense Visualisation Suite (Plotly)
Verifies that all charts render figures successfully without layout or argument errors.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.visualisations import (
    plot_sales_trend,
    plot_category_mix,
    plot_forecast,
    plot_margin_quadrant,
    plot_pareto_curve,
    plot_inventory_runway,
    plot_what_if_comparison
)
from src.forecasting import generate_forecast
from src.inventory import analyze_inventory
from src.profitability import compute_product_profitability, run_what_if_simulation
from src.kpi_engine import get_timeseries_breakdown, get_category_breakdown

@pytest.fixture
def chart_test_df():
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(45)]
    rows = []
    for d in dates:
        rows.append({
            "date": d,
            "order_id": f"ORD-{d.day}",
            "product_id": "SKU-TEST-1",
            "product_name": "Test Mug",
            "category": "Home",
            "unit_price": 20.0,
            "unit_cost": 8.0,
            "quantity": 3,
            "revenue": 60.0,
            "cost_of_goods": 24.0,
            "gross_profit": 36.0,
            "current_stock": 25,
            "lead_time_days": 10,
            "supplier_name": "Vendor Test"
        })
        rows.append({
            "date": d,
            "order_id": f"ORD-{d.day}-2",
            "product_id": "SKU-TEST-2",
            "product_name": "Test Shirt",
            "category": "Apparel",
            "unit_price": 40.0,
            "unit_cost": 25.0,
            "quantity": 2,
            "revenue": 80.0,
            "cost_of_goods": 50.0,
            "gross_profit": 30.0,
            "current_stock": 10,
            "lead_time_days": 14,
            "supplier_name": "Vendor Apparel"
        })
    return pd.DataFrame(rows)

def test_all_charts_render_successfully(chart_test_df):
    # 1. Sales Trend Plot
    ts_df = get_timeseries_breakdown(chart_test_df, freq="D")
    fig_sales = plot_sales_trend(ts_df, metric="revenue")
    assert fig_sales is not None

    # 2. Category Mix Donut
    cat_df = get_category_breakdown(chart_test_df)
    fig_cat = plot_category_mix(cat_df)
    assert fig_cat is not None

    # 3. Forecast Plot
    fc_res = generate_forecast(chart_test_df, target="quantity", horizon_days=14)
    fig_fc = plot_forecast(fc_res)
    assert fig_fc is not None

    # 4. Profitability Quadrant Matrix
    profit_items = compute_product_profitability(chart_test_df)
    fig_quad = plot_margin_quadrant(profit_items)
    assert fig_quad is not None

    # 5. Pareto 80/20 Curve
    fig_pareto = plot_pareto_curve(profit_items)
    assert fig_pareto is not None

    # 6. Inventory Runway Bar
    inv_rep = analyze_inventory(chart_test_df)
    fig_inv = plot_inventory_runway(inv_rep.items)
    assert fig_inv is not None

    # 7. What-If Comparison Bar
    sim_res = run_what_if_simulation(chart_test_df, price_change_pct=5.0)
    fig_sim = plot_what_if_comparison(sim_res)
    assert fig_sim is not None
