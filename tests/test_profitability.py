"""
Unit Tests for ShopSense Profitability, Margin Analysis, and What-If Simulation
"""

import pytest
import pandas as pd
import numpy as np

from src.profitability import compute_product_profitability, run_what_if_simulation

@pytest.fixture
def mock_profitability_df():
    rows = []
    # High revenue, high margin -> Star Performer
    rows.append({
        "product_id": "SKU-STAR",
        "product_name": "Star Candle",
        "category": "Home",
        "quantity": 100,
        "unit_price": 30.0,
        "unit_cost": 10.0,
        "revenue": 3000.0,
        "cost_of_goods": 1000.0,
        "gross_profit": 2000.0, # 66.7% margin
        "discount_amount": 0.0
    })
    # High revenue, low margin -> Volume Trap
    rows.append({
        "product_id": "SKU-TRAP",
        "product_name": "Volume Scarf",
        "category": "Apparel",
        "quantity": 30,
        "unit_price": 90.0,
        "unit_cost": 75.0,
        "revenue": 2700.0,
        "cost_of_goods": 2250.0,
        "gross_profit": 450.0, # 16.7% margin
        "discount_amount": 0.0
    })
    # Low revenue, high margin -> Hidden Gem
    rows.append({
        "product_id": "SKU-GEM",
        "product_name": "Artisan Oil",
        "category": "Pantry",
        "quantity": 10,
        "unit_price": 25.0,
        "unit_cost": 5.0,
        "revenue": 250.0,
        "cost_of_goods": 50.0,
        "gross_profit": 200.0, # 80.0% margin
        "discount_amount": 0.0
    })
    # Low revenue, low margin -> Prune Candidate
    rows.append({
        "product_id": "SKU-PRUNE",
        "product_name": "Cheap Trinket",
        "category": "Novelty",
        "quantity": 15,
        "unit_price": 10.0,
        "unit_cost": 8.0,
        "revenue": 150.0,
        "cost_of_goods": 120.0,
        "gross_profit": 30.0, # 20.0% margin
        "discount_amount": 0.0
    })
    return pd.DataFrame(rows)

def test_product_profitability_unit_margins(mock_profitability_df):
    items = compute_product_profitability(mock_profitability_df)
    item_map = {it.product_id: it for it in items}

    star = item_map["SKU-STAR"]
    assert star.unit_margin_dollars == 20.0
    assert star.unit_margin_pct == pytest.approx(66.67, abs=0.01)
    assert star.total_revenue == 3000.0
    assert star.gross_profit == 2000.0

    trap = item_map["SKU-TRAP"]
    assert trap.unit_margin_dollars == 15.0
    assert trap.unit_margin_pct == pytest.approx(16.67, abs=0.01)

def test_strategic_quadrant_assignment(mock_profitability_df):
    items = compute_product_profitability(mock_profitability_df)
    item_map = {it.product_id: it for it in items}

    assert item_map["SKU-STAR"].quadrant == "Star Performer"
    assert item_map["SKU-TRAP"].quadrant == "Volume Trap"
    assert item_map["SKU-GEM"].quadrant == "Hidden Gem"
    assert item_map["SKU-PRUNE"].quadrant == "Prune Candidate"

def test_pareto_abc_ranking(mock_profitability_df):
    items = compute_product_profitability(mock_profitability_df)
    # Total revenue = 3000 + 2700 + 250 + 150 = 6100
    # SKU-STAR = 3000 / 6100 = 49.2% -> Class A
    # SKU-TRAP = 2700 / 6100 = 44.3% -> Cumulative 93.4% -> Class B
    # SKU-GEM = 250 / 6100 = 4.1% -> Cumulative 97.5% -> Class C
    # SKU-PRUNE = 150 / 6100 = 2.5% -> Cumulative 100% -> Class C
    item_map = {it.product_id: it for it in items}
    assert item_map["SKU-STAR"].abc_class == "A"
    assert item_map["SKU-TRAP"].abc_class == "B"
    assert item_map["SKU-GEM"].abc_class == "C"

def test_what_if_simulation_price_elasticity(mock_profitability_df):
    # Simulate +10% price increase with elasticity E = -1.0
    # Expected volume change = -10%
    sim = run_what_if_simulation(
        mock_profitability_df,
        product_id="SKU-STAR",
        price_change_pct=10.0,
        cost_change_pct=0.0,
        price_elasticity=-1.0
    )

    # Baseline: 100 units @ $30 = $3000. Cost = 100 * $10 = $1000. Profit = $2000.
    # Simulated units = 100 * (1 - 0.10) = 90 units
    # New price = $33. Simulated revenue = 90 * $33 = $2970.
    # Simulated cost = 90 * $10 = $900.
    # Simulated profit = $2970 - $900 = $2070.
    assert sim.baseline_units == 100
    assert sim.simulated_units == 90
    assert sim.simulated_revenue == pytest.approx(2970.0, abs=1.0)
    assert sim.simulated_profit == pytest.approx(2070.0, abs=1.0)
    assert sim.profit_delta > 0 # Higher net profit despite slight volume drop
