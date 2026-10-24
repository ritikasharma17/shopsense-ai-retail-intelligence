"""
Unit Tests for ShopSense Recommendation Engine
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.kpi_engine import calculate_core_kpis
from src.inventory import analyze_inventory
from src.profitability import compute_product_profitability
from src.recommendations import generate_recommendations

@pytest.fixture
def sample_business_data():
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(45)]
    rows = []
    for d in dates:
        # High velocity item with low stock -> Critical
        rows.append({
            "date": d,
            "order_id": f"ORD-{d.day}",
            "product_id": "SKU-URGENT",
            "product_name": "Urgent French Roast",
            "category": "Coffee",
            "quantity": 6,
            "unit_price": 15.0,
            "unit_cost": 7.0,
            "revenue": 90.0,
            "cost_of_goods": 42.0,
            "gross_profit": 48.0,
            "current_stock": 10,
            "lead_time_days": 12,
            "supplier_name": "Pacific Roasters"
        })
    return pd.DataFrame(rows)

def test_recommendation_generation_and_urgency(sample_business_data):
    kpis = calculate_core_kpis(sample_business_data)
    inv_rep = analyze_inventory(sample_business_data, trailing_days=45)
    prof_items = compute_product_profitability(sample_business_data)

    recs = generate_recommendations(sample_business_data, kpis, inv_rep, prof_items)

    assert len(recs) > 0
    top_rec = recs[0]
    assert top_rec.urgency == "CRITICAL"
    assert "Urgent French Roast" in top_rec.title
    assert "Pacific Roasters" in top_rec.observed_evidence or "Pacific Roasters" in top_rec.recommended_action
    assert len(top_rec.business_impact) > 0
    assert len(top_rec.caveats_and_assumptions) > 0
