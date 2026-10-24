"""
Unit Tests for ShopSense Ask ShopSense Business Q&A Engine (Deterministic Fallback)
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.kpi_engine import calculate_core_kpis
from src.inventory import analyze_inventory
from src.profitability import compute_product_profitability
from src.recommendations import generate_recommendations
from src.insights_engine import answer_business_query

@pytest.fixture
def store_context_df():
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(45)]
    rows = []
    for d in dates:
        rows.append({
            "date": d,
            "order_id": f"ORD-{d.day}",
            "product_id": "SKU-MUG",
            "product_name": "Ceramic Mug",
            "category": "Home",
            "quantity": 4,
            "unit_price": 20.0,
            "unit_cost": 8.0,
            "revenue": 80.0,
            "cost_of_goods": 32.0,
            "gross_profit": 48.0,
            "current_stock": 8,
            "lead_time_days": 14,
            "supplier_name": "Studio Kiln"
        })
    return pd.DataFrame(rows)

def test_query_stockout_risk_deterministic(store_context_df):
    kpis = calculate_core_kpis(store_context_df)
    inv_rep = analyze_inventory(store_context_df)
    prof_items = compute_product_profitability(store_context_df)
    recs = generate_recommendations(store_context_df, kpis, inv_rep, prof_items)

    resp = answer_business_query(
        query="Which products are running out of stock?",
        df=store_context_df,
        kpis=kpis,
        inv_report=inv_rep,
        profit_items=prof_items,
        recommendations=recs,
        gemini_api_key=None # Testing deterministic zero-dependency fallback
    )

    assert resp.relevant_view == "Inventory Intelligence"
    assert "Inventory Alert" in resp.answer_markdown or "Ceramic Mug" in resp.answer_markdown
    assert "Critical Items Count" in resp.grounded_metrics

def test_query_profitability_deterministic(store_context_df):
    kpis = calculate_core_kpis(store_context_df)
    inv_rep = analyze_inventory(store_context_df)
    prof_items = compute_product_profitability(store_context_df)
    recs = generate_recommendations(store_context_df, kpis, inv_rep, prof_items)

    resp = answer_business_query(
        query="What is our store gross margin and profit?",
        df=store_context_df,
        kpis=kpis,
        inv_report=inv_rep,
        profit_items=prof_items,
        recommendations=recs,
        gemini_api_key=None
    )

    assert resp.relevant_view == "Profitability"
    assert "60.0%" in resp.answer_markdown or f"{kpis.gross_margin_pct:.1f}%" in resp.answer_markdown
    assert "Gross Margin" in resp.grounded_metrics or "Total Gross Profit" in resp.grounded_metrics
