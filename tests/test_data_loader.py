"""
Unit Tests for ShopSense Data Loader, Schema Mapping, and Quality Auditing
"""

import io
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    detect_column_mappings,
    validate_and_clean_data,
    load_dataset,
    DataQualityReport
)

def test_detect_column_mappings_standard():
    cols = ["date", "order_id", "product_id", "product_name", "category",
            "unit_price", "unit_cost", "quantity", "discount_amount", "revenue"]
    mapping, missing = detect_column_mappings(cols)
    assert len(missing) == 0
    assert mapping["product_id"] == "product_id"
    assert mapping["revenue"] == "revenue"

def test_detect_column_mappings_aliases():
    cols = ["Transaction_Date", "SKU", "Item_Name", "Department",
            "Price", "Cost", "Qty", "Sales"]
    mapping, missing = detect_column_mappings(cols)
    assert len(missing) == 0
    assert mapping["date"] == "Transaction_Date"
    assert mapping["product_id"] == "SKU"
    assert mapping["product_name"] == "Item_Name"
    assert mapping["category"] == "Department"
    assert mapping["unit_price"] == "Price"
    assert mapping["unit_cost"] == "Cost"
    assert mapping["quantity"] == "Qty"
    assert mapping["revenue"] == "Sales"

def test_missing_required_columns():
    # Missing 'unit_cost' and 'revenue'
    df_bad = pd.DataFrame({
        "date": ["2024-01-01"],
        "product_id": ["P1"],
        "product_name": ["Widget"],
        "category": ["General"]
    })
    df_clean, report = validate_and_clean_data(df_bad)
    assert df_clean is None
    assert len(report.errors) > 0
    assert any("unit_cost" in err for err in report.errors)

def test_negative_values_cleaned_and_reported():
    df_raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "product_id": ["P1", "P2"],
        "product_name": ["Item A", "Item B"],
        "category": ["Cat 1", "Cat 2"],
        "unit_price": [10.0, -5.0], # Negative price
        "unit_cost": [5.0, 2.0],
        "quantity": [-2, 4],        # Negative quantity
        "revenue": [20.0, 10.0]
    })
    df_clean, report = validate_and_clean_data(df_raw)
    assert df_clean is not None
    assert report.negative_values_count == 2
    # Check negatives clamped to 0
    assert df_clean.loc[1, "unit_price"] == 0.0
    assert df_clean.loc[0, "quantity"] == 0.0

def test_empty_dataframe_handled_gracefully():
    df_empty = pd.DataFrame()
    df_clean, report = validate_and_clean_data(df_empty)
    assert df_clean is None
    assert report.health_score == 0.0
    assert len(report.errors) > 0

def test_data_quality_report_markdown():
    rep = DataQualityReport(
        total_rows=1000,
        total_columns=12,
        date_range_start="2024-01-01",
        date_range_end="2024-12-31",
        total_days=366,
        unique_skus=20,
        unique_categories=4,
        health_score=96.5
    )
    md = rep.to_markdown()
    assert "ShopSense Data Quality Audit Report" in md
    assert "96.5%" in md
    assert "EXCELLENT" in md
