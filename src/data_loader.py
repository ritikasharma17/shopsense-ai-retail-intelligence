"""
ShopSense - Data Ingestion, Schema Validation, and Quality Auditing
Robust CSV loader with column mapping, boundary checking, and health scoring.
"""

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import REQUIRED_COLUMNS, COLUMN_ALIASES

@dataclass
class DataQualityReport:
    total_rows: int = 0
    total_columns: int = 0
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    total_days: int = 0
    unique_skus: int = 0
    unique_categories: int = 0
    missing_cells_count: int = 0
    missing_cells_pct: float = 0.0
    duplicate_rows_count: int = 0
    negative_values_count: int = 0
    zero_price_or_cost_count: int = 0
    health_score: float = 100.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    column_mapping_applied: Dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render data quality summary in clean markdown format."""
        status_label = "EXCELLENT" if self.health_score >= 90 else ("ACCEPTABLE" if self.health_score >= 70 else "ACTION REQUIRED")
        md = [
            f"# ShopSense Data Quality Audit Report",
            f"**Overall Dataset Health Score**: `{self.health_score:.1f}%` ({status_label})\n",
            f"### Executive Summary",
            f"- **Total Rows Analyzed**: {self.total_rows:,}",
            f"- **Total Columns**: {self.total_columns}",
            f"- **Date Range**: {self.date_range_start} to {self.date_range_end} ({self.total_days} days)",
            f"- **Catalog Breadth**: {self.unique_skus} SKUs across {self.unique_categories} categories",
            f"- **Duplicate Rows**: {self.duplicate_rows_count}",
            f"- **Missing Cells**: {self.missing_cells_count} ({self.missing_cells_pct:.2f}% of matrix)",
            f"- **Negative Value Flags**: {self.negative_values_count}",
            f"- **Zero/Blank Price or Cost Flags**: {self.zero_price_or_cost_count}\n",
        ]

        if self.errors:
            md.append("### Critical Errors")
            for err in self.errors:
                md.append(f"- 🔴 **{err}**")
            md.append("")

        if self.warnings:
            md.append("### Observations & Non-Blocking Warnings")
            for warn in self.warnings:
                md.append(f"- ⚠️ {warn}")
            md.append("")

        md.append("### Applied Schema Column Mapping")
        for std_col, orig_col in self.column_mapping_applied.items():
            md.append(f"- Standard `{std_col}` mapped from upload column `{orig_col}`")

        return "\n".join(md)


def detect_column_mappings(df_columns: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Automatically maps uploaded dataframe columns to standard ShopSense schema.
    Returns (mapping_dict: standard_col -> uploaded_col, unmapped_required_cols).
    """
    col_lower_map = {c.lower().strip().replace(" ", "_"): c for c in df_columns}
    applied_mapping: Dict[str, str] = {}
    unmapped_required: List[str] = []

    for standard_col, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            norm_alias = alias.lower().strip().replace(" ", "_")
            if norm_alias in col_lower_map:
                applied_mapping[standard_col] = col_lower_map[norm_alias]
                found = True
                break
        if not found and standard_col in REQUIRED_COLUMNS:
            unmapped_required.append(standard_col)

    return applied_mapping, unmapped_required


def validate_and_clean_data(
    df_raw: pd.DataFrame,
    custom_mapping: Optional[Dict[str, str]] = None
) -> Tuple[Optional[pd.DataFrame], DataQualityReport]:
    """
    Validates, parses types, cleans anomalies, and returns the sanitized DataFrame
    along with a comprehensive DataQualityReport.
    """
    report = DataQualityReport()
    report.total_rows = len(df_raw)
    report.total_columns = len(df_raw.columns)

    if df_raw.empty:
        report.errors.append("Uploaded file contains zero records (empty table).")
        report.health_score = 0.0
        return None, report

    # Duplicate check on raw data
    report.duplicate_rows_count = int(df_raw.duplicated().sum())

    # Column Mapping
    if custom_mapping:
        applied_mapping = custom_mapping
        missing_req = [c for c in REQUIRED_COLUMNS if c not in applied_mapping or not applied_mapping[c]]
    else:
        applied_mapping, missing_req = detect_column_mappings(list(df_raw.columns))

    report.column_mapping_applied = applied_mapping

    if missing_req:
        report.errors.append(f"Missing mandatory schema columns: {', '.join(missing_req)}")
        report.health_score = max(0.0, 50.0 - len(missing_req) * 10)
        return None, report

    # Rename mapped columns to standard names
    rename_dict = {orig: std for std, orig in applied_mapping.items() if orig in df_raw.columns}
    df = df_raw.rename(columns=rename_dict).copy()

    # Verify required standard columns are now present
    for req in REQUIRED_COLUMNS:
        if req not in df.columns:
            # Fallback calculation if possible
            if req == "revenue" and "unit_price" in df.columns and "quantity" in df.columns:
                discount = df["discount_amount"] if "discount_amount" in df.columns else 0.0
                df["revenue"] = (pd.to_numeric(df["unit_price"], errors="coerce") *
                                 pd.to_numeric(df["quantity"], errors="coerce") -
                                 pd.to_numeric(discount, errors="coerce").fillna(0.0))
                report.warnings.append("Inferred 'revenue' as (unit_price * quantity - discount_amount).")
            else:
                report.errors.append(f"Required field '{req}' could not be established.")
                return None, report

    # 1. Date parsing
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        invalid_dates = int(df["date"].isna().sum())
        if invalid_dates > 0:
            report.warnings.append(f"{invalid_dates} records had unparseable dates and were excluded.")
            df = df.dropna(subset=["date"])
        if df.empty:
            report.errors.append("No valid date records found in dataset.")
            return None, report
        df["date"] = df["date"].dt.normalize()
        report.date_range_start = df["date"].min().strftime("%Y-%m-%d")
        report.date_range_end = df["date"].max().strftime("%Y-%m-%d")
        report.total_days = (df["date"].max() - df["date"].min()).days + 1
    except Exception as e:
        report.errors.append(f"Failed to parse 'date' column: {str(e)}")
        return None, report

    # 2. Numeric casting and boundary verification
    numeric_cols = ["unit_price", "unit_cost", "quantity", "revenue"]
    for nc in ["discount_amount", "current_stock", "lead_time_days"]:
        if nc in df.columns:
            numeric_cols.append(nc)

    negative_count = 0
    zero_price_or_cost = 0

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        neg_mask = df[col] < 0
        neg_in_col = int(neg_mask.sum())
        if neg_in_col > 0:
            negative_count += neg_in_col
            # Clean negatives to zero
            df.loc[neg_mask, col] = 0.0

    report.negative_values_count = negative_count
    if negative_count > 0:
        report.warnings.append(f"Replaced {negative_count} anomalous negative values with zero.")

    # Price and Cost validation
    zero_price_mask = (df["unit_price"] <= 0) & (df["revenue"] <= 0)
    zero_price_count = int(zero_price_mask.sum())
    if zero_price_count > 0:
        zero_price_or_cost += zero_price_count
        report.warnings.append(f"{zero_price_count} records had non-positive selling price.")

    zero_cost_count = int((df["unit_cost"] <= 0).sum())
    if zero_cost_count > 0:
        zero_price_or_cost += zero_cost_count
        report.warnings.append(f"{zero_cost_count} records had zero or missing unit cost.")

    report.zero_price_or_cost_count = zero_price_or_cost

    # 3. String sanitation
    string_cols = ["product_id", "product_name", "category"]
    for sc in ["order_id", "sales_channel", "supplier_name", "customer_type"]:
        if sc in df.columns:
            string_cols.append(sc)

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "None", "null"]), col] = f"Unknown {col.replace('_', ' ').title()}"

    # 4. Computed Profitability Fields if absent
    if "cost_of_goods" not in df.columns:
        df["cost_of_goods"] = df["unit_cost"] * df["quantity"]

    if "gross_profit" not in df.columns:
        df["gross_profit"] = df["revenue"] - df["cost_of_goods"]

    # 5. Inventory and supplier defaults if missing
    if "current_stock" not in df.columns:
        df["current_stock"] = 50  # Sensible neutral default if small retailer didn't track
        report.warnings.append("Current stock not provided; initialized to baseline 50 units for inventory analysis.")

    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 14
        report.warnings.append("Supplier lead time not provided; defaulted to standard 14 business days.")

    if "supplier_name" not in df.columns:
        df["supplier_name"] = "Direct Vendor"

    if "order_id" not in df.columns:
        df["order_id"] = [f"ORD-{i+10001}" for i in range(len(df))]

    # Missing value metrics
    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    report.missing_cells_count = missing_cells
    report.missing_cells_pct = (missing_cells / total_cells * 100.0) if total_cells > 0 else 0.0

    report.unique_skus = int(df["product_id"].nunique())
    report.unique_categories = int(df["category"].nunique())

    # Calculate overall health score (starts at 100, penalized by issues)
    score = 100.0
    if report.duplicate_rows_count > 0:
        score -= min(15.0, (report.duplicate_rows_count / len(df)) * 50)
    if report.negative_values_count > 0:
        score -= min(15.0, (report.negative_values_count / len(df)) * 50)
    if report.zero_price_or_cost_count > 0:
        score -= min(10.0, (report.zero_price_or_cost_count / len(df)) * 25)
    if report.missing_cells_pct > 0:
        score -= min(15.0, report.missing_cells_pct * 2)

    report.health_score = round(max(10.0, min(100.0, score)), 1)

    # Sort chronological
    df = df.sort_values("date").reset_index(drop=True)

    return df, report


def load_dataset(source: Union[str, io.BytesIO], custom_mapping: Optional[Dict[str, str]] = None) -> Tuple[Optional[pd.DataFrame], DataQualityReport]:
    """
    High-level entrypoint to load CSV from file path or file stream.
    Safely captures unhandled parsing crashes and returns a graceful report.
    """
    try:
        if isinstance(source, str):
            df_raw = pd.read_csv(source)
        else:
            df_raw = pd.read_csv(source)
        return validate_and_clean_data(df_raw, custom_mapping=custom_mapping)
    except Exception as e:
        report = DataQualityReport()
        report.errors.append(f"Failed to read file as valid CSV: {str(e)}")
        report.health_score = 0.0
        return None, report
