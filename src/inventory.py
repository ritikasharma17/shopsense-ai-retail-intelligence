"""
ShopSense - Retail Inventory Intelligence & Replenishment Advisory
Calculates Days of Inventory, Safety Stock, Reorder Points, and Stockout Risks.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_SERVICE_LEVEL_Z,
    DEFAULT_REVIEW_PERIOD_DAYS,
    DEFAULT_SLOW_MOVING_DAYS,
    DEFAULT_SLOW_MOVING_VELOCITY,
    CRITICAL_DOI_DAYS,
    WARNING_DOI_DAYS,
    DEAD_STOCK_DOI_DAYS,
)

@dataclass
class InventoryItemSummary:
    product_id: str
    product_name: str
    category: str
    current_stock: int
    lead_time_days: int
    supplier_name: str
    unit_cost: float
    unit_price: float
    avg_daily_sales: float
    daily_demand_std: float
    days_of_inventory: float
    runout_date: Optional[str]
    safety_stock: int
    reorder_point: int
    stockout_status: str  # 'CRITICAL', 'WARNING', 'ADEQUATE', 'OVERSTOCKED'
    is_slow_moving: bool
    suggested_reorder_qty: int
    capital_tied_up: float
    alert_explanation: str


@dataclass
class InventoryIntelligenceReport:
    items: List[InventoryItemSummary]
    critical_count: int
    warning_count: int
    slow_moving_count: int
    total_capital_tied_up: float
    dead_stock_capital: float
    service_level_target_pct: float
    trailing_days_evaluated: int


def analyze_inventory(
    df: pd.DataFrame,
    trailing_days: int = 45,
    service_level_z: float = DEFAULT_SERVICE_LEVEL_Z,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    slow_moving_velocity: float = DEFAULT_SLOW_MOVING_VELOCITY
) -> InventoryIntelligenceReport:
    """
    Evaluates current inventory levels against historical sales velocity,
    replenishment lead times, and stochastic demand variation.
    """
    if df.empty or "product_id" not in df.columns:
        return InventoryIntelligenceReport(
            items=[],
            critical_count=0,
            warning_count=0,
            slow_moving_count=0,
            total_capital_tied_up=0.0,
            dead_stock_capital=0.0,
            service_level_target_pct=round((1 - (1 - 0.95)) * 100, 1),
            trailing_days_evaluated=trailing_days
        )

    max_date = df["date"].max()
    cutoff_date = max_date - timedelta(days=trailing_days)
    df_trailing = df[df["date"] >= cutoff_date].copy()

    # Get latest known metadata per product
    product_meta = df.sort_values("date").groupby("product_id").last().reset_index()

    items: List[InventoryItemSummary] = []
    critical_count = 0
    warning_count = 0
    slow_moving_count = 0
    total_capital = 0.0
    dead_capital = 0.0

    # Continuous date index for trailing window to calculate true daily velocity
    daily_dates = pd.date_range(start=cutoff_date, end=max_date, freq="D")
    actual_trailing_days = len(daily_dates)

    for _, meta in product_meta.iterrows():
        sku = meta["product_id"]
        name = meta.get("product_name", sku)
        category = meta.get("category", "General")
        curr_stock = int(meta.get("current_stock", 0))
        lead_time = int(meta.get("lead_time_days", 14))
        supplier = meta.get("supplier_name", "Primary Supplier")
        unit_cost = float(meta.get("unit_cost", 0.0))
        unit_price = float(meta.get("unit_price", 0.0))

        # Trailing sales series for this SKU
        sku_trail = df_trailing[df_trailing["product_id"] == sku]
        if not sku_trail.empty:
            daily_series = sku_trail.groupby("date")["quantity"].sum().reindex(daily_dates, fill_value=0.0)
            ads = float(daily_series.mean())
            sigma_d = float(daily_series.std(ddof=1)) if len(daily_series) > 1 else 0.0
            total_units_period = float(daily_series.sum())
        else:
            ads = 0.0
            sigma_d = 0.0
            total_units_period = 0.0

        # Capital tied up
        cap = curr_stock * unit_cost
        total_capital += cap

        # Days of Inventory (DOI)
        if ads > 0.01:
            doi = round(curr_stock / ads, 1)
            days_to_runout = int(np.floor(curr_stock / ads))
            runout_dt = (max_date + timedelta(days=days_to_runout)).strftime("%Y-%m-%d")
        else:
            doi = 999.0
            runout_dt = "No immediate depletion (Zero velocity)"

        # Safety Stock formula: SS = Z * sigma_d * sqrt(Lead Time)
        # Accounts for stochastic demand variance during supplier lead time
        safety_stock_calc = service_level_z * sigma_d * np.sqrt(max(1, lead_time))
        safety_stock = int(np.ceil(safety_stock_calc))

        # Reorder Point (ROP): Lead time demand + Safety Stock
        lead_time_demand = ads * lead_time
        reorder_point = int(np.ceil(lead_time_demand + safety_stock))

        # Stockout Risk Status
        if curr_stock <= np.ceil(lead_time_demand):
            status = "CRITICAL"
            critical_count += 1
            days_gap = max(0.0, lead_time - (curr_stock / ads if ads > 0 else 0))
            explanation = (
                f"CRITICAL DEFICIT: Current stock ({curr_stock} units) covers only {doi} days of sales "
                f"at {ads:.2f} units/day. Supplier lead time is {lead_time} days. "
                f"An unavoidable stockout of ~{days_gap:.1f} days is imminent unless expedited."
            )
        elif curr_stock <= reorder_point:
            status = "WARNING"
            warning_count += 1
            explanation = (
                f"REORDER THRESHOLD BREACHED: Stock ({curr_stock} units) has dropped below the Reorder Point "
                f"of {reorder_point} units (Lead Time Demand: {lead_time_demand:.1f} + Safety Buffer: {safety_stock}). "
                f"Place replenishment order with {supplier} now."
            )
        elif doi > DEAD_STOCK_DOI_DAYS or (ads < slow_moving_velocity and curr_stock > 20):
            status = "OVERSTOCKED"
            explanation = (
                f"CAPITAL ACCUMULATION: {curr_stock} units in inventory represents {doi} days of supply "
                f"at slow sales rate of {ads:.2f} units/day. Ties up ${cap:,.2f} in working capital."
            )
        else:
            status = "ADEQUATE"
            explanation = (
                f"HEALTHY BUFFER: Stock ({curr_stock} units) provides {doi} days of runway. "
                f"Comfortably above Reorder Point ({reorder_point} units) for target 95% service level."
            )

        # Slow-moving test
        is_slow = (ads < slow_moving_velocity) and (curr_stock > 10)
        if is_slow:
            slow_moving_count += 1
            dead_capital += cap

        # Suggested Reorder Quantity: Order-Up-To Level minus on-hand
        # Target S = (Lead Time + Review Period) * ADS + Safety Stock
        target_s = (lead_time + review_period_days) * ads + safety_stock
        suggested_order = max(0, int(np.ceil(target_s - curr_stock)))

        items.append(InventoryItemSummary(
            product_id=sku,
            product_name=name,
            category=category,
            current_stock=curr_stock,
            lead_time_days=lead_time,
            supplier_name=supplier,
            unit_cost=round(unit_cost, 2),
            unit_price=round(unit_price, 2),
            avg_daily_sales=round(ads, 2),
            daily_demand_std=round(sigma_d, 2),
            days_of_inventory=doi,
            runout_date=runout_dt,
            safety_stock=safety_stock,
            reorder_point=reorder_point,
            stockout_status=status,
            is_slow_moving=is_slow,
            suggested_reorder_qty=suggested_order,
            capital_tied_up=round(cap, 2),
            alert_explanation=explanation
        ))

    # Sort priority: CRITICAL first, then WARNING, then OVERSTOCKED, then ADEQUATE
    priority_map = {"CRITICAL": 0, "WARNING": 1, "OVERSTOCKED": 2, "ADEQUATE": 3}
    items.sort(key=lambda x: (priority_map.get(x.stockout_status, 9), x.days_of_inventory))

    # Map service level Z to percentage string
    z_to_pct = {1.28: 90.0, 1.65: 95.0, 1.96: 97.5, 2.33: 99.0}
    srv_pct = z_to_pct.get(service_level_z, 95.0)

    return InventoryIntelligenceReport(
        items=items,
        critical_count=critical_count,
        warning_count=warning_count,
        slow_moving_count=slow_moving_count,
        total_capital_tied_up=round(total_capital, 2),
        dead_stock_capital=round(dead_capital, 2),
        service_level_target_pct=srv_pct,
        trailing_days_evaluated=actual_trailing_days
    )


def inventory_to_dataframe(report: InventoryIntelligenceReport) -> pd.DataFrame:
    """Convert item summaries to a clean pandas DataFrame for table presentation."""
    records = []
    for item in report.items:
        records.append({
            "SKU": item.product_id,
            "Product Name": item.product_name,
            "Category": item.category,
            "Current Stock": item.current_stock,
            "Lead Time (Days)": item.lead_time_days,
            "Avg Daily Sales": item.avg_daily_sales,
            "Days of Inventory": item.days_of_inventory if item.days_of_inventory < 900 else "> 90 days",
            "Runout Date": item.runout_date,
            "Safety Stock": item.safety_stock,
            "Reorder Point": item.reorder_point,
            "Status": item.stockout_status,
            "Suggested Order": item.suggested_reorder_qty,
            "Capital ($)": item.capital_tied_up,
            "Supplier": item.supplier_name,
            "Explanation": item.alert_explanation
        })
    return pd.DataFrame(records)
