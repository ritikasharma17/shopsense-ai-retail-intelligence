"""
ShopSense - Retail Profitability, Margin Optimization, and Scenario Simulation
Computes unit economics, ABC/Pareto ranking, and What-If price elasticity modeling.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class ProductProfitability:
    product_id: str
    product_name: str
    category: str
    units_sold: int
    unit_price: float
    unit_cost: float
    unit_margin_dollars: float
    unit_margin_pct: float
    total_revenue: float
    total_cost: float
    gross_profit: float
    gross_margin_pct: float
    revenue_share_pct: float
    cumulative_rev_share_pct: float
    abc_class: str  # 'A' (top 80%), 'B' (next 15%), 'C' (bottom 5%)
    quadrant: str   # 'Star Performer', 'Volume Trap', 'Hidden Gem', 'Prune Candidate'


@dataclass
class WhatIfScenarioResult:
    baseline_units: int
    baseline_revenue: float
    baseline_cost: float
    baseline_profit: float
    baseline_margin_pct: float

    simulated_units: int
    simulated_revenue: float
    simulated_cost: float
    simulated_profit: float
    simulated_margin_pct: float

    revenue_delta: float
    revenue_delta_pct: float
    profit_delta: float
    profit_delta_pct: float
    margin_pts_delta: float

    assumptions_summary: str


def compute_product_profitability(df: pd.DataFrame) -> List[ProductProfitability]:
    """
    Computes comprehensive SKU-level unit economics, margin contribution,
    and ABC / Pareto classification.
    """
    if df.empty or "product_id" not in df.columns:
        return []

    grouped = df.groupby(["product_id", "product_name", "category"]).agg({
        "quantity": "sum",
        "revenue": "sum",
        "cost_of_goods": "sum",
        "gross_profit": "sum",
        "unit_price": "mean",
        "unit_cost": "mean"
    }).reset_index()

    total_revenue = grouped["revenue"].sum()
    if total_revenue <= 0:
        return []

    # Sort descending by revenue for Pareto calculation
    grouped = grouped.sort_values("revenue", ascending=False).reset_index(drop=True)
    grouped["revenue_share_pct"] = (grouped["revenue"] / total_revenue * 100.0)
    grouped["cumulative_share"] = grouped["revenue_share_pct"].cumsum()

    # Median thresholds for quadrant plotting
    med_revenue = grouped["revenue"].median()
    overall_margin = (grouped["gross_profit"].sum() / total_revenue * 100.0)

    results: List[ProductProfitability] = []

    for _, row in grouped.iterrows():
        sku = row["product_id"]
        name = row["product_name"]
        cat = row["category"]
        qty = int(row["quantity"])
        u_price = float(row["unit_price"])
        u_cost = float(row["unit_cost"])
        rev = float(row["revenue"])
        cogs = float(row["cost_of_goods"])
        prof = float(row["gross_profit"])
        margin_pct = (prof / rev * 100.0) if rev > 0 else 0.0

        u_margin_dollars = u_price - u_cost
        u_margin_pct = (u_margin_dollars / u_price * 100.0) if u_price > 0 else 0.0

        cum_share = float(row["cumulative_share"])
        if cum_share <= 80.0:
            abc = "A"
        elif cum_share <= 95.0:
            abc = "B"
        else:
            abc = "C"

        # Strategic Quadrant Assignment
        is_high_rev = rev >= med_revenue
        is_high_margin = margin_pct >= overall_margin

        if is_high_rev and is_high_margin:
            quadrant = "Star Performer"
        elif is_high_rev and not is_high_margin:
            quadrant = "Volume Trap"
        elif not is_high_rev and is_high_margin:
            quadrant = "Hidden Gem"
        else:
            quadrant = "Prune Candidate"

        results.append(ProductProfitability(
            product_id=sku,
            product_name=name,
            category=cat,
            units_sold=qty,
            unit_price=round(u_price, 2),
            unit_cost=round(u_cost, 2),
            unit_margin_dollars=round(u_margin_dollars, 2),
            unit_margin_pct=round(u_margin_pct, 2),
            total_revenue=round(rev, 2),
            total_cost=round(cogs, 2),
            gross_profit=round(prof, 2),
            gross_margin_pct=round(margin_pct, 2),
            revenue_share_pct=round(float(row["revenue_share_pct"]), 2),
            cumulative_rev_share_pct=round(cum_share, 2),
            abc_class=abc,
            quadrant=quadrant
        ))

    return results


def run_what_if_simulation(
    df: pd.DataFrame,
    product_id: Optional[str] = None,
    category: Optional[str] = None,
    price_change_pct: float = 0.0,
    cost_change_pct: float = 0.0,
    discount_adjustment_pct: float = 0.0,
    price_elasticity: float = -0.8
) -> WhatIfScenarioResult:
    """
    Simulates retail financial outcome under hypothesized pricing, cost inflation,
    and discount promotions using empirical price elasticity of demand.
    Elasticity E = (% change in quantity) / (% change in price).
    """
    sub = df.copy()
    if product_id and product_id != "All Products":
        sub = sub[sub["product_id"] == product_id]
        scope_desc = f"Product '{product_id}'"
    elif category and category != "All Categories":
        sub = sub[sub["category"] == category]
        scope_desc = f"Category '{category}'"
    else:
        scope_desc = "Entire Store Catalog"

    if sub.empty:
        return WhatIfScenarioResult(
            0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Empty selection"
        )

    b_units = int(sub["quantity"].sum())
    b_rev = float(sub["revenue"].sum())
    b_cost = float(sub["cost_of_goods"].sum())
    b_profit = float(sub["gross_profit"].sum())
    b_margin = (b_profit / b_rev * 100.0) if b_rev > 0 else 0.0

    # Calculate simulated quantity via price elasticity of demand
    # % delta Q = E * (% delta P)
    qty_change_pct = price_elasticity * price_change_pct
    sim_units = max(0, int(round(b_units * (1.0 + qty_change_pct / 100.0))))

    # Unit price impact
    sim_sub = sub.copy()
    new_price_mult = 1.0 + (price_change_pct / 100.0)
    new_cost_mult = 1.0 + (cost_change_pct / 100.0)
    qty_mult = (1.0 + qty_change_pct / 100.0)

    # Simulated revenue calculation
    sim_gross_sales = (sim_sub["unit_price"] * new_price_mult * sim_sub["quantity"] * qty_mult).sum()
    base_discount = sim_sub["discount_amount"].sum() if "discount_amount" in sim_sub.columns else 0.0
    sim_discount = max(0.0, base_discount * (1.0 + discount_adjustment_pct / 100.0) * qty_mult)
    sim_rev = max(0.0, float(sim_gross_sales - sim_discount))

    sim_cost = max(0.0, float((sim_sub["unit_cost"] * new_cost_mult * sim_sub["quantity"] * qty_mult).sum()))
    sim_profit = sim_rev - sim_cost
    sim_margin = (sim_profit / sim_rev * 100.0) if sim_rev > 0 else 0.0

    rev_delta = sim_rev - b_rev
    rev_delta_pct = (rev_delta / b_rev * 100.0) if b_rev > 0 else 0.0
    profit_delta = sim_profit - b_profit
    profit_delta_pct = (profit_delta / b_profit * 100.0) if b_profit > 0 else 0.0
    margin_pts_delta = sim_margin - b_margin

    assumptions = (
        f"Simulated {scope_desc}: Price adjusted by {price_change_pct:+.1f}%, "
        f"Unit COGS adjusted by {cost_change_pct:+.1f}%, Promo Discounts adjusted by {discount_adjustment_pct:+.1f}%. "
        f"Assumed price elasticity of {price_elasticity:.2f} results in an estimated {qty_change_pct:+.1f}% volume change "
        f"({sim_units - b_units:+,} units)."
    )

    return WhatIfScenarioResult(
        baseline_units=b_units,
        baseline_revenue=round(b_rev, 2),
        baseline_cost=round(b_cost, 2),
        baseline_profit=round(b_profit, 2),
        baseline_margin_pct=round(b_margin, 2),
        simulated_units=sim_units,
        simulated_revenue=round(sim_rev, 2),
        simulated_cost=round(sim_cost, 2),
        simulated_profit=round(sim_profit, 2),
        simulated_margin_pct=round(sim_margin, 2),
        revenue_delta=round(rev_delta, 2),
        revenue_delta_pct=round(rev_delta_pct, 2),
        profit_delta=round(profit_delta, 2),
        profit_delta_pct=round(profit_delta_pct, 2),
        margin_pts_delta=round(margin_pts_delta, 2),
        assumptions_summary=assumptions
    )
