"""
ShopSense - Prioritized Retail Action & Decision Advisory Engine
Synthesizes cross-module diagnostics into actionable, evidence-backed recommendations.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from src.inventory import InventoryIntelligenceReport
from src.profitability import ProductProfitability
from src.kpi_engine import BusinessKPIs

@dataclass
class BusinessRecommendation:
    id: str
    title: str
    category: str  # 'Inventory Risk', 'Margin Optimization', 'Dead Stock Liquidation', 'Sales Momentum'
    urgency: str   # 'CRITICAL', 'HIGH', 'MEDIUM', 'INFO'
    observed_evidence: str
    business_impact: str
    recommended_action: str
    caveats_and_assumptions: str
    target_sku_or_category: Optional[str] = None


def generate_recommendations(
    df: pd.DataFrame,
    kpis: BusinessKPIs,
    inv_report: InventoryIntelligenceReport,
    profit_items: List[ProductProfitability]
) -> List[BusinessRecommendation]:
    """
    Evaluates multi-dimensional business performance and generates
    a strictly grounded, prioritized action list.
    """
    recs: List[BusinessRecommendation] = []
    rec_counter = 1

    # 1. Critical Stockout Risks (Immediate Action)
    critical_items = [it for it in inv_report.items if it.stockout_status == "CRITICAL"]
    for item in critical_items[:3]: # top 3 critical
        gap_days = max(0.0, item.lead_time_days - item.days_of_inventory)
        est_lost_sales = round(gap_days * item.avg_daily_sales * item.unit_price, 2)
        recs.append(BusinessRecommendation(
            id=f"REC-{rec_counter:03d}",
            title=f"Expedite Urgent Replenishment for {item.product_name}",
            category="Inventory Risk",
            urgency="CRITICAL",
            observed_evidence=(
                f"Current stock is {item.current_stock} units. At current trailing velocity of {item.avg_daily_sales} units/day, "
                f"stock will deplete in ~{item.days_of_inventory} days. Supplier ({item.supplier_name}) lead time is {item.lead_time_days} days, "
                f"resulting in an anticipated stockout window of ~{gap_days:.1f} days."
            ),
            business_impact=(
                f"An estimated ~{int(gap_days * item.avg_daily_sales)} customer units will be unfulfillable, "
                f"risking approximately ${est_lost_sales:,.2f} in unrealized revenue and customer churn."
            ),
            recommended_action=(
                f"Contact {item.supplier_name} immediately to arrange an expedited shipment of at least {item.suggested_reorder_qty} units. "
                f"If lead time cannot be compressed, consider temporarily rationing online sales to protect in-store footfall."
            ),
            caveats_and_assumptions="Assumes sales rate continues at recent 45-day average without unrecorded promotional bursts.",
            target_sku_or_category=item.product_id
        ))
        rec_counter += 1

    # 2. Standard Reorder Warnings
    warning_items = [it for it in inv_report.items if it.stockout_status == "WARNING"]
    if warning_items:
        top_warn = warning_items[0]
        recs.append(BusinessRecommendation(
            id=f"REC-{rec_counter:03d}",
            title=f"Place Standard Reorder for {top_warn.product_name}",
            category="Inventory Risk",
            urgency="HIGH",
            observed_evidence=(
                f"Current stock of {top_warn.current_stock} units has crossed below the Reorder Point ({top_warn.reorder_point} units). "
                f"Safety buffer is currently {top_warn.safety_stock} units with {top_warn.days_of_inventory} days of supply remaining."
            ),
            business_impact=(
                f"Entering supplier lead time window ({top_warn.lead_time_days} days). Delaying reorder will erode safety buffer "
                f"and increase stockout probability above acceptable 5% threshold."
            ),
            recommended_action=(
                f"Issue a standard purchase order of {top_warn.suggested_reorder_qty} units to {top_warn.supplier_name} "
                f"to restore inventory to optimal Order-Up-To level."
            ),
            caveats_and_assumptions="Reorder quantity is calculated for a 14-day review cycle and 95% cycle service level target.",
            target_sku_or_category=top_warn.product_id
        ))
        rec_counter += 1

    # 3. Slow-Moving / Dead Stock Liquidation
    slow_items = [it for it in inv_report.items if it.is_slow_moving or it.stockout_status == "OVERSTOCKED"]
    if slow_items:
        top_slow = max(slow_items, key=lambda x: x.capital_tied_up)
        recs.append(BusinessRecommendation(
            id=f"REC-{rec_counter:03d}",
            title=f"Unlock Working Capital from Stagnant {top_slow.product_name}",
            category="Dead Stock Liquidation",
            urgency="MEDIUM",
            observed_evidence=(
                f"Inventory holds {top_slow.current_stock} units ({top_slow.days_of_inventory} days of supply) "
                f"against slow velocity of {top_slow.avg_daily_sales} units/day. Currently ties up ${top_slow.capital_tied_up:,.2f} in idle inventory."
            ),
            business_impact=(
                f"Ties up ${top_slow.capital_tied_up:,.2f} in working capital that could be reinvested into faster-turning inventory "
                f"like coffee or seasonal goods, while incurring shelf-space storage holding cost."
            ),
            recommended_action=(
                f"Launch a targeted 15% markdown or create a complementary bundle with high-velocity items "
                f"(e.g., pair with a popular kitchen or home accessory) to accelerate turnover."
            ),
            caveats_and_assumptions="Assumes holding cost of capital is 15-20% annualized; markdown will trim gross margin % on this specific SKU.",
            target_sku_or_category=top_slow.product_id
        ))
        rec_counter += 1

    # 4. Volume Trap / Margin Dilution Investigation
    volume_traps = [p for p in profit_items if p.quadrant == "Volume Trap"]
    if volume_traps:
        top_trap = volume_traps[0]
        recs.append(BusinessRecommendation(
            id=f"REC-{rec_counter:03d}",
            title=f"Protect Margins on Volume Driver: {top_trap.product_name}",
            category="Margin Optimization",
            urgency="MEDIUM",
            observed_evidence=(
                f"Generated ${top_trap.total_revenue:,.2f} in revenue ({top_trap.revenue_share_pct}% of total sales), "
                f"but operates at a below-average gross margin of {top_trap.gross_margin_pct:.1f}% "
                f"(Store benchmark is {kpis.gross_margin_pct:.1f}%)."
            ),
            business_impact=(
                f"High sales volume is masking significant gross profit compression. Each unit generates only "
                f"${top_trap.unit_margin_dollars:.2f} in contribution margin."
            ),
            recommended_action=(
                f"1. Test a modest $1-$2 price adjustment (or run What-If simulation). "
                f"2. Negotiate tiered volume discounts with {top_trap.product_name}'s supplier. "
                f"3. Reduce promotional discounting on this specific SKU."
            ),
            caveats_and_assumptions="Product demand may exhibit higher price sensitivity; monitor units sold closely following any price change.",
            target_sku_or_category=top_trap.product_id
        ))
        rec_counter += 1

    # 5. Core Performance Momentum
    star_items = [p for p in profit_items if p.quadrant == "Star Performer"]
    if star_items:
        top_star = star_items[0]
        recs.append(BusinessRecommendation(
            id=f"REC-{rec_counter:03d}",
            title=f"Capitalize on Star Performer: {top_star.product_name}",
            category="Sales Momentum",
            urgency="INFO",
            observed_evidence=(
                f"Leads the store in both revenue (${top_star.total_revenue:,.2f}) and gross profit contribution (${top_star.gross_profit:,.2f}) "
                f"with a strong gross margin of {top_star.gross_margin_pct:.1f}%."
            ),
            business_impact=(
                f"Key anchor SKU generating consistent high-margin cash flow. Maintaining 100% in-stock availability "
                f"is vital to overall store financial stability."
            ),
            recommended_action=(
                f"Prioritize supplier relationships to secure preferential delivery terms, feature prominently on homepage and storefront displays, "
                f"and explore adjacent complementary merchandise."
            ),
            caveats_and_assumptions="Ensure catalog diversity is maintained to prevent over-reliance on a single product.",
            target_sku_or_category=top_star.product_id
        ))
        rec_counter += 1

    return recs
