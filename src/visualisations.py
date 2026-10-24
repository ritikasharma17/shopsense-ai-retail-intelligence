"""
ShopSense - Interactive Visualisation Suite (Plotly)
Constructs responsive, high-contrast, professional SaaS-grade retail charts.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import THEME_COLORS
from src.forecasting import ForecastResult
from src.inventory import InventoryItemSummary
from src.profitability import ProductProfitability, WhatIfScenarioResult

# Chart Layout Template
BASE_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color=THEME_COLORS["secondary"], size=12),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=30, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)


def plot_sales_trend(df_ts: pd.DataFrame, metric: str = "revenue") -> go.Figure:
    """
    Renders daily sales trajectory with 7-day rolling trendline.
    """
    fig = go.Figure()

    label = "Revenue ($)" if metric == "revenue" else ("Gross Profit ($)" if metric == "gross_profit" else "Units Sold")
    fmt = "$,.2f" if metric in ["revenue", "gross_profit"] else ",d"

    # Daily bar/area
    fig.add_trace(go.Bar(
        x=df_ts["date"],
        y=df_ts[metric],
        name=f"Daily {label}",
        marker_color="rgba(59, 130, 246, 0.25)",
        hovertemplate=f"%{{x|%b %d, %Y}}<br>Daily {label}: %{{y:{fmt}}}<extra></extra>"
    ))

    # 7-day smoothed rolling line
    rolling_col = f"rolling_7d_{'rev' if metric == 'revenue' else ('profit' if metric == 'gross_profit' else metric)}"
    if rolling_col in df_ts.columns:
        fig.add_trace(go.Scatter(
            x=df_ts["date"],
            y=df_ts[rolling_col],
            name="7-Day Moving Avg",
            line=dict(color=THEME_COLORS["primary"], width=2.8),
            hovertemplate=f"7D Trend: %{{y:{fmt}}}<extra></extra>"
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"<b>Historical {label} Trajectory</b>", font=dict(size=15)),
        xaxis=dict(showgrid=True, gridcolor=THEME_COLORS["border"], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=THEME_COLORS["border"], zeroline=False),
        height=380
    )
    return fig


def plot_category_mix(cat_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of revenue contribution by category.
    """
    colors = ["#1E3A8A", "#2563EB", "#0D9488", "#10B981", "#8B5CF6", "#F59E0B"]

    fig = go.Figure(data=[go.Pie(
        labels=cat_df["category"],
        values=cat_df["revenue"],
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Share: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Revenue Share by Category</b>", font=dict(size=15)),
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_forecast(res: ForecastResult) -> go.Figure:
    """
    Renders historical sales observations, forecast path, and 80% / 95% uncertainty bands.
    """
    fig = go.Figure()
    hist_tail = res.historical_df.tail(60)

    # 1. Historical Actuals
    fig.add_trace(go.Scatter(
        x=hist_tail["date"],
        y=hist_tail[res.target_variable],
        name="Historical Actual",
        line=dict(color="#64748B", width=1.8),
        hovertemplate="%{x|%b %d}: %{y:,.1f}<extra></extra>"
    ))

    # 2. 95% Confidence Interval (outer shading)
    fig.add_trace(go.Scatter(
        x=res.forecast_df["date"],
        y=res.forecast_df["upper_95"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=res.forecast_df["date"],
        y=res.forecast_df["lower_95"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.12)",
        name="95% Confidence Interval",
        hoverinfo="skip"
    ))

    # 3. 80% Confidence Interval (inner shading)
    fig.add_trace(go.Scatter(
        x=res.forecast_df["date"],
        y=res.forecast_df["upper_80"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=res.forecast_df["date"],
        y=res.forecast_df["lower_80"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.22)",
        name="80% Confidence Interval",
        hoverinfo="skip"
    ))

    # 4. Point Forecast
    fig.add_trace(go.Scatter(
        x=res.forecast_df["date"],
        y=res.forecast_df["forecast"],
        name=f"Forecast ({res.selected_model})",
        line=dict(color=THEME_COLORS["primary"], width=2.8, dash="solid"),
        hovertemplate="Forecast: %{y:,.1f}<br>80% Range: [%{customdata[0]:,.1f} - %{customdata[1]:,.1f}]<extra></extra>",
        customdata=np.stack((res.forecast_df["lower_80"], res.forecast_df["upper_80"]), axis=-1)
    ))

    # Vertical demarcation line between historical and future
    split_date = hist_tail["date"].iloc[-1]
    fig.add_vline(x=split_date, line_width=1.5, line_dash="dash", line_color="#94A3B8")

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"<b>Demand Forecast ({res.forecast_horizon_days} Days Out-of-Sample)</b>", font=dict(size=15)),
        xaxis=dict(showgrid=True, gridcolor=THEME_COLORS["border"]),
        yaxis=dict(showgrid=True, gridcolor=THEME_COLORS["border"], title=res.target_variable.replace("_", " ").title()),
        height=420
    )
    return fig


def plot_margin_quadrant(profit_items: List[ProductProfitability]) -> go.Figure:
    """
    Strategic 4-quadrant scatter plot: Revenue vs Gross Margin %.
    """
    df_p = pd.DataFrame([
        {
            "sku": p.product_id,
            "name": p.product_name,
            "category": p.category,
            "revenue": p.total_revenue,
            "margin_pct": p.gross_margin_pct,
            "quadrant": p.quadrant,
            "profit": p.gross_profit,
            "units": p.units_sold
        }
        for p in profit_items
    ])

    med_rev = df_p["revenue"].median()
    med_margin = df_p["margin_pct"].median()

    color_map = {
        "Star Performer": "#10B981", # Green
        "Volume Trap": "#EF4444",    # Crimson
        "Hidden Gem": "#3B82F6",     # Blue
        "Prune Candidate": "#94A3B8" # Slate
    }

    fig = go.Figure()

    for quad, group in df_p.groupby("quadrant"):
        fig.add_trace(go.Scatter(
            x=group["revenue"],
            y=group["margin_pct"],
            mode="markers+text",
            name=quad,
            text=group["name"].apply(lambda n: n[:14] + "..."),
            textposition="top center",
            marker=dict(size=12, color=color_map.get(quad, "#2563EB"), opacity=0.85),
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Revenue: $%{x:,.2f}<br>Gross Margin: %{y:.1f}%<br>Gross Profit: $%{customdata[2]:,.2f}<extra></extra>",
            customdata=np.stack((group["name"], group["category"], group["profit"]), axis=-1)
        ))

    # Add quadrant dividing lines
    fig.add_vline(x=med_rev, line_width=1, line_dash="dash", line_color="#CBD5E1")
    fig.add_hline(y=med_margin, line_width=1, line_dash="dash", line_color="#CBD5E1")

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Product Portfolio Strategic Margin Matrix</b>", font=dict(size=15)),
        xaxis=dict(title="Revenue ($)", showgrid=True, gridcolor=THEME_COLORS["border"]),
        yaxis=dict(title="Gross Margin (%)", showgrid=True, gridcolor=THEME_COLORS["border"]),
        height=450
    )
    return fig


def plot_pareto_curve(profit_items: List[ProductProfitability]) -> go.Figure:
    """
    Pareto 80/20 cumulative revenue contribution curve.
    """
    df_p = pd.DataFrame([
        {
            "sku": p.product_id,
            "name": p.product_name,
            "revenue": p.total_revenue,
            "share": p.revenue_share_pct,
            "cumulative": p.cumulative_rev_share_pct,
            "abc": p.abc_class
        }
        for p in profit_items
    ])

    fig = go.Figure()

    # Bar of SKU revenue
    fig.add_trace(go.Bar(
        x=df_p["name"].apply(lambda n: n[:16] + "..."),
        y=df_p["revenue"],
        name="SKU Revenue ($)",
        marker_color="rgba(37, 99, 235, 0.4)",
        yaxis="y",
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>"
    ))

    # Line of cumulative %
    fig.add_trace(go.Scatter(
        x=df_p["name"].apply(lambda n: n[:16] + "..."),
        y=df_p["cumulative"],
        name="Cumulative Revenue %",
        line=dict(color="#EF4444", width=2.5),
        yaxis="y2",
        hovertemplate="Cumulative: %{y:.1f}%<extra></extra>"
    ))

    # 80% Pareto threshold line
    fig.add_hline(y=80, line_width=1.5, line_dash="dot", line_color="#0F172A", yref="y2")

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Pareto (80/20) Revenue Concentration Curve</b>", font=dict(size=15)),
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor=THEME_COLORS["border"]),
        yaxis2=dict(title="Cumulative Share (%)", overlaying="y", side="right", range=[0, 105], showgrid=False),
        height=400
    )
    return fig


def plot_inventory_runway(items: List[InventoryItemSummary]) -> go.Figure:
    """
    Bar chart showing Days of Inventory Remaining per SKU, colored by risk status.
    """
    df_inv = pd.DataFrame([
        {
            "name": it.product_name[:18] + "...",
            "doi": min(90.0, it.days_of_inventory),
            "status": it.stockout_status,
            "stock": it.current_stock,
            "ads": it.avg_daily_sales,
            "rop": it.reorder_point
        }
        for it in items[:15] # Top 15 prioritized
    ]).sort_values("doi", ascending=True)

    status_colors = {
        "CRITICAL": "#EF4444",   # Red
        "WARNING": "#F59E0B",    # Amber
        "ADEQUATE": "#10B981",   # Emerald
        "OVERSTOCKED": "#64748B" # Slate
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_inv["name"],
        x=df_inv["doi"],
        orientation="h",
        marker=dict(color=[status_colors.get(s, "#3B82F6") for s in df_inv["status"]]),
        hovertemplate="<b>%{y}</b><br>Days of Inventory: %{x:.1f} days<br>Current Stock: %{customdata[0]} units<br>ADS: %{customdata[1]:.2f}/day<extra></extra>",
        customdata=np.stack((df_inv["stock"], df_inv["ads"]), axis=-1)
    ))

    # Lead time / critical risk line at 14 days
    fig.add_vline(x=14, line_width=1.5, line_dash="dash", line_color="#EF4444", annotation_text="14-Day Reorder Threshold", annotation_position="top right")

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Days of Supply Runway by Product (Prioritized)</b>", font=dict(size=15)),
        xaxis=dict(title="Days of Inventory (Capped at 90d)", showgrid=True, gridcolor=THEME_COLORS["border"]),
        yaxis=dict(autorange="reversed"),
        height=440
    )
    return fig


def plot_what_if_comparison(sim: WhatIfScenarioResult) -> go.Figure:
    """
    Grouped bar chart comparing Baseline vs Simulated financial outcome.
    """
    categories = ["Revenue", "Cost of Goods (COGS)", "Gross Profit"]
    baseline_vals = [sim.baseline_revenue, sim.baseline_cost, sim.baseline_profit]
    simulated_vals = [sim.simulated_revenue, sim.simulated_cost, sim.simulated_profit]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Historical Baseline",
        x=categories,
        y=baseline_vals,
        marker_color="#94A3B8",
        hovertemplate="Baseline %{x}: $%{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="Simulated Scenario",
        x=categories,
        y=simulated_vals,
        marker_color=THEME_COLORS["primary"],
        hovertemplate="Simulated %{x}: $%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="group",
        title=dict(text="<b>Scenario Impact: Baseline vs. Simulated Projection</b>", font=dict(size=15)),
        yaxis=dict(title="Amount ($)", showgrid=True, gridcolor=THEME_COLORS["border"]),
        height=360
    )
    return fig
