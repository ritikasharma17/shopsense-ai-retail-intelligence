"""
ShopSense - AI-Powered Retail Business Decision Assistant
Main Streamlit Application
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="ShopSense — Retail Business Decision Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS for High Polish
st.markdown("""
<style>
    /* Clean corporate slate typography and styling */
    .main {
        background-color: #F8FAFC;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.80rem;
        font-weight: 500;
        margin-top: 6px;
    }
    .delta-pos {
        color: #10B981;
    }
    .delta-neg {
        color: #EF4444;
    }
    .alert-banner-crit {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 14px;
        color: #991B1B;
    }
    .alert-banner-warn {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 14px;
        color: #92400E;
    }
    .badge-crit {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-warn {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-ok {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding-top: 8px;
        padding-bottom: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src
from src.config import APP_NAME, APP_TAGLINE, APP_VERSION
from src.data_loader import load_dataset, DataQualityReport
from src.kpi_engine import (
    calculate_core_kpis,
    get_timeseries_breakdown,
    get_category_breakdown,
    get_channel_breakdown,
)
from src.forecasting import generate_forecast
from src.inventory import analyze_inventory, inventory_to_dataframe
from src.profitability import compute_product_profitability, run_what_if_simulation
from src.recommendations import generate_recommendations
from src.insights_engine import answer_business_query
from src.visualisations import (
    plot_sales_trend,
    plot_category_mix,
    plot_forecast,
    plot_margin_quadrant,
    plot_pareto_curve,
    plot_inventory_runway,
    plot_what_if_comparison,
)

# ----------------------------------------------------
# Data Loading & Session Cache
# ----------------------------------------------------
DEMO_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "demo_retail_sales.csv")

@st.cache_data
def load_default_data():
    df, report = load_dataset(DEMO_DATA_PATH)
    return df, report

# Initialize session state
if "df" not in st.session_state:
    df_init, rep_init = load_default_data()
    st.session_state["df"] = df_init
    st.session_state["report"] = rep_init
    st.session_state["is_custom_data"] = False
    st.session_state["chat_history"] = []

# ----------------------------------------------------
# Sidebar Controls & Navigation
# ----------------------------------------------------
with st.sidebar:
    st.markdown(f"## 🛍️ **{APP_NAME}**")
    st.caption(f"{APP_TAGLINE} • v{APP_VERSION}")
    st.markdown("---")

    # Data Source Selector
    st.markdown("### 📂 **Dataset Source**")
    data_mode = st.radio(
        "Active Data Mode",
        options=["UrbanPulse Boutique (Demo)", "Upload Custom CSV"],
        index=0 if not st.session_state["is_custom_data"] else 1
    )

    if data_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload Retail POS / Sales CSV",
            type=["csv"],
            help="Ensure CSV has date, product_id, product_name, category, unit_price, unit_cost, quantity, and revenue."
        )
        if uploaded_file is not None:
            df_up, rep_up = load_dataset(uploaded_file)
            if df_up is not None and rep_up.health_score > 0:
                st.session_state["df"] = df_up
                st.session_state["report"] = rep_up
                st.session_state["is_custom_data"] = True
                st.success(f"Loaded {len(df_up):,} rows (Health: {rep_up.health_score:.1f}%)")
            else:
                st.error("Failed to validate uploaded file. See 'Data Quality & Methodology' tab.")
    else:
        if st.session_state["is_custom_data"]:
            df_init, rep_init = load_default_data()
            st.session_state["df"] = df_init
            st.session_state["report"] = rep_init
            st.session_state["is_custom_data"] = False

    df_active = st.session_state["df"]
    report_active = st.session_state["report"]

    st.markdown("---")
    st.markdown("### 🔍 **Filter Scope**")

    # Date Range Filter
    min_date = df_active["date"].min().date()
    max_date = df_active["date"].max().date()
    selected_dates = st.date_input(
        "Reporting Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Category Filter
    all_categories = ["All Categories"] + sorted(list(df_active["category"].unique()))
    selected_category = st.selectbox("Category Filter", options=all_categories, index=0)

    # Optional Gemini API Key
    st.markdown("---")
    with st.expander("🔑 **AI Assistant Settings**", expanded=False):
        api_key_env = os.environ.get("GEMINI_API_KEY", "")
        gemini_key_input = st.text_input(
            "Optional Gemini API Key",
            value=api_key_env,
            type="password",
            help="If provided, enables narrative AI synthesis. Without an API key, ShopSense operates on its built-in deterministic fact engine with zero data egress."
        )

# Apply filters
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    df_filtered = df_active[
        (df_active["date"].dt.date >= start_d) &
        (df_active["date"].dt.date <= end_d)
    ].copy()
else:
    df_filtered = df_active.copy()

if selected_category != "All Categories":
    df_filtered = df_filtered[df_filtered["category"] == selected_category]

if df_filtered.empty:
    st.warning("No transactions match the selected date range and category filter. Please widen the filter criteria.")
    st.stop()

# Compute core analytics for the filtered data
kpis = calculate_core_kpis(df_filtered)
inv_report = analyze_inventory(df_filtered)
profit_items = compute_product_profitability(df_filtered)
recommendations = generate_recommendations(df_filtered, kpis, inv_report, profit_items)

# ----------------------------------------------------
# Main Application Navigation Tabs
# ----------------------------------------------------
tab_overview, tab_sales, tab_forecast, tab_inventory, tab_profit, tab_ai, tab_methodology = st.tabs([
    "📈 Executive Overview",
    "🛒 Sales & Products",
    "🔮 Demand Forecast",
    "📦 Inventory Intelligence",
    "💰 Profitability & What-If",
    "🤖 Ask ShopSense",
    "📋 Data Quality & Methodology"
])

# ====================================================
# TAB 1: EXECUTIVE OVERVIEW
# ====================================================
with tab_overview:
    st.markdown(f"### **Executive Performance Dashboard**")
    st.caption(f"Reporting Window: **{kpis.date_start}** to **{kpis.date_end}** ({kpis.total_days} days) • Catalog: **{kpis.unique_skus_sold} Active SKUs**")

    # Critical Alerts Banner
    critical_inv = [it for it in inv_report.items if it.stockout_status == "CRITICAL"]
    volume_traps = [p for p in profit_items if p.quadrant == "Volume Trap"]

    if critical_inv:
        st.markdown(
            f'<div class="alert-banner-crit">'
            f'⚠️ <b>URGENT SUPPLY CHAIN DEFICIT:</b> {len(critical_inv)} product(s) (including <b>{critical_inv[0].product_name}</b>) '
            f'will stock out within {critical_inv[0].days_of_inventory} days. Replenishment lead time is {critical_inv[0].lead_time_days} days. '
            f'Immediate expedited reorder required to avert unfulfillable sales.'
            f'</div>',
            unsafe_allow_html=True
        )

    # KPI Top Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        rev_delta_str = f"{kpis.revenue_delta_pct:+.1f}% vs prior" if kpis.revenue_delta_pct is not None else "Baseline period"
        rev_delta_class = "delta-pos" if (kpis.revenue_delta_pct or 0) >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Net Revenue</div>
            <div class="metric-value">${kpis.total_revenue:,.0f}</div>
            <div class="metric-sub {rev_delta_class}">{rev_delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        prof_delta_str = f"{kpis.profit_delta_pct:+.1f}% vs prior" if kpis.profit_delta_pct is not None else "Baseline period"
        prof_delta_class = "delta-pos" if (kpis.profit_delta_pct or 0) >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gross Profit</div>
            <div class="metric-value">${kpis.total_gross_profit:,.0f}</div>
            <div class="metric-sub {prof_delta_class}">{prof_delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        margin_delta_str = f"{kpis.margin_delta_pts:+.1f} pts vs prior" if kpis.margin_delta_pts is not None else "Baseline period"
        margin_delta_class = "delta-pos" if (kpis.margin_delta_pts or 0) >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gross Margin</div>
            <div class="metric-value">{kpis.gross_margin_pct:.1f}%</div>
            <div class="metric-sub {margin_delta_class}">{margin_delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        units_delta_str = f"{kpis.units_delta_pct:+.1f}% vs prior" if kpis.units_delta_pct is not None else "Baseline period"
        units_delta_class = "delta-pos" if (kpis.units_delta_pct or 0) >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Units Sold</div>
            <div class="metric-value">{kpis.total_units_sold:,}</div>
            <div class="metric-sub {units_delta_class}">{units_delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        aov_delta_str = f"{kpis.aov_delta_pct:+.1f}% vs prior" if kpis.aov_delta_pct is not None else "Baseline period"
        aov_delta_class = "delta-pos" if (kpis.aov_delta_pct or 0) >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Order Value</div>
            <div class="metric-value">${kpis.average_order_value:.2f}</div>
            <div class="metric-sub {aov_delta_class}">{aov_delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with c6:
        crit_color = "delta-neg" if inv_report.critical_count > 0 else "delta-pos"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Inventory Alerts</div>
            <div class="metric-value">{inv_report.critical_count + inv_report.warning_count}</div>
            <div class="metric-sub {crit_color}">{inv_report.critical_count} Critical • {inv_report.warning_count} Warning</div>
        </div>
        """, unsafe_allow_html=True)

    # Primary Charts Row
    st.markdown("<br>", unsafe_allow_html=True)
    col_chart_left, col_chart_right = st.columns([7, 3])

    with col_chart_left:
        df_ts_daily = get_timeseries_breakdown(df_filtered, freq="D")
        metric_choice = st.selectbox("Trajectory Metric", ["revenue", "gross_profit", "quantity"], format_func=lambda x: x.replace("_", " ").title())
        st.plotly_chart(plot_sales_trend(df_ts_daily, metric=metric_choice), use_container_width=True)

    with col_chart_right:
        cat_df = get_category_breakdown(df_filtered)
        st.plotly_chart(plot_category_mix(cat_df), use_container_width=True)

    # Prioritized Strategic Recommendations
    st.markdown("### 🎯 **Top Business Recommendations**")
    st.caption("Evidence-backed operational and pricing decisions synthesized from live dataset metrics.")

    for rec in recommendations[:4]:
        urgency_badge = f'<span class="badge-{"crit" if rec.urgency == "CRITICAL" else ("warn" if rec.urgency == "HIGH" else "ok")}">{rec.urgency}</span>'
        with st.expander(f"**{rec.title}** — {rec.category}", expanded=(rec.urgency == "CRITICAL")):
            st.markdown(f"**Urgency**: {urgency_badge}", unsafe_allow_html=True)
            st.markdown(f"- **What is Happening (Evidence)**: {rec.observed_evidence}")
            st.markdown(f"- **Why it Matters (Impact)**: {rec.business_impact}")
            st.markdown(f"- **Recommended Action**: **{rec.recommended_action}**")
            st.caption(f"*Assumptions & Caveats*: {rec.caveats_and_assumptions}")


# ====================================================
# TAB 2: SALES & PRODUCTS
# ====================================================
with tab_sales:
    st.markdown("### **Sales Trends & Merchandise Analytics**")
    st.caption("Investigate product velocity, 80/20 concentration, and omni-channel distribution.")

    col_s1, col_s2 = st.columns([6, 4])
    with col_s1:
        st.plotly_chart(plot_pareto_curve(profit_items), use_container_width=True)
    with col_s2:
        ch_df = get_channel_breakdown(df_filtered)
        if not ch_df.empty:
            st.markdown("#### **Channel Performance**")
            st.dataframe(
                ch_df.style.format({
                    "revenue": "${:,.2f}",
                    "revenue_share_pct": "{:.1f}%",
                    "units_sold": "{:,}",
                    "orders": "{:,}",
                    "aov": "${:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Sales channel field not present in current dataset.")

    st.markdown("#### **Product Catalog Leaderboard**")
    sku_rows = []
    for p in profit_items:
        sku_rows.append({
            "SKU": p.product_id,
            "Product Name": p.product_name,
            "Category": p.category,
            "Units Sold": p.units_sold,
            "Revenue": p.total_revenue,
            "Gross Profit": p.gross_profit,
            "Margin %": p.gross_margin_pct,
            "Rev Share %": p.revenue_share_pct,
            "Pareto Class": p.abc_class,
            "Quadrant": p.quadrant
        })
    df_leaderboard = pd.DataFrame(sku_rows)
    st.dataframe(
        df_leaderboard.style.format({
            "Units Sold": "{:,}",
            "Revenue": "${:,.2f}",
            "Gross Profit": "${:,.2f}",
            "Margin %": "{:.1f}%",
            "Rev Share %": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


# ====================================================
# TAB 3: DEMAND FORECAST
# ====================================================
with tab_forecast:
    st.markdown("### 🔮 **Retail Demand Forecasting Engine**")
    st.caption("Time-aware validation, baseline comparison, and parametric uncertainty intervals.")

    fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
    with fc_col1:
        fc_target = st.selectbox("Forecast Target", ["quantity", "revenue"], format_func=lambda x: "Units Sold (Quantity)" if x == "quantity" else "Net Revenue ($)")
    with fc_col2:
        fc_scope_type = st.radio("Aggregation Scope", ["Storewide", "Category", "Product SKU"], horizontal=True)
    with fc_col3:
        if fc_scope_type == "Category":
            fc_scope_val = st.selectbox("Select Category", sorted(list(df_filtered["category"].unique())))
            fc_cat_arg = fc_scope_val
            fc_prod_arg = None
        elif fc_scope_type == "Product SKU":
            prod_dict = {f"{p.product_id} — {p.product_name}": p.product_id for p in profit_items}
            selected_sku_label = st.selectbox("Select Product SKU", list(prod_dict.keys()))
            fc_prod_arg = prod_dict[selected_sku_label]
            fc_cat_arg = None
        else:
            fc_prod_arg = None
            fc_cat_arg = None
    with fc_col4:
        horizon_days = st.slider("Forecast Horizon (Days)", min_value=14, max_value=90, value=30, step=7)

    # Model evaluation and forecast run
    with st.spinner("Executing time-series training and holdout validation..."):
        fc_res = generate_forecast(
            df_filtered,
            target=fc_target,
            horizon_days=horizon_days,
            product_id=fc_prod_arg,
            category=fc_cat_arg
        )

    # Forecast chart
    st.plotly_chart(plot_forecast(fc_res), use_container_width=True)

    # Model Evaluation Metrics Comparison Table
    st.markdown("#### **Time-Series Model Performance on Holdout Test Set**")
    st.caption(f"Evaluated on {fc_res.test_points_count} consecutive holdout days (time-ordered split, zero data leakage).")

    metrics_data = []
    for m in fc_res.model_metrics:
        metrics_data.append({
            "Model Name": m.model_name + (" ⭐ (Selected)" if m.is_best else ""),
            "Mean Absolute Error (MAE)": f"{m.mae:,.2f}",
            "Root Mean Squared Error (RMSE)": f"{m.rmse:,.2f}",
            "WAPE % (Weighted Error)": f"{m.wape_pct:.1f}%",
            "Selection Status": "Optimal Accuracy" if m.is_best else "Alternative / Baseline"
        })
    st.table(pd.DataFrame(metrics_data))

    # Limitations & Business Methodology
    with st.expander("ℹ️ **Model Architecture & Limitation Disclaimers**", expanded=False):
        st.markdown("""
        **Methodological Principles**:
        1. **Strict Time Ordering**: Time series data is never randomly partitioned. The model trains strictly on past history and predicts the consecutive holdout test period.
        2. **Baseline Standard**: Every advanced algorithm (e.g. Ridge Regression with autoregressive lags) is tested against a simple 7-Day Moving Average baseline.
        3. **Parametric Uncertainty**: 80% and 95% confidence intervals expand proportionally over the forecast horizon based on residual standard deviation.
        4. **Non-Negativity Constraint**: Retail sales cannot be negative; forecasts and confidence floors are strictly clamped at zero.
        """)
        for lim in fc_res.limitations:
            st.markdown(f"- ⚠️ {lim}")


# ====================================================
# TAB 4: INVENTORY INTELLIGENCE
# ====================================================
with tab_inventory:
    st.markdown("### 📦 **Inventory Intelligence & Replenishment Advisory**")
    st.caption("Calculate Days of Supply, Safety Stock buffers, Reorder Points, and capital tie-up.")

    # Inventory Control Parameters
    inv_c1, inv_c2, inv_c3 = st.columns(3)
    with inv_c1:
        service_level_choice = st.selectbox(
            "Target In-Stock Service Level",
            options=[0.90, 0.95, 0.975, 0.99],
            index=1,
            format_func=lambda x: f"{x*100:.1f}% In-Stock Target (Z={ {0.90:1.28, 0.95:1.65, 0.975:1.96, 0.99:2.33}[x] })"
        )
        z_val = {0.90: 1.28, 0.95: 1.65, 0.975: 1.96, 0.99: 2.33}[service_level_choice]
    with inv_c2:
        review_days = st.slider("Supplier Review Period (Days)", min_value=7, max_value=28, value=14, step=7)
    with inv_c3:
        trailing_window = st.slider("Demand Velocity History (Days)", min_value=30, max_value=90, value=45, step=15)

    inv_rep_custom = analyze_inventory(
        df_filtered,
        trailing_days=trailing_window,
        service_level_z=z_val,
        review_period_days=review_days
    )

    # Inventory Metrics Summary Cards
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Critical Stockouts</div>
            <div class="metric-value delta-neg">{inv_rep_custom.critical_count} SKUs</div>
            <div class="metric-sub">Will stock out before lead time</div>
        </div>
        """, unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reorder Warnings</div>
            <div class="metric-value" style="color: #F59E0B;">{inv_rep_custom.warning_count} SKUs</div>
            <div class="metric-sub">Below Reorder Point (ROP)</div>
        </div>
        """, unsafe_allow_html=True)
    with ic3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Slow-Moving Dead Stock</div>
            <div class="metric-value">{inv_rep_custom.slow_moving_count} SKUs</div>
            <div class="metric-sub">${inv_rep_custom.dead_stock_capital:,.2f} idle capital</div>
        </div>
        """, unsafe_allow_html=True)
    with ic4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Capital in Stock</div>
            <div class="metric-value">${inv_rep_custom.total_capital_tied_up:,.2f}</div>
            <div class="metric-sub">At direct unit COGS</div>
        </div>
        """, unsafe_allow_html=True)

    # Inventory Runway Visualisation
    st.plotly_chart(plot_inventory_runway(inv_rep_custom.items), use_container_width=True)

    # Complete Inventory Replenishment Table
    st.markdown("#### **Complete Product Replenishment Schedule**")
    inv_df_display = inventory_to_dataframe(inv_rep_custom)
    st.dataframe(
        inv_df_display.style.map(
            lambda v: "background-color: #FEE2E2; color: #991B1B; font-weight: bold;" if v == "CRITICAL"
            else ("background-color: #FEF3C7; color: #92400E; font-weight: bold;" if v == "WARNING"
            else ("background-color: #E2E8F0; color: #334155;" if v == "OVERSTOCKED"
            else "background-color: #D1FAE5; color: #065F46;")),
            subset=["Status"]
        ).format({
            "Avg Daily Sales": "{:.2f}",
            "Capital ($)": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ====================================================
# TAB 5: PROFITABILITY & WHAT-IF SIMULATOR
# ====================================================
with tab_profit:
    st.markdown("### 💰 **Profitability Analysis & What-If Pricing Simulator**")
    st.caption("Diagnose unit economics, margin dilution, and model hypothetical price/cost scenarios.")

    st.plotly_chart(plot_margin_quadrant(profit_items), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎛️ **Interactive What-If Scenario Simulator**")
    st.caption("Simulate the net financial impact of price changes, supplier inflation, and discount promotions.")

    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
    with sim_col1:
        sim_scope_type = st.radio("Simulation Scope", ["All Products", "By Category", "By SKU"], key="sim_scope")
    with sim_col2:
        if sim_scope_type == "By Category":
            sim_cat = st.selectbox("Select Target Category", sorted(list(df_filtered["category"].unique())), key="sim_c_box")
            sim_sku = None
        elif sim_scope_type == "By SKU":
            prod_names = {f"{p.product_id} - {p.product_name}": p.product_id for p in profit_items}
            selected_sku_key = st.selectbox("Select Target SKU", list(prod_names.keys()), key="sim_s_box")
            sim_sku = prod_names[selected_sku_key]
            sim_cat = None
        else:
            sim_sku = None
            sim_cat = None
    with sim_col3:
        price_adj = st.slider("Price Adjustment (%)", min_value=-20.0, max_value=30.0, value=5.0, step=1.0)
        cost_adj = st.slider("Supplier Cost Change (%)", min_value=-15.0, max_value=25.0, value=0.0, step=1.0)
    with sim_col4:
        discount_adj = st.slider("Discount Promo Change (%)", min_value=-50.0, max_value=100.0, value=0.0, step=10.0)
        elasticity_val = st.slider("Price Elasticity of Demand (E)", min_value=-2.5, max_value=0.0, value=-0.8, step=0.1, help="Typical retail elasticity ranges from -0.5 (inelastic essentials) to -1.5 (elastic discretionary).")

    # Run Simulation
    sim_result = run_what_if_simulation(
        df_filtered,
        product_id=sim_sku,
        category=sim_cat,
        price_change_pct=price_adj,
        cost_change_pct=cost_adj,
        discount_adjustment_pct=discount_adj,
        price_elasticity=elasticity_val
    )

    # Simulator KPI Delta Row
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        rev_d_color = "delta-pos" if sim_result.revenue_delta >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Simulated Revenue</div>
            <div class="metric-value">${sim_result.simulated_revenue:,.0f}</div>
            <div class="metric-sub {rev_d_color}">{sim_result.revenue_delta:+,.0f} ({sim_result.revenue_delta_pct:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        prof_d_color = "delta-pos" if sim_result.profit_delta >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Simulated Gross Profit</div>
            <div class="metric-value">${sim_result.simulated_profit:,.0f}</div>
            <div class="metric-sub {prof_d_color}">{sim_result.profit_delta:+,.0f} ({sim_result.profit_delta_pct:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        margin_d_color = "delta-pos" if sim_result.margin_pts_delta >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Simulated Gross Margin</div>
            <div class="metric-value">{sim_result.simulated_margin_pct:.1f}%</div>
            <div class="metric-sub {margin_d_color}">{sim_result.margin_pts_delta:+.2f} percentage pts</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        unit_diff = sim_result.simulated_units - sim_result.baseline_units
        u_d_color = "delta-pos" if unit_diff >= 0 else "delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Projected Units Sold</div>
            <div class="metric-value">{sim_result.simulated_units:,}</div>
            <div class="metric-sub {u_d_color}">{unit_diff:+,} units ({ (unit_diff/sim_result.baseline_units*100) if sim_result.baseline_units>0 else 0:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    # Simulator Comparison Visualisation
    st.plotly_chart(plot_what_if_comparison(sim_result), use_container_width=True)
    st.info(f"💡 **Simulation Context**: {sim_result.assumptions_summary}")


# ====================================================
# TAB 6: ASK SHOPSENSE (AI BUSINESS INSIGHTS)
# ====================================================
with tab_ai:
    st.markdown("### 🤖 **Ask ShopSense — Intelligent Decision Assistant**")
    st.caption("Ask natural language business questions grounded in verified dataset metrics.")

    # Quick prompt buttons
    st.markdown("**Sample Business Questions:**")
    qp_cols = st.columns(5)
    sample_queries = [
        "Which products need urgent attention?",
        "What are our most profitable products?",
        "Which products are at critical stockout risk?",
        "Identify slow-moving dead stock",
        "What is driving our recent sales performance?"
    ]

    selected_quick_prompt = None
    for i, col in enumerate(qp_cols):
        with col:
            if st.button(sample_queries[i], key=f"qp_{i}", use_container_width=True):
                selected_quick_prompt = sample_queries[i]

    # Query input
    user_input = st.text_input(
        "Enter your retail business question:",
        value=selected_quick_prompt if selected_quick_prompt else "",
        placeholder="e.g. Which products are at risk of running out of stock?"
    )

    if user_input:
        with st.spinner("Analyzing store metrics and synthesizing grounded insights..."):
            ai_resp = answer_business_query(
                query=user_input,
                df=df_filtered,
                kpis=kpis,
                inv_report=inv_report,
                profit_items=profit_items,
                recommendations=recommendations,
                gemini_api_key=gemini_key_input
            )

        # Render response
        st.markdown("---")
        st.markdown(ai_resp.answer_markdown)

        if ai_resp.grounded_metrics:
            st.markdown("##### 📌 **Grounded Quantitative Evidence**")
            gm_cols = st.columns(len(ai_resp.grounded_metrics))
            for idx, (m_lbl, m_val) in enumerate(ai_resp.grounded_metrics.items()):
                with gm_cols[idx]:
                    st.metric(m_lbl, m_val)

        st.caption(f"🛡️ *Inference Source: {ai_resp.source} • Recommended View: {ai_resp.relevant_view}*")


# ====================================================
# TAB 7: DATA QUALITY & METHODOLOGY
# ====================================================
with tab_methodology:
    st.markdown("### 📋 **Data Quality Audit & Analytical Methodology**")
    st.caption("Transparent documentation of health checks, statistical formulas, and retail business logic.")

    dq_c1, dq_c2 = st.columns([4, 6])

    with dq_c1:
        st.markdown("#### **Data Health Score**")
        health_color = "#10B981" if report_active.health_score >= 90 else ("#F59E0B" if report_active.health_score >= 70 else "#EF4444")
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 24px; text-align: center;">
            <div style="font-size: 3rem; font-weight: 800; color: {health_color};">{report_active.health_score:.1f}%</div>
            <div style="font-size: 1rem; font-weight: 600; color: #0F172A; margin-top: 4px;">Health & Reliability Score</div>
            <div style="font-size: 0.82rem; color: #64748B; margin-top: 8px;">
                Evaluated against schema integrity, missing values, date consistency, and value boundaries.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Download Data Quality Report
        st.download_button(
            label="📥 Download Data Quality Report (Markdown)",
            data=report_active.to_markdown(),
            file_name="shopsense_data_quality_report.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Download Sample Template
        template_path = os.path.join(os.path.dirname(__file__), "data", "sample_upload_template.csv")
        if os.path.exists(template_path):
            with open(template_path, "rb") as tf:
                st.download_button(
                    label="📥 Download POS Upload Template (CSV)",
                    data=tf.read(),
                    file_name="shopsense_retail_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with dq_c2:
        st.markdown("#### **Audit Metrics Breakdown**")
        audit_summary = pd.DataFrame([
            {"Audit Dimension": "Total Transactions Analyzed", "Observed Value": f"{report_active.total_rows:,}"},
            {"Audit Dimension": "Date Span Covered", "Observed Value": f"{report_active.date_range_start} to {report_active.date_range_end} ({report_active.total_days} days)"},
            {"Audit Dimension": "Active Catalog Breadth", "Observed Value": f"{report_active.unique_skus} SKUs across {report_active.unique_categories} categories"},
            {"Audit Dimension": "Duplicate Records Detected", "Observed Value": f"{report_active.duplicate_rows_count}"},
            {"Audit Dimension": "Missing / Null Cell Rate", "Observed Value": f"{report_active.missing_cells_count} ({report_active.missing_cells_pct:.2f}%)"},
            {"Audit Dimension": "Negative Value Anomalies", "Observed Value": f"{report_active.negative_values_count}"},
            {"Audit Dimension": "Zero Selling Price / Cost Flags", "Observed Value": f"{report_active.zero_price_or_cost_count}"},
        ])
        st.table(audit_summary)

    st.markdown("---")
    st.markdown("#### 📐 **Comprehensive Mathematical & Retail Methodology**")

    m_exp1, m_exp2, m_exp3 = st.columns(3)
    with m_exp1:
        st.markdown("""
        **1. Inventory Safety Stock & ROP**
        $$\\text{Safety Stock (SS)} = Z \\times \\sigma_d \\times \\sqrt{L}$$
        $$\\text{Reorder Point (ROP)} = (\\text{ADS} \\times L) + SS$$
        - $Z$: Standard normal score for target service level (1.65 for 95% cycle service level).
        - $\\sigma_d$: Standard deviation of daily unit sales over trailing 45 days.
        - $L$: Replenishment lead time in days.
        - $\\text{ADS}$: Average Daily Sales.
        """)
    with m_exp2:
        st.markdown("""
        **2. Demand Forecast & WAPE Metric**
        $$\\text{WAPE} = \\frac{\\sum_{t=1}^n |y_t - \\hat{y}_t|}{\\sum_{t=1}^n y_t} \\times 100\\%$$
        - Weighted Absolute Percentage Error (WAPE) is preferred over MAPE in retail because it handles zero-sales days without infinite division.
        - Time series models (Ridge with lag features, Holt's Exponential Smoothing) evaluated on strictly chronological holdout split.
        """)
    with m_exp3:
        st.markdown("""
        **3. Profitability & Price Elasticity**
        $$\\text{Gross Margin \\%} = \\frac{\\text{Revenue} - \\text{COGS}}{\\text{Revenue}} \\times 100\\%$$
        $$\\%\\Delta Q = E \\times \\%\\Delta P$$
        - Unit Contribution: $\\text{Unit Price} - \\text{Unit Cost}$.
        - $E$: Price elasticity of demand (retail benchmark $E = -0.80$).
        - Models sales volume sensitivity to pricing adjustments.
        """)
