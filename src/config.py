"""
ShopSense - Application Configuration and Business Constants
Defines theme styling, business rule defaults, and column mapping dictionaries.
"""

from typing import Dict, List

# Application Identity
APP_NAME = "ShopSense"
APP_TAGLINE = "AI-Powered Retail Business Decision Assistant"
APP_VERSION = "1.0.0"

# Visual Theme Tokens (Navy / Slate / Neutral Palette)
THEME_COLORS = {
    "primary": "#1E3A8A",       # Deep Royal Navy
    "primary_light": "#3B82F6", # Bright Blue
    "secondary": "#0F172A",     # Slate 900
    "accent": "#0D9488",        # Teal Accent
    "success": "#10B981",       # Emerald Green
    "warning": "#F59E0B",       # Amber Warning
    "danger": "#EF4444",        # Crimson Danger
    "neutral_bg": "#F8FAFC",    # Slate 50
    "card_bg": "#FFFFFF",       # Card White
    "border": "#E2E8F0",        # Slate 200
    "text_muted": "#64748B",    # Slate 500
}

# Retail Decision Rules & Thresholds
DEFAULT_SERVICE_LEVEL_Z = 1.65   # 95% cycle service level in-stock target
DEFAULT_REVIEW_PERIOD_DAYS = 14  # Review cycle for inventory replenishment
DEFAULT_SLOW_MOVING_DAYS = 60    # Window to evaluate slow-moving velocity
DEFAULT_SLOW_MOVING_VELOCITY = 0.30 # Units per day threshold
CRITICAL_DOI_DAYS = 7            # Days of Inventory < 7 triggers Critical Stockout
WARNING_DOI_DAYS = 14            # Days of Inventory < 14 triggers Warning Stockout
DEAD_STOCK_DOI_DAYS = 90         # Days of Inventory > 90 flagged as dead / overstocked

# Column Name Requirements & Flexible Mapping Aliases
REQUIRED_COLUMNS = [
    "date", "product_id", "product_name", "category",
    "unit_price", "unit_cost", "quantity", "revenue"
]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["date", "transaction_date", "order_date", "sale_date", "invoice_date"],
    "order_id": ["order_id", "invoice_no", "transaction_id", "order_num", "receipt_id"],
    "product_id": ["product_id", "sku", "item_code", "product_code", "item_id"],
    "product_name": ["product_name", "item_name", "product_description", "description", "title"],
    "category": ["category", "department", "product_category", "merchandise_group"],
    "unit_price": ["unit_price", "price", "selling_price", "retail_price"],
    "unit_cost": ["unit_cost", "cost", "cogs_per_unit", "wholesale_price", "buy_price"],
    "quantity": ["quantity", "qty", "units", "quantity_sold", "units_sold"],
    "discount_amount": ["discount_amount", "discount", "discount_value", "promo_discount"],
    "revenue": ["revenue", "sales", "net_sales", "line_total", "total_amount", "amount"],
    "sales_channel": ["sales_channel", "channel", "order_channel", "fulfillment_channel"],
    "current_stock": ["current_stock", "inventory", "stock_on_hand", "qty_on_hand", "available_stock"],
    "lead_time_days": ["lead_time_days", "lead_time", "supplier_lead_time", "reorder_lead_time"],
    "supplier_name": ["supplier_name", "supplier", "vendor", "distributor"],
}
