#!/usr/bin/env python3
"""
ShopSense - Synthetic Retail Dataset Generator
Generates a realistic, multi-category retail transaction dataset representing
'UrbanPulse Lifestyle & Goods', an independent specialty boutique.
Includes realistic seasonal trends, day-of-week variation, inventory levels,
and product unit economics.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_dataset(output_path: str, template_path: str, seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

    products = [
        # Specialty Coffee & Pantry
        {"sku": "COF-001", "name": "Organic French Roast Coffee (340g)", "category": "Specialty Coffee & Pantry",
         "unit_price": 16.50, "unit_cost": 8.25, "lead_time": 10, "current_stock": 22, "supplier": "Pacific Roasters Co.", "base_vol": 6.8},
        {"sku": "COF-002", "name": "Single Origin Ethiopian Guji (340g)", "category": "Specialty Coffee & Pantry",
         "unit_price": 19.00, "unit_cost": 10.50, "lead_time": 14, "current_stock": 48, "supplier": "Pacific Roasters Co.", "base_vol": 4.5},
        {"sku": "COF-003", "name": "Cold Brew Glass Infuser Bottle", "category": "Specialty Coffee & Pantry",
         "unit_price": 28.00, "unit_cost": 11.20, "lead_time": 21, "current_stock": 35, "supplier": "Artisan Ware Inc.", "base_vol": 2.2},
        {"sku": "COF-004", "name": "Raw Wildflower Honey (500g)", "category": "Specialty Coffee & Pantry",
         "unit_price": 14.00, "unit_cost": 6.00, "lead_time": 12, "current_stock": 55, "supplier": "Valley Apiaries", "base_vol": 3.4},
        {"sku": "COF-005", "name": "Ceremonial Uji Matcha Tin (50g)", "category": "Specialty Coffee & Pantry",
         "unit_price": 32.00, "unit_cost": 15.00, "lead_time": 28, "current_stock": 26, "supplier": "Kyoto Direct Imports", "base_vol": 2.1},

        # Home & Living
        {"sku": "HOM-001", "name": "Hand-Thrown Ceramic Mug (12oz)", "category": "Home & Living",
         "unit_price": 24.00, "unit_cost": 8.50, "lead_time": 15, "current_stock": 18, "supplier": "Stone & Kiln Studio", "base_vol": 5.4},
        {"sku": "HOM-002", "name": "Lavender & Cedar Soy Candle", "category": "Home & Living",
         "unit_price": 22.00, "unit_cost": 6.50, "lead_time": 10, "current_stock": 72, "supplier": "North Light Candle Co.", "base_vol": 5.0},
        {"sku": "HOM-003", "name": "Pure Washed Linen Throw Pillow", "category": "Home & Living",
         "unit_price": 45.00, "unit_cost": 18.00, "lead_time": 18, "current_stock": 30, "supplier": "Loom & Flax Textiles", "base_vol": 2.6},
        {"sku": "HOM-004", "name": "Fluted Stoneware Indoor Planter", "category": "Home & Living",
         "unit_price": 38.00, "unit_cost": 21.00, "lead_time": 25, "current_stock": 64, "supplier": "Stone & Kiln Studio", "base_vol": 0.8}, # Slow moving
        {"sku": "HOM-005", "name": "Reclaimed Acacia Wood Coaster Set", "category": "Home & Living",
         "unit_price": 18.00, "unit_cost": 5.40, "lead_time": 14, "current_stock": 80, "supplier": "Timber Craft Goods", "base_vol": 3.8},

        # Stationery & Desk
        {"sku": "DES-001", "name": "Minimalist Bamboo Desk Organizer", "category": "Stationery & Desk",
         "unit_price": 34.00, "unit_cost": 13.50, "lead_time": 16, "current_stock": 42, "supplier": "Timber Craft Goods", "base_vol": 3.1},
        {"sku": "DES-002", "name": "Full-Grain Leather Notebook (A5)", "category": "Stationery & Desk",
         "unit_price": 29.00, "unit_cost": 11.00, "lead_time": 14, "current_stock": 58, "supplier": "Heritage Leather Co.", "base_vol": 4.2},
        {"sku": "DES-003", "name": "Solid Brass Rollerball Pen", "category": "Stationery & Desk",
         "unit_price": 42.00, "unit_cost": 14.50, "lead_time": 20, "current_stock": 36, "supplier": "Precision Tools LLC", "base_vol": 1.9},
        {"sku": "DES-004", "name": "Wool Felt Desk Pad (Charcoal)", "category": "Stationery & Desk",
         "unit_price": 36.00, "unit_cost": 12.00, "lead_time": 14, "current_stock": 45, "supplier": "Loom & Flax Textiles", "base_vol": 2.7},
        {"sku": "DES-005", "name": "Ergonomic Walnut Wrist Rest", "category": "Stationery & Desk",
         "unit_price": 26.00, "unit_cost": 13.00, "lead_time": 15, "current_stock": 50, "supplier": "Timber Craft Goods", "base_vol": 2.0},

        # Apparel & Everyday
        {"sku": "APP-001", "name": "Merino Wool Ribbed Beanie", "category": "Apparel & Everyday",
         "unit_price": 32.00, "unit_cost": 12.50, "lead_time": 21, "current_stock": 40, "supplier": "Highland Knits", "base_vol": 3.5},
        {"sku": "APP-002", "name": "Heavy Canvas Market Tote", "category": "Apparel & Everyday",
         "unit_price": 24.00, "unit_cost": 7.20, "lead_time": 10, "current_stock": 90, "supplier": "Canvas & Thread Co.", "base_vol": 5.6},
        {"sku": "APP-003", "name": "Crossback Japanese Linen Apron", "category": "Apparel & Everyday",
         "unit_price": 52.00, "unit_cost": 22.00, "lead_time": 18, "current_stock": 28, "supplier": "Loom & Flax Textiles", "base_vol": 1.8},
        {"sku": "APP-004", "name": "Organic Pima Cotton Crewneck Tee", "category": "Apparel & Everyday",
         "unit_price": 38.00, "unit_cost": 14.00, "lead_time": 14, "current_stock": 65, "supplier": "Canvas & Thread Co.", "base_vol": 3.8},
        {"sku": "APP-005", "name": "Scottish Cashmere Fringed Scarf", "category": "Apparel & Everyday",
         "unit_price": 95.00, "unit_cost": 62.00, "lead_time": 30, "current_stock": 25, "supplier": "Highland Knits", "base_vol": 0.7}, # High price, lower margin %

        # Wellness & Personal Care
        {"sku": "WEL-001", "name": "Botanical Rosewater Hydrating Mist", "category": "Wellness & Personal Care",
         "unit_price": 21.00, "unit_cost": 5.50, "lead_time": 12, "current_stock": 44, "supplier": "Botanica Labs", "base_vol": 3.9},
        {"sku": "WEL-002", "name": "Cold-Pressed Castile Soap Bar 3-Pack", "category": "Wellness & Personal Care",
         "unit_price": 15.00, "unit_cost": 4.20, "lead_time": 8, "current_stock": 110, "supplier": "Botanica Labs", "base_vol": 6.2},
        {"sku": "WEL-003", "name": "Organic Eucalyptus Body Wash (400ml)", "category": "Wellness & Personal Care",
         "unit_price": 24.00, "unit_cost": 7.50, "lead_time": 10, "current_stock": 52, "supplier": "Botanica Labs", "base_vol": 4.1},
        {"sku": "WEL-004", "name": "Intensive Herbal Hand Salve (60g)", "category": "Wellness & Personal Care",
         "unit_price": 16.00, "unit_cost": 4.80, "lead_time": 10, "current_stock": 68, "supplier": "Botanica Labs", "base_vol": 3.6},
    ]

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 6, 30)
    delta_days = (end_date - start_date).days + 1

    records = []
    order_counter = 10001
    channels = ["In-Store", "In-Store", "Online", "Online", "Click & Collect"]
    customer_types = ["Regular", "Regular", "New", "Loyalty VIP"]

    for d in range(delta_days):
        current_date = start_date + timedelta(days=d)
        day_of_week = current_date.weekday() # 0 = Monday, 6 = Sunday
        month = current_date.month

        # Day of week multiplier (retail peak on Friday, Saturday, Sunday)
        dow_mult = 1.0
        if day_of_week in [4, 5, 6]: # Fri, Sat, Sun
            dow_mult = 1.45
        elif day_of_week in [1, 2]: # Tue, Wed
            dow_mult = 0.85

        # Seasonal multiplier
        season_mult = 1.0
        # Holiday season surge (Nov 15 - Dec 24)
        if (month == 11 and current_date.day >= 15) or (month == 12 and current_date.day <= 24):
            season_mult = 1.75
            if month == 11 and 24 <= current_date.day <= 30: # Black Friday week
                season_mult = 2.4
        elif month == 1: # Post-holiday January dip
            season_mult = 0.75
        elif month in [4, 5]: # Spring gift / Mother's day lift
            season_mult = 1.15

        # Organic business growth trend across 18 months (~15% annualized)
        growth_trend = 1.0 + (d / delta_days) * 0.18

        # Generate orders for the day
        for prod in products:
            expected_sales = prod["base_vol"] * dow_mult * season_mult * growth_trend
            # Poisson or negative binomial daily transaction frequency
            daily_orders = np.random.poisson(lam=max(0.2, expected_sales * 0.6))

            for _ in range(daily_orders):
                order_id = f"ORD-{order_counter}"
                order_counter += 1

                # Quantity per order: mostly 1, occasionally 2-4
                qty = np.random.choice([1, 2, 3, 4], p=[0.72, 0.20, 0.06, 0.02])

                # Occasional promo / discount (10-15% chance of 5-15% discount)
                has_discount = random.random() < 0.14
                discount_rate = random.choice([0.05, 0.10, 0.15]) if has_discount else 0.0
                unit_price = prod["unit_price"]
                line_total = unit_price * qty
                discount_amount = round(line_total * discount_rate, 2)
                net_revenue = round(line_total - discount_amount, 2)
                unit_cost = prod["unit_cost"]
                total_cogs = round(unit_cost * qty, 2)
                gross_profit = round(net_revenue - total_cogs, 2)

                channel = random.choice(channels)
                cust_type = random.choice(customer_types)

                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "order_id": order_id,
                    "product_id": prod["sku"],
                    "product_name": prod["name"],
                    "category": prod["category"],
                    "unit_price": unit_price,
                    "unit_cost": unit_cost,
                    "quantity": qty,
                    "discount_amount": discount_amount,
                    "revenue": net_revenue,
                    "cost_of_goods": total_cogs,
                    "gross_profit": gross_profit,
                    "sales_channel": channel,
                    "customer_type": cust_type,
                    "current_stock": prod["current_stock"],
                    "lead_time_days": prod["lead_time"],
                    "supplier_name": prod["supplier"]
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions in {output_path}")

    # Generate sample template for small business owners
    template_cols = [
        "date", "order_id", "product_id", "product_name", "category",
        "unit_price", "unit_cost", "quantity", "discount_amount",
        "sales_channel", "current_stock", "lead_time_days", "supplier_name"
    ]
    template_df = df[template_cols].head(10)
    template_df.to_csv(template_path, index=False)
    print(f"Generated template in {template_path}")

if __name__ == "__main__":
    generate_dataset(
        output_path="/Users/gitanshgarg/Documents/shopsense/data/demo_retail_sales.csv",
        template_path="/Users/gitanshgarg/Documents/shopsense/data/sample_upload_template.csv"
    )
