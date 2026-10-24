# ShopSense Data Documentation & Data Dictionary

## 1. Dataset Overview
ShopSense includes an authentic, realistically modeled benchmark retail dataset simulating **UrbanPulse Lifestyle & Goods**, an independent multi-channel boutique retailer operating an in-store brick-and-mortar storefront alongside an e-commerce website and click-and-collect fulfillment.

- **File Path**: `data/demo_retail_sales.csv`
- **Time Horizon**: January 1, 2024 – June 30, 2025 (18 months, 547 contiguous days)
- **Record Count**: 36,850 transaction rows
- **Unique Products**: 24 active Stock Keeping Units (SKUs)
- **Product Categories**: 5 core departments:
  1. *Specialty Coffee & Pantry*
  2. *Home & Living*
  3. *Stationery & Desk*
  4. *Apparel & Everyday*
  5. *Wellness & Personal Care*

## 2. Licensing & Attribution
- **Synthetic Benchmark Dataset**: Generated specifically for ShopSense portfolio and research demonstrations.
- **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Permitted Uses**: Academic, educational, commercial evaluation, portfolio demonstration, and software testing.
- **Privacy Assurance**: No personally identifiable information (PII) or proprietary commercial data is contained in this dataset.

## 3. Data Dictionary

| Column Name | Data Type | Example Value | Description |
| :--- | :--- | :--- | :--- |
| `date` | String (ISO-8601 `YYYY-MM-DD`) | `2024-11-29` | Transaction purchase date |
| `order_id` | String | `ORD-10492` | Unique transaction order identifier |
| `product_id` | String | `HOM-001` | Unique Stock Keeping Unit (SKU) identifier |
| `product_name` | String | `Hand-Thrown Ceramic Mug (12oz)` | Descriptive merchandise name |
| `category` | String | `Home & Living` | Retail department / merchandise hierarchy |
| `unit_price` | Float ($ USD) | `24.00` | Regular retail list price per unit |
| `unit_cost` | Float ($ USD) | `8.50` | Direct unit Cost of Goods Sold (COGS) |
| `quantity` | Integer | `2` | Number of units purchased in the transaction |
| `discount_amount` | Float ($ USD) | `2.40` | Total discount deducted from gross sales |
| `revenue` | Float ($ USD) | `45.60` | Net sales revenue: `(unit_price * quantity) - discount_amount` |
| `cost_of_goods` | Float ($ USD) | `17.00` | Total COGS: `unit_cost * quantity` |
| `gross_profit` | Float ($ USD) | `28.60` | Gross profit contribution: `revenue - cost_of_goods` |
| `sales_channel` | String | `In-Store` | Point of purchase (`In-Store`, `Online`, `Click & Collect`) |
| `customer_type` | String | `Loyalty VIP` | Buyer segment (`Regular`, `New`, `Loyalty VIP`) |
| `current_stock` | Integer | `18` | Available physical inventory on hand at reporting date |
| `lead_time_days` | Integer | `15` | Supplier replenishment lead time in business days |
| `supplier_name` | String | `Stone & Kiln Studio` | Commercial vendor or distributor |

## 4. Realistic Retail Dynamics Incorporated
1. **Day-of-Week Seasonality**: Empirical weekend footfall multiplier (+45% lift on Fri–Sun vs. midweek low on Tue–Wed).
2. **Calendar Events**:
   - Holiday rush (November 15 – December 24) with peak Black Friday multiplier (2.4x baseline).
   - Post-holiday January trough (-25% demand drop).
   - Spring and Mother's Day lift in April–May (+15%).
3. **Compound Business Growth**: Organic ~15% annualized baseline revenue expansion across the 18-month duration.
4. **Authentic Unit Economics**:
   - Fast-moving staples (e.g., Organic French Roast, Castile Soap) with high volume and tight margins.
   - Luxury specialty goods (e.g., Scottish Cashmere Scarf, Brass Rollerball Pen) with lower turnover and higher unit capital commitment.
   - Genuine inventory friction: At-risk stockouts (`HOM-001`, `COF-001`) vs. slow-moving capital traps (`HOM-004`).

## 5. Sample Upload Template
For small retail business owners who wish to test their own point-of-sale (POS) exports, a standardized template is provided at `data/sample_upload_template.csv`. Required fields and flexible column mapping are described in the main documentation.
