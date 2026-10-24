# ShopSense Analytical & Retail Methodology Guide

## 1. Executive Summary & Philosophy
ShopSense is engineered to solve a pervasive vulnerability in small-to-midsize retail management: **data opacity**. Small retailers frequently track total sales volume on their point-of-sale (POS) terminal, yet remain blind to stockout horizons, margin compression, slow-moving capital traps, and forward demand variance until cash flow or customer fulfillment is actively compromised.

This document formalizes the mathematical definitions, statistical heuristics, machine learning architectures, and decision rules powering ShopSense. It is structured to serve as both an engineering specification and an interview-ready playbook for placement discussions.

---

## 2. Foundational Retail Financial & Operational KPIs

### 2.1 Net Sales Revenue
Net sales represents genuine incoming retail turnover net of item-level and promotional discounts:
$$\text{Revenue} = \sum_{i=1}^N \left( \text{unit\_price}_i \times \text{quantity}_i - \text{discount\_amount}_i \right)$$
*Distinction*: Gross sales ($\text{unit\_price} \times \text{quantity}$) is never conflated with realized net revenue.

### 2.2 Cost of Goods Sold (COGS)
Direct wholesale acquisition cost of products sold to customers:
$$\text{COGS} = \sum_{i=1}^N \left( \text{unit\_cost}_i \times \text{quantity}_i \right)$$
*Boundary rule*: Excludes operating overhead (rent, marketing, wages) to isolate pure merchandise unit contribution.

### 2.3 Gross Profit & Gross Margin Percentage
Gross profit evaluates direct merchandise cash generation:
$$\text{Gross Profit} = \text{Revenue} - \text{COGS}$$
$$\text{Gross Margin \%} = \left( \frac{\text{Gross Profit}}{\text{Revenue}} \right) \times 100\%$$
*Business significance*: In boutique specialty retail, benchmark gross margins typically span **50% to 65%**. A drop below 45% flags severe margin dilution or excessive discounting.

### 2.4 Average Order Value (AOV)
Measures basket size efficiency across distinct transactions:
$$\text{AOV} = \frac{\text{Total Net Revenue}}{\text{Total Distinct Orders (Invoices)}}$$

### 2.5 Period-over-Period Growth Deltas
Evaluates trajectory by splitting filtered horizons into continuous halves of length $T / 2$:
$$\Delta \% \text{ Metric} = \left( \frac{\text{Metric}_{\text{current}} - \text{Metric}_{\text{prior}}}{\text{Metric}_{\text{prior}}} \right) \times 100\%$$
$$\Delta \text{ Margin Points} = \text{Gross Margin \%}_{\text{current}} - \text{Gross Margin \%}_{\text{prior}}$$

---

## 3. Inventory Intelligence & Stochastic Replenishment

Small retailers struggle with two opposing costs:
1. **Stockout Cost**: Lost sales revenue, broken customer habits, and search diversion.
2. **Holding Cost**: Capital tied up in stagnant inventory, obsolescence, damage, and storage rent.

ShopSense balances these forces through classical stochastic inventory control:

### 3.1 Average Daily Sales (ADS) & Demand Volatility ($\sigma_d$)
Evaluated over a trailing evaluation window $W$ (default 45 days) mapped onto a continuous calendar date range:
$$\text{ADS} = \frac{1}{W} \sum_{t=1}^W q_t$$
$$\sigma_d = \sqrt{\frac{1}{W - 1} \sum_{t=1}^W (q_t - \text{ADS})^2}$$
*Continuous index rule*: Days with zero sales must be explicitly filled with $0.0$ rather than omitted; otherwise, sales velocity is artificially inflated.

### 3.2 Days of Inventory Remaining (DOI)
$$\text{DOI} = \frac{\text{Current Stock on Hand}}{\text{ADS}}$$
Estimates runway days until complete depletion assuming steady sales velocity.

### 3.3 Safety Stock ($SS$)
Buffers against random demand surges during supplier replenishment lead time $L$:
$$SS = \lceil Z \times \sigma_d \times \sqrt{L} \rceil$$
- $Z$: Standard normal inverse cumulative density quantile for the target Cycle Service Level (CSL):
  - $90\%$ In-Stock Target $\rightarrow Z = 1.28$
  - $95\%$ In-Stock Target $\rightarrow Z = 1.65$ (ShopSense Default)
  - $97.5\%$ In-Stock Target $\rightarrow Z = 1.96$
  - $99\%$ In-Stock Target $\rightarrow Z = 2.33$
- $\sigma_d$: Daily demand standard deviation.
- $L$: Supplier replenishment lead time in business days.

### 3.4 Reorder Point (ROP)
Trigger threshold for generating purchase orders:
$$ROP = \lceil (\text{ADS} \times L) + SS \rceil$$

### 3.5 Stockout Urgency Classification

| Status | Mathematical Condition | Operational Meaning | Action Urgency |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | $\text{Current Stock} \le \lceil \text{ADS} \times L \rceil$ | Stock will deplete **before** normal supplier delivery arrives. | 🔴 Emergency expedited freight / air shipment |
| **WARNING** | $\lceil \text{ADS} \times L \rceil < \text{Current Stock} \le ROP$ | Safety buffer breached; within reorder window. | 🟡 Standard purchase order to supplier |
| **ADEQUATE** | $ROP < \text{Current Stock} \le ROP \times 2.5$ | Healthy balance between availability and holding cost. | 🟢 Monitor routine sales |
| **OVERSTOCKED** | $\text{DOI} > 90\text{ days}$ OR ($\text{ADS} < 0.30$ and $\text{Stock} > 20$) | Working capital trap; high risk of obsolescence. | ⚪ Promotional markdown / bundle liquidation |

### 3.6 Periodic Review Order-Up-To Level ($S$)
For a retailer reviewing orders every $R$ days (e.g., bi-weekly $R=14$):
$$S = \text{ADS} \times (L + R) + SS$$
$$\text{Suggested Order Quantity} = \max\left(0, \lceil S - \text{Current Stock} \rceil\right)$$

---

## 4. Demand Forecasting & Time-Aware Model Selection

### 4.1 Strict Temporal Holdout Protocol
In financial and demand time series, random train/test shuffling represents a severe methodological flaw (lookahead bias / data leakage). ShopSense enforces strict chronological partitioning:
- **Training Set**: Days $1$ to $T - H_{\text{test}}$.
- **Testing Holdout**: Days $(T - H_{\text{test}} + 1)$ to $T$ (last 14 to 30 days).

```
[===================== Training History (Past) =====================][=== Holdout Test ===]
                                                                     ^ Split Point (No Shuffling)
```

### 4.2 Candidate Algorithms

#### Model 1: 7-Day Moving Average Baseline (Seasonal-Naive)
Computes a rolling 7-day trailing mean adjusted by day-of-week historical multipliers:
$$\hat{y}_{t} = \left(\frac{1}{7} \sum_{i=1}^7 y_{t-i}\right) \times \left(\frac{\bar{y}_{\text{DOW}(t)}}{\bar{y}_{\text{all}}}\right)$$

#### Model 2: Holt's Linear Exponential Smoothing
Incorporates smoothed level $l_t$ and damped trend $b_t$:
$$l_t = \alpha y_t + (1 - \alpha)(l_{t-1} + \phi b_{t-1})$$
$$b_t = \beta (l_t - l_{t-1}) + (1 - \beta)\phi b_{t-1}$$
$$\hat{y}_{t+h} = \max\left(0, l_t + \sum_{i=1}^h \phi^i b_t\right)$$
where $\alpha = 0.3$, $\beta = 0.1$, and damping factor $\phi = 0.95$.

#### Model 3: L2-Regularized Ridge Lag Regression (Machine Learning)
Extracts engineered calendar and autoregressive lag features:
- Calendar: $\text{day\_of\_week}$, $\text{day\_of\_year}$, $\text{month}$, $\text{is\_weekend}$.
- Autoregressive Lags: $\text{lag}_1$, $\text{lag}_7$, $\text{lag}_{14}$, $\text{rolling\_mean}_7$, $\text{rolling\_std}_7$.
- Objective: Minimize $\sum (y_i - \hat{y}_i)^2 + \lambda \|\mathbf{w}\|_2^2$ with $\lambda = 1.0$.

### 4.3 Evaluation Metrics
1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}$$
3. **Weighted Absolute Percentage Error (WAPE)**:
   $$\text{WAPE} = \frac{\sum_{t=1}^n |y_t - \hat{y}_t|}{\sum_{t=1}^n y_t} \times 100\%$$
   *Why WAPE over MAPE?* In retail SKU forecasting, many days exhibit zero sales ($y_t = 0$). Standard MAPE ($\frac{|y - \hat{y}|}{y}$) incurs division-by-zero errors. WAPE aggregates total volume first, providing a robust, interpretable percentage metric.

### 4.4 Parametric Uncertainty Intervals
For forecast horizon step $h \in [1, H]$, prediction intervals expand with horizon square root:
$$\hat{y}_{t+h} \pm Z_{\text{interval}} \times \hat{\sigma}_{\text{res}} \times (0.8 + 0.05 \sqrt{h})$$
- 80% Confidence Interval: $Z = 1.28$
- 95% Confidence Interval: $Z = 1.96$
- Clamped at zero: $\text{Lower Bound} = \max(0.0, \text{Lower Bound})$.

---

## 5. Strategic Profitability & What-If Pricing Elasticity

### 5.1 Product Portfolio Strategic Matrix
Products are mapped into a 4-quadrant coordinate space based on median revenue and store-average gross margin:

```
Gross Margin %
      ^
      |   [HIDDEN GEMS]              |   [STAR PERFORMERS]
 High |   High Margin, Low Revenue   |   High Margin, High Revenue
      |   Action: Promote & Bundle   |   Action: Protect Stock & Feature
      |------------------------------+-------------------------------> Revenue ($)
      |   [PRUNE CANDIDATES]         |   [VOLUME TRAPS]
 Low  |   Low Margin, Low Revenue    |   Low Margin, High Revenue
      |   Action: Delist or Replace  |   Action: Negotiate Cost / Raise Price
      |
```

### 5.2 Pareto (80/20) ABC Classification
- **Class A** (Top 80% Cumulative Revenue): Vital core contributors (~20% of SKUs).
- **Class B** (Next 15% Cumulative Revenue): Secondary contributors (~30% of SKUs).
- **Class C** (Bottom 5% Cumulative Revenue): Tail items (~50% of SKUs).

### 5.3 What-If Scenario Modeling via Price Elasticity
Price Elasticity of Demand ($E$) governs how quantity demanded responds to price changes:
$$E = \frac{\% \Delta Q}{\% \Delta P} \implies \% \Delta Q = E \times \% \Delta P$$
For example, with standard discretionary retail elasticity $E = -0.80$, a $+5\%$ price increase yields:
$$\% \Delta Q = -0.80 \times (+5.0\%) = -4.0\%$$
Simulated financial outcome:
$$\text{Revenue}_{\text{sim}} = \sum \left( P \times (1 + \Delta P) \times Q \times (1 + \Delta Q) - \text{Discount}_{\text{sim}} \right)$$
$$\text{COGS}_{\text{sim}} = \sum \left( C \times (1 + \Delta C) \times Q \times (1 + \Delta Q) \right)$$
$$\text{Profit}_{\text{sim}} = \text{Revenue}_{\text{sim}} - \text{COGS}_{\text{sim}}$$

---

## 6. Worked Business Decision Example (Placement Interview Case)

### The Problem:
An independent retailer is reviewing performance for SKU `HOM-001` (*Hand-Thrown Ceramic Mug*).
- List Price ($P$): \$24.00
- Unit COGS ($C$): \$8.50 (Gross margin: 64.6%)
- Current Physical Stock on Hand: 18 units
- Trailing Average Daily Sales ($\text{ADS}$): 5.4 units/day
- Demand Volatility ($\sigma_d$): 2.1 units
- Supplier Lead Time ($L$): 15 business days

### The Mathematical Diagnosis:
1. **Days of Inventory**: $\text{DOI} = \frac{18}{5.4} = 3.33 \text{ days}$.
2. **Lead Time Demand**: $\text{ADS} \times L = 5.4 \times 15 = 81.0 \text{ units}$.
3. **Safety Stock Buffer** ($Z=1.65$): $SS = \lceil 1.65 \times 2.1 \times \sqrt{15} \rceil = \lceil 1.65 \times 2.1 \times 3.87 \rceil = \lceil 13.4 \rceil = 14 \text{ units}$.
4. **Reorder Point**: $ROP = 81 + 14 = 95 \text{ units}$.
5. **Stockout Horizon**: Physical stock runs out in 3.3 days, but replacement stock requires 15 days.
   $$\text{Stockout Deficit Window} = 15 - 3.33 = 11.67 \text{ days}$$
   $$\text{Unfulfillable Demand} = 11.67 \times 5.4 \approx 63 \text{ units}$$
   $$\text{Estimated Unrealized Revenue} = 63 \times \$24.00 = \$1,512.00$$

### The Executive Recommendation:
- **Urgency**: `CRITICAL`
- **Immediate Action**: Issue an emergency expedited replenishment of at least $S - \text{Stock} = ((15 + 14) \times 5.4 + 14) - 18 = 153 \text{ units}$ to supplier *Stone & Kiln Studio*.
- **Mitigation**: Temporarily deactivate e-commerce availability to reserve remaining 18 units for high-value in-store footfall, preventing customer frustration and store reputation damage.
