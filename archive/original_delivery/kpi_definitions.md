# CLEARS — KPI Definitions
**File:** `docs/kpi_definitions.md`
**Owner:** Person 1 — Data & Analytics Lead
**Purpose:** Defines all dashboard KPIs for Person 3 to implement correctly

---

## KPI 1 — Disruption Rate (%)

**Plain English:** The percentage of flights at a selected airport and time window that experienced a departure delay greater than 15 minutes.

**Formula:** `sum(disruption == 1) / count(disruption) × 100`

**Data Columns Used:** `disruption`, `airport_code`, `flight_date`

**Unit:** Percentage (%)

**Computed At:** Query time — filtered by airport selector and date range picker in dashboard

**Dashboard Display:** KPI card showing current disruption rate with trend arrow vs prior period

**Example:** If 180 out of 1,000 flights at ORD in January 2022 had departure delay > 15 min, Disruption Rate = 18.0%

---

## KPI 2 — Average Recovery Time (minutes)

**Plain English:** Among flights that were disrupted, the average number of minutes they were delayed. This tells operations teams how long disruptions typically last once they start.

**Formula:** `mean(delay_minutes) where disruption == 1`

**Data Columns Used:** `delay_minutes`, `disruption`, `airport_code`

**Unit:** Minutes

**Computed At:** Query time — per airport and selected time window

**Dashboard Display:** KPI card alongside disruption rate. Higher = disruptions are more severe when they occur.

**Example:** If disrupted flights at ORD averaged 68.2 minutes of delay, Average Recovery Time = 68.2 min

**Note:** `delay_minutes` is capped at 300 to prevent extreme outliers from distorting this metric.

---

## KPI 3 — Weather Sensitivity Score

**Plain English:** How strongly correlated weather danger events are with disruptions at a given airport. Higher score = more weather-vulnerable airport.

**Formula:** `Pearson correlation coefficient between storm_flag and disruption` computed over the full dataset period for each airport

**Data Columns Used:** `storm_flag`, `disruption`, `airport_code`

**Unit:** Correlation coefficient (-1 to +1)

**Computed At:** Pre-computed (stored in `airport_resilience_inputs.csv`)

**Dashboard Display:** Shown in resilience panel per airport. ORD = 0.0415, MEM = 0.0102

**Interpretation:** Positive = weather events increase disruption probability. Higher = more sensitive to weather.

---

## KPI 4 — Current Risk Level

**Plain English:** The ML model's predicted probability that the current weather and operational conditions will cause a disruption. Shown as Low, Medium, or High.

**Formula:**
- **Low:** model predicted probability < 0.30
- **Medium:** 0.30 ≤ probability ≤ 0.65
- **High:** probability > 0.65

**Data Columns Used:** All features in `features_final.csv` (model input)

**Unit:** Categorical (Low / Medium / High) + raw probability (0–1)

**Computed At:** Real-time inference — called via `predict.predict_disruption(feature_dict)`

**Dashboard Display:** Colour-coded gauge (green/amber/red) as the primary dashboard indicator

---

## KPI 5 — Resilience Index

**Plain English:** A single 0–100 score summarising how well an airport handles weather stress. Higher = more resilient. Computed from four dimensions: disruption frequency, recovery speed, weather sensitivity, and congestion sensitivity.

**Formula:**
1. Min-max normalise each dimension across airports to [0,1]
2. Vulnerability = 0.35 × disruption_rate + 0.30 × avg_recovery_time + 0.20 × weather_sensitivity + 0.15 × congestion_sensitivity
3. Resilience Index = 100 − (vulnerability × 100)

**Tier Classification:**
- 0–40: Vulnerable
- 41–70: Moderate
- 71–100: Resilient

**Data Columns Used:** `disruption_rate`, `avg_recovery_time`, `weather_sensitivity`, `congestion_sensitivity` from `airport_resilience_inputs.csv`

**Unit:** Score (0–100) + tier label

**Computed At:** Pre-computed (stored in `airport_resilience_inputs.csv`)

**Dashboard Values:** ORD = 30.0 (Vulnerable) | MEM = 70.0 (Moderate)

---

## KPI 6 — Peak Hour Disruption Premium

**Plain English:** How much more likely disruptions are during peak hours compared to off-peak hours. A value of 1.5 means disruptions are 50% more likely during peak hours.

**Formula:** `disruption_rate(is_peak_hour == 1) / disruption_rate(is_peak_hour == 0)`

**Data Columns Used:** `disruption`, `is_peak_hour`, `airport_code`

**Unit:** Ratio (dimensionless)

**Computed At:** Query time per airport

**Dashboard Display:** Shown in Risk Overview tab as a contextual stat

**Dataset Values:**
- ORD peak disruption rate: 17.85% | ORD off-peak: ~18.2% → ratio ≈ 0.98
- MEM peak disruption rate: 14.57% | MEM off-peak: ~15.7% → ratio ≈ 0.93

---

## Summary Table for Person 3

| KPI | Column(s) | Pre-computed or Query-time | File |
|---|---|---|---|
| Disruption Rate | disruption, flight_date | Query-time | features_final.csv |
| Average Recovery Time | delay_minutes, disruption | Query-time | features_final.csv |
| Weather Sensitivity | storm_flag, disruption | Pre-computed | airport_resilience_inputs.csv |
| Current Risk Level | all features | Real-time (model) | predict.py |
| Resilience Index | resilience_index, tier | Pre-computed | airport_resilience_inputs.csv |
| Peak Hour Premium | disruption, is_peak_hour | Query-time | features_final.csv |
