> **v3** - numbers recomputed on `data/processed/features_final.csv` (regenerated from the raw NOAA/BTS files; see `data_quality_report.md`).

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

**Caveat:** Data covers only Jan/Apr/Jul/Oct of each year, and most cancelled flights were dropped during cleaning (only 287 remain; 15,875 were removed), so the rate is a lower bound for true disruption. If cancelled flights were counted as disrupted the overall rate would be about 21.3% instead of 17.8%.

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

**Note:** `delay_minutes` is capped at 300 to prevent extreme outliers from distorting this metric (1,308 disrupted flights sit at the cap). This is the average *delay* of disrupted flights, not a measured time-to-recover.

---

## KPI 3 — Weather Sensitivity Score

**Plain English:** How strongly correlated weather danger events are with disruptions at a given airport. Higher score = more weather-vulnerable airport.

**Formula:** `Pearson correlation coefficient between storm_flag and disruption` computed over the full dataset period for each airport

**Data Columns Used:** `storm_flag`, `disruption`, `airport_code`

**Unit:** Correlation coefficient (-1 to +1)

**Computed At:** Pre-computed (stored in `airport_resilience_inputs.csv`)

**Dashboard Display:** Shown in resilience panel per airport. ORD = 0.0429, MEM = 0.0521 (v3 data)

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

**Open decision:** the project report (§4.9) and task guide (T-3.6) use different bands for the recommendation engine (> 0.80 / 0.50–0.80 / ≤ 0.50). The team must pick **one** set of thresholds and apply it here, in `recommend.py`, the dashboard and the report. With an ~18% base rate, calibrate on the test set so that High can actually trigger.

---

## KPI 5 — Resilience Index

**Plain English:** A single 0–100 score summarising how well an airport handles weather stress. Higher = more resilient. Computed from four dimensions: disruption frequency, recovery speed, weather sensitivity, and congestion sensitivity.

**Formula:**
1. Min-max normalise each dimension across airports to [0,1]
2. Vulnerability = 0.35 × disruption_rate + 0.30 × avg_recovery_time + 0.20 × weather_sensitivity + 0.15 × congestion_sensitivity
3. Resilience Index = 100 − (vulnerability × 100)

**Tier Classification (continuous boundaries, v2):**
- ≤ 40: Vulnerable
- > 40 and ≤ 70: Moderate
- > 70: Resilient

(v1 listed 0–40 / 41–70 / 71–100, which left scores such as 40.5 and 70.5 unassigned; the boundaries above give identical tiers for every current score.)

**Data Columns Used:** `disruption_rate`, `avg_recovery_time`, `weather_sensitivity`, `congestion_sensitivity` from `airport_resilience_inputs.csv`

**Unit:** Score (0–100) + tier label

**Computed At:** Pre-computed (stored in `airport_resilience_inputs.csv`)

**Dashboard Values (v3 data):** ORD = 50.0 (Moderate) | MEM = 50.0 (Moderate)

**Important limitation:** with only two airports, min-max scaling sends every dimension to exactly 0 or 1, so the index depends only on *which* airport is higher on each dimension, not by how much. On the v3 data ORD is higher on disruption rate (18.04% vs 15.42%) and congestion sensitivity (0.080 vs 0.050), MEM is higher on average delay (70.3 vs 68.2 min) and weather sensitivity (0.052 vs 0.043): ORD = 0.35 + 0.15 = 0.50 vulnerability, MEM = 0.30 + 0.20 = 0.50, so both score **50**.
On the original data (weather misaligned, duplicated flights) MEM's weather sensitivity was 0.010, which gave ORD 30 (Vulnerable) and MEM 70 (Moderate). One reversed ordering changed the result completely: **the earlier "ORD is Vulnerable, MEM is Moderate" conclusion was an artefact** of the two-airport scaling and the data problems. The index is only meaningful with >= 4 airports or fixed external reference ranges; do not describe it as generalisable until then.
Inputs: `data/processed/airport_resilience_inputs.csv`; output: `data/processed/resilience_scores.csv`.

---

## KPI 6 — Peak Hour Disruption Premium

**Plain English:** How much more likely disruptions are during peak hours compared to off-peak hours. A value of 1.5 means disruptions are 50% more likely during peak hours; a value **below 1 means peak hours are less disrupted** (the case in this data).

**Formula:** `disruption_rate(is_peak_hour == 1) / disruption_rate(is_peak_hour == 0)`

**Data Columns Used:** `disruption`, `is_peak_hour`, `airport_code`

**Unit:** Ratio (dimensionless)

**Computed At:** Query time per airport

**Dashboard Display:** Shown in Risk Overview tab as a contextual stat

**Dataset Values:**
- ORD peak disruption rate: 17.84% | ORD off-peak: 18.31% → ratio ≈ 0.97
- MEM peak disruption rate: 14.41% | MEM off-peak: 16.85% → ratio ≈ 0.86
- Both airports: peak 17.59% vs off-peak 18.20% → ratio ≈ 0.97

**Interpretation:** the `is_peak_hour` window (06–09, 16–20) mixes the lowest-risk hours (06–09) with high-risk hours (16–20), so the premium is below 1. Disruption actually rises steadily through the day; consider a by-hour chart or a 16–22 vs 05–12 ratio for the dashboard.

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
