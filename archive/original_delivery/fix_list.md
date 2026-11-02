# CLEARS — Fix List

> **STATUS UPDATE (2 Nov 2026): this list is historical.** It was written before the raw data was downloaded. Most items are now resolved or superseded in `fixed/` (see `fixed/README.md` and `fixed/data_quality_report_v3.md`). Still open: items 8 (cancelled-flight policy), 11 (recommendation thresholds) and 5 (resilience index with two airports), plus the report/PDF items (19-23) for Person 4. Two statements below were **corrected by the raw data**: item 1's cause is confirmed (NOAA UTC vs local time, fixed), and item 5's old "30 / 70" scores are now 50 / 50. Item 2 was smaller than feared (rolling-feature correlation fell from ~0.30-0.35 to ~0.26).


Based on an audit of `features_final.csv` against the docs, the EDA notebook and the task guide.
Priority: **P0** = invalidates model results, fix before training. **P1** = wrong or misleading output. **P2** = cleanup.
Owners follow the Task Guide (Person 1–4).

---

## P0 — Fix before any model is trained

### 1. Weather is ~5–6 hours out of sync with flights  *(Person 1 — `src/data_prep.py`)*
- **Evidence:** In every month and at both airports, mean temperature is lowest at hour 11–13 and highest at 19–22 (true local: ~6am low, ~3pm high). Disruption correlates best with weather labelled 2–6 hours *after* the flight hour.
- **Likely cause:** NOAA ISD timestamps are UTC; BTS `CRS_DEP_TIME` is local. They were merged without conversion. Both airports are US Central time.
- **Fix:**
  1. In `load_weather`, localise to UTC and convert to `America/Chicago` (DST-aware) before `merge_asof`.
  2. Regenerate `clean_data.csv` → `features_final.csv` → `airport_resilience_inputs.csv`.
  3. Re-check: mean temperature by hour should bottom out around 5–7 and peak around 14–16.
  4. Add a "timezone handling" line to `feature_dictionary.md` (Section 2 and Appendix).
- **Verify using:** `df.drop_duplicates(['airport_code','timestamp']).groupby('hour_of_day').temperature_c.mean()`.

### 2. Rolling features use outcomes that are unknown at prediction time  *(Person 1 — `src/features.py`)*
- **Evidence:** `timestamp` is hourly, so `rolling('1h')` after `shift(1)` only covers earlier-listed flights **in the same scheduled hour** (corr 0.96 with that same-hour mean). Those flights' delays are realised after the scheduled time. `congestion_level` correlates 0.70 with a flight's row position inside its hour, so it partly encodes row order. `delay_last_1h/3h` are the strongest correlates of the label (0.30 / 0.35), so any model will lean on them.
- **Fix:**
  1. Keep the minute-level scheduled time (`CRS_DEP_TIME`) instead of truncating to the hour.
  2. Build history features from information available at prediction time: for a flight scheduled at T, use only flights whose **actual** departure (scheduled + `dep_delay`) is before T, or, simpler, only flights scheduled before T − 1h (hour-level aggregates of the previous hours).
  3. Compute `congestion_level` as scheduled departures in a fixed window before T, not a running row count.
  4. Sort deterministically (`airport_code`, scheduled time, flight id) before any rolling operation.
  5. Rebuild the features and replace the old values.
- **Also:** Data-quality Check 1 only verifies that the first row is 0.0, which can't detect this. Replace it with a test that recomputes one feature by hand for 5 random rows using only information available before T (this is what T-4.4 already asks for).

### 3. Delay and label columns will leak if used as features  *(Person 2 — `src/model_train.py`; Person 1 — docs)*
- **Evidence:** The Task Guide (T-2.3) only says to drop `airport_code`, `timestamp`, `delay_minutes`. `dep_delay` alone reconstructs `disruption` exactly.
- **Fix:** Define one explicit `FEATURE_COLS` list (names from the dictionary Sections 3–6 inputs plus raw weather, time, history and airport columns) and use it for training. Never train on: `disruption`, `delay_minutes`, `dep_delay`, `arr_delay`, `cancelled`, `weather_delay`, `carrier_delay`, `nas_delay`, `late_aircraft_delay`, `flight_date`, `timestamp`, `airport_code`. Encode or drop the string columns (`wind_category`, `rain_severity`, `hub_size`). Add a test that fails if any banned column is in the feature matrix.

### 4. Severity labels would be NaN for most rows  *(Person 2 — T-2.4)*
- **Evidence:** 63% of `delay_minutes` is negative (early departures). `pd.cut(..., bins=[0,15,45,120,9999])` turns all of them into NaN.
- **Fix:** Build the label as: class 0 if `disruption == 0`; otherwise `pd.cut(delay_minutes, [15,45,120,9999], labels=[1,2,3])`. Observed class sizes for disrupted flights: minor 32,659, moderate 23,907, severe 9,647. Note 1,351 rows sit at the 300-minute cap.

---

## P1 — Wrong or misleading results

### 5. Resilience Index is degenerate with 2 airports  *(Person 1 + 2 — `src/resilience.py`, `kpi_definitions.md`, report Table 4.4)*
- **Evidence:** Min–max over two airports sends every dimension to exactly 0 or 1. ORD = 0.35 + 0.20 + 0.15 = 0.70 vulnerability → 30; MEM = 0.30 → 70. A 68.2 vs 70.6 minute recovery gap carries the full 0.30 weight. MEM at exactly 70 sits on the Moderate/Resilient boundary.
- **Fix (pick one):**
  - Add more airports (the guide recommends 2–4; 4+ makes min–max meaningful), **or**
  - Replace min–max with fixed reference benchmarks (e.g. national BTS disruption-rate and delay-minute ranges) or z-scores against a broader airport set, **and**
  - State the 2-airport limitation in the report, and drop the "generalisable across airports" claim until it is true.
- Also fix the tier ranges: `0–40 / 41–70 / 71–100` leaves gaps for scores like 40.5 and 70.5. Use `<40`, `40–<70`, `≥70` or similar.

### 6. `is_peak_hour` contradicts the data  *(Person 1 — `features.py`, `kpi_definitions.md`, EDA cell 12)*
- **Evidence:** Peak 17.59% vs off-peak 18.23% disruption (correlation −0.008). Hours 5–9 have the *lowest* rates (8–13%), hours 16–22 the highest (21–25%). The flag also covers 57.7% of rows, so it is not "peak".
- **Fix:** Either redefine it from traffic volume (flights per hour above the airport's median) or replace it with an `evening` flag, and let the model use `hour_of_day`. Update the EDA text (it currently says the data "validates" the feature) and KPI 6.
- **KPI 6 numbers:** MEM off-peak is 16.71%, so the ratio is **0.87**, not 0.93 (ORD 0.97).

### 7. Train/test split  *(Person 1 — `data_quality_report.md`; Person 2)*
- **Evidence:** Cut falls at 2022-01-31 09:00, so that date appears in both sets ("Overlap: None" is not strictly true). The test set is 579 rows from late January plus Apr/Jul/Oct 2022, so it has effectively no winter. Disruption rate is 17.7% train vs 18.3% test; yearly rates range from 10.5% (2020) to 21.7% (2019).
- **Fix:** Split on a whole-date boundary. Report test metrics per airport and per month. Consider a second, season-balanced validation (e.g. train on 2019–2021, test on all of 2022 and separately on Jan–Oct blocks) and say plainly that the test set lacks winter. Change "most recent 20% of dates" to "most recent 20% of rows".

### 8. Cancelled flights are mostly missing  *(Person 1 — `data_prep.py`, docs)*
- **Evidence:** Rows with null `dep_delay` were dropped, which removes most cancellations. Only 295 cancelled rows remain, and 47% of them are labelled `disruption = 0`.
- **Fix:** Decide the policy. Preferably treat cancelled flights as `disruption = 1` and keep them (give `delay_minutes` a sentinel such as 300). If dropping stays, document it in the dictionary and report Limitations, since weather-driven cancellations are the most severe disruptions. Recompute all KPIs afterwards.

### 9. `data_period_days` is misleading  *(Person 1 — `build_resilience_inputs`, `kpi_definitions.md`, `data_quality_report.md`)*
- **Evidence:** It is 1399 (calendar span), but data exists for only 492 dates (Jan/Apr/Jul/Oct).
- **Fix:** Output both `data_period_days` (span) and `days_with_data` (unique dates). Replace "1,399-day period" in the quality report and dictionary header with "4 months × 4 years (492 days)". The report's "Jan 2019 – Oct 2022" should say the same.

### 10. Dead or redundant features  *(Person 1 — `features.py`, dictionary)*
- `humidity_stress` is always 0 and `hub_size_enc` is always 1: remove both from the model inputs (keep the dictionary note).
- `is_holiday_season` is identical to `month == 10` here. Either rename to reflect that, or drop it. The rationale text mentioning Thanksgiving/Christmas/spring break is not true for this data.
- `storm_flag` never fires on wind (max 30.3 knots; 0 rows over 50). The `severe` wind category never occurs. Fix the dictionary wording and the T-2.8 test inputs, which currently use `wind_category = 'severe'`.
- `runway_count` is a perfect proxy for `airport_code`; it adds nothing with two airports.

### 11. Recommendation and KPI thresholds disagree  *(Person 1, 3, 4)*
- KPI 4 (and T-1.6): Low < 0.30, Medium 0.30–0.65, High > 0.65.
- Report 4.9 and T-3.6: low ≤ 0.50, medium 0.50–0.80, high > 0.80.
- **Fix:** Pick one set, ideally calibrated on the test set (the base rate is ~18%, so 0.65 or 0.80 may rarely fire). Apply it in the KPI doc, `recommend.py`, the dashboard gauge and the report.

---

## P2 — Documentation and report cleanup

| # | Issue | Where | Fix |
|---|---|---|---|
| 12 | EDA says ORD has 356,828 flights | `01_eda.ipynb` cell 6 | Change to 341,399 |
| 13 | EDA calls `delay_last_1h` (0.3016) the strongest predictor; `delay_last_3h` is 0.3472 | cell 24 | Correct, and note the leakage caveat from item 2 |
| 14 | EDA says disruption "peaks 14:00–18:00"; data peaks around 19:00–23:00 | cell 12 | Re-read the chart; correct the text |
| 15 | EDA says weekend disruption is "marginally lower"; weekend is 19.3% vs weekday 17.3% | cell 15 | Correct (Sunday is highest at 20.3%) |
| 16 | `delay_last_*` first-row check is described as "PASS" | `data_quality_report.md` Check 1 | Re-run after item 2 and describe the new test |
| 17 | `hub_size` description: "Both airports are large_airport type" but the column is `large_hub` | dictionary Sec. 8 | Align wording with values |
| 18 | `wind_category` formula lists 4 labels but only 3 occur; range stated as calm/moderate/strong | dictionary Sec. 3 | Add "severe: 0 rows in data" |
| 19 | Report abstract says F1 > 0.70 "achieved"; no model results exist yet | Abstract, Ch. 5 | State actual F1 and ROC-AUC after retraining; weather correlations are all < 0.08, so 0.70 may not be reachable without the leaky features |
| 20 | Report claims SHAP "confirmed" top features | Ch. 5.2 | Only keep after running `explain.py` and saving `assets/shap_summary.png` |
| 21 | Citations: text cites Xu, Tian & Ning (2019), Manna, Biswas & Saha (2020), Ip, Wang & Cheung (2011); reference list has Xu et al. (2005), Manna et al. (2017), Ip, Wong & Cheung (2011) | Report Ch. 2 vs References | Make in-text citations match the reference list; verify each reference exists |
| 22 | Cover says Academic Year 2024–2025; docs are dated September 2026 | Report cover | Use the correct year |
| 23 | Student/guide details page is blank | Report p. 21 | Fill in |
| 24 | `deli.md` is empty | folder | Delete, or write the intended content |
| 25 | `docs/data_sources.md`, `model_report.md`, `review_log.md`, `label_verification.md` are not in the folder | guide T-1.1, T-2.7, T-4.4, T-4.6 | Add before the final report |

---

## Suggested order
1. Items 1, 2, 8 (data rebuild) → regenerate `features_final.csv` and `airport_resilience_inputs.csv`.
2. Items 3, 4, 7, 10 (Person 2 training setup).
3. Items 5, 6, 9, 11 (KPIs and resilience), then update `kpi_definitions.md`.
4. Re-run the data quality audit with the new checks, then items 12–25.
