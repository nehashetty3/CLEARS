# CLEARS - Data Quality Report
**Owner:** Person 1 - Data & Analytics Lead
**Dataset audited:** `data/processed/features_final.csv` (regenerated from the raw NOAA and BTS files), compared with the original `archive/original_delivery/features_final.csv`
**Audit date:** November 2026 (supersedes the September 2026 report and `archive/v2_intermediate/data_quality_report_v2.md`)

This is the first audit of data built from the raw source files (data version v3; the original delivery is v1, see `archive/`). It repairs three problems found in the original data: weather 5-6 hours out of sync with the flights, duplicated flight records, and leaking rolling features. Every statement below was checked on the data, not carried over.

---

## Check 1 - Data leakage (rolling features)
**Original: FAIL (reported PASS). v3: PASS.**

The original test only confirmed that the first row of each airport was 0.0, which cannot detect leakage. With hourly timestamps, `shift(1)` + a 1-hour window contained only earlier-listed flights **from the same scheduled hour** (`delay_last_1h` correlated 0.96 with that same-hour mean); `congestion_level` correlated 0.70 with a flight's row position inside its hour.
v3 ends every window two hours before the flight's scheduled hour (a flight's status is known about 15 minutes after its scheduled time) and counts scheduled departures in the two previous hours. Verified by brute-force recomputation of random rows, by `tests/test_features.py` (flipping the outcomes of hours t-1 and t leaves the features of hour t unchanged), and by the fact that all flights of one hour share identical history values. Correlation with `disruption`, original -> v3: `delay_last_1h` 0.302 -> 0.264, `delay_last_3h` 0.347 -> 0.267, `delay_last_24h` 0.254 -> 0.227, `congestion_level` 0.035 -> 0.076.

---

## Check 2 - Label integrity
**PASS (re-verified on v3)** - 0 rows with `disruption = 1` and `dep_delay <= 15`, 0 with `disruption = 0` and `dep_delay > 15`, 0 nulls in the target and 0 nulls in the whole matrix (358,594 rows). Cancelled flights are not part of the label (Check 9).

---

## Check 3 - Train/test split
**PASS with caveats**
- Recommended whole-day boundary: train = dates before 2022-02-01 (287,475 rows, disruption 17.72%), test = 2022-02-01 onward (71,119 rows = 19.83%, disruption 18.37%; ORD 65,891, MEM 5,228). No date is shared. The original description ("most recent 20% of dates", cut at 2022-01-31 09:00) split 2022-01-31 across both sets.
- The test set holds only April, July and October 2022 - **no winter month** - so winter performance is not measured. Report metrics per airport and per month.
- Disruption rates differ by year (2019 21.7%, 2020 10.5%, 2021 16.9%, 2022 19.0%). **In the test months MEM is more disrupted than ORD** (Apr 22.4% vs 18.6%, Jul 25.2% vs 23.3%, Oct 14.6% vs 12.5%), whereas over the whole period ORD is higher (18.04% vs 15.42%); ORD is higher in only 7 of the 16 months.

---

## Check 4 - Class balance
**FLAG - action for Person 2 (unchanged in substance):** disruption = 17.85% (63,996 of 358,594); class 0 = 294,598; `scale_pos_weight` = 4.60. By airport: ORD 18.04%, MEM 15.42%.

---

## Check 5 - Airport coverage
**PASS (wording corrected)** - ORD 332,108 rows, MEM 26,486 rows (both far above 500). Both airports have data on the same **492 dates**: January, April, July and October of 2019-2022, only 35% of the 1,400-day calendar span, so `data_period_days` overstates coverage (v3 adds `days_with_data`).

---

## Check 6 - Column name consistency
**PASS** - 38 columns, identical names and order to the original file; every column is documented in `feature_dictionary.md`.

---

## Check 7 - Null values in target
**PASS** - zero nulls in `disruption`.

---

## Check 8 (new) - Weather / flight time alignment
**Original: FAIL. v3: PASS - proven with the raw data.**

In the original file mean temperature bottomed out at hours 11-13 and peaked at 19-22. Regenerating from the raw NOAA files with an explicit UTC -> `America/Chicago` conversion shows why:
- Original temperature at hour *t* correlates **0.9999** with the correctly localised temperature **6 hours earlier in January** (CST) and **0.9997-0.9999 5 hours earlier in April, July and October** (CDT), i.e. the original is the UTC-labelled series, 5-6 hours stale. Temperature, wind, visibility and precipitation values are identical in 96-98% of hours after the shift (humidity 84% identical, correlation 0.9992), and the means agree to three decimals; this also confirms the ISD parsing rules and the precipitation scaling (depth in mm x 10) used in `data_prep.py`.
- v3 mean temperature by departure hour: lowest at 06-07 h and highest at 16 h in April, July and October (January: warms from 07 h to a peak at 14 h; its small winter amplitude is dominated by passing weather systems).
- Weather matters more once aligned: weather-only model ROC-AUC 0.590 (v2 data: 0.586) and the full model now reacts to weather overrides (+16 points for 10 mm/h rain, +33 points for combined severe weather) where it barely reacted before (+4 points).

---

## Check 9 (new) - Cancelled flights
**FLAG - policy decision needed.** The raw BTS data contain 15,875 cancelled departures (ORD 14,938, MEM 937) with no departure delay; `clean_labels(cancelled_policy='drop')` removes them (as the original did). 287 cancelled rows with a recorded delay remain, 52.6% of them labelled disrupted. If cancelled flights were counted as disrupted the overall rate would be about **21.3%** instead of 17.8%, and weather-driven disruption would be understated less. `cancelled_policy='disrupt'` implements that; the team must choose, then regenerate and re-train.

---

## Check 10 (new) - Redundant or misleading features
**FLAG - documented**

| Feature | Finding |
|---|---|
| `humidity_stress`, `hub_size`, `hub_size_enc` | Constant |
| `is_holiday_season` | Identical to `month == 10` (only Jan/Apr/Jul/Oct present) |
| `wind_category` | `severe` never occurs (max wind 35.0 kn); `storm_flag` never fires on wind (it fires on visibility < 1 km in 1,216 rows and heavy rain in 522 rows) |
| `runway_count` | Perfect proxy for `airport_code` (ORD 8, MEM 4) |
| `is_peak_hour` | No premium: peak 17.59% vs off-peak 18.20% (ratio 0.97, correlation -0.008); covers 57.8% of rows |

---

## Check 11 (new) - Columns that leak the label
**FLAG - action for Person 2.** `dep_delay`, `arr_delay`, `delay_minutes`, `weather_delay`, `carrier_delay`, `nas_delay`, `late_aircraft_delay` and `cancelled` are post-departure information; `dep_delay > 15` *is* the label. Use the explicit allowed-feature list in `feature_dictionary.md` Section 9 (`src/config.py` enforces it).

---

## Check 12 (new) - Code and tests
**PASS** - `tests/test_features.py` (3) and `tests/test_data_prep.py` (7): 10 passed. `src/data_prep.py` now **has been run on the real raw files** (16 BTS zips, 8 NOAA files): it reproduces the original pipeline apart from the timezone fix and the duplicates (Checks 8 and 13). It required three fixes found only on real data: BTS zips contain a `readme.html` next to the CSV; the ISD file contains daily/monthly summary records (`SOD`, `SOM`) that must not be matched to flights; and the zips must be read in chunks (all US flights).

---

## Check 13 (new) - Duplicated flights in the original data
**Original: FAIL. v3: PASS.**

Reconciling the original with the raw BTS departures of ORD and MEM:
- All 358,594 flights of v3 exist in the original with identical delay values (0 missing).
- The original has **12,135 extra rows** (ORD 9,291 = 2.7% and MEM 2,844 = 9.7% of each airport's original rows) that do not exist in BTS. MEM's original count (29,330) even exceeds the 27,423 MEM departures BTS lists including cancelled ones.
- The number of excess exact-duplicate flight records in the original (24,199 vs 12,064 in the real data, which has genuine coincidences) is **12,135 - exactly the number of extra rows**, and the extras appear in 49% of airport-hours (always in addition to, never instead of, the real flights). The extras are copies of real flights, probably created by a merge that duplicated rows.
- The earlier "no duplicates" statement was true only because the order-dependent history columns made every copy look different. The duplicates inflated `congestion_level` and weighted some flights twice.
v3 holds each BTS flight once.

---

## Summary

| Check | Original | v3 | Notes |
|---|---|---|---|
| 1 Data leakage | PASS (invalid test) | PASS | Rolling features rebuilt and tested |
| 2 Label integrity | PASS | PASS | 0 errors, 0 nulls |
| 3 Train/test split | PASS | PASS with caveats | Whole-day cut 2022-02-01; no winter in test; MEM > ORD in test months |
| 4 Class balance | FLAG | FLAG | `scale_pos_weight` 4.60 |
| 5 Airport coverage | PASS | PASS | 492 days with data |
| 6 Column consistency | PASS | PASS | Names and order unchanged |
| 7 Null target | PASS | PASS | |
| 8 Weather time alignment | - | **FAIL -> PASS** | UTC vs local proven and fixed |
| 9 Cancelled flights | - | FLAG | 15,875 removed; policy decision (rate 21.3% if counted) |
| 10 Redundant features | - | FLAG | Documented |
| 11 Label-leaking columns | - | FLAG | Explicit feature list |
| 12 Code and tests | - | PASS | 10 tests; pipeline run on real raw data |
| 13 Duplicated flights | - | **FAIL -> PASS** | 12,135 duplicate rows in the original |

**Overall: 9 PASS (one with caveats) and 4 FLAG (class balance, cancelled-flight policy, redundant features, label-leaking columns). The two FAILs found in the original data (weather alignment, duplicated flights) are resolved; the leakage check that wrongly passed in the original is fixed.**
