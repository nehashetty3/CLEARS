# CLEARS - Data Quality Report (v2)
**Owner:** Person 1 - Data & Analytics Lead
**Dataset audited:** `features_final_v2.csv` (original `features_final.csv` re-audited for comparison)
**Audit date:** November 2026 (supersedes the September 2026 report)

v2 changes only the four rolling columns (see `feature_dictionary_v2.md`, Section 5). Everything else is identical to the original file. This report is stricter than the first one: it records what was verified, what was corrected, and what is still open.

---

## Check 1 - Data leakage (rolling features)
**Original result: FAIL (it had been reported PASS). v2: PASS.**

The original test only confirmed that the first row of each airport was 0.0, which is true of almost any rolling feature and cannot detect leakage. Re-testing the original file showed:
- `timestamp` is hourly, so `shift(1)` followed by a 1-hour rolling window contained only earlier-listed flights **from the same scheduled hour**. `delay_last_1h` correlated 0.96 with the mean disruption of those flights, whose outcomes (delay > 15 min) are not known when a flight is scheduled.
- `congestion_level` correlated 0.70 with a flight's row position inside its hour, i.e. it partly encoded row order.

v2 uses an availability rule: history windows end two hours before the flight's scheduled hour (a flight's status is known about 15 minutes after its scheduled time, i.e. by the end of the next hour). `congestion_level` counts scheduled departures in the two previous hours.

**Verification of v2:**
- 8 random rows (both airports) recomputed independently from raw rows: 8 / 8 match.
- Every rolling feature has zero variation within a scheduled hour.
- `tests/test_features.py` flips the outcomes of flights in hours t-1 and t and checks that the features of hour t do not change; it also checks they do change when hour t-2 changes. Both pass.
- `src/features.py` rebuilds all 38 columns from the raw columns: the 34 non-rolling columns match the original file exactly and the 4 rolling columns match v2 exactly.
- Correlation with `disruption`, original -> v2: `delay_last_1h` 0.302 -> 0.262; `delay_last_3h` 0.347 -> 0.265; `delay_last_24h` 0.254 -> 0.227; `congestion_level` 0.035 -> 0.076. The cascade signal is real but smaller than first reported.
- Rows with no earlier-hour traffic (05:00-06:00) carry the last known rate forward for up to 12 hours, else 0.0 (about 0.1-0.4% of rows had no history at all).

---

## Check 2 - Label integrity
**Result: PASS (re-verified)**

Zero rows with `disruption = 1` and `dep_delay <= 15`, zero with `disruption = 0` and `dep_delay > 15`, zero nulls across 370,729 rows. No duplicate rows.

Caveat: `cancelled` is not part of the label. 295 cancelled rows remain and 47.5% of them have `disruption = 0`. See Check 9.

---

## Check 3 - Train/test split
**Result: PASS with caveats**

The split is time-ordered. Corrections to the original description:
- The cut is by **row count (20% of rows)**, not "20% of dates", and falls mid-day at 2022-01-31 09:00. The date 2022-01-31 therefore appears in both train and test (the original said "Overlap: None").
- **Recommended whole-day boundary:** train = before 2022-02-01 (297,121 rows, disruption rate 17.73%); test = 2022-02-01 onward (73,608 rows = 19.85%, rate 18.39%; ORD 67,827, MEM 5,781).
- The test set contains only Apr, Jul and Oct 2022 - **no winter month** - so winter performance is not measured. Report metrics per airport and per month.
- Regimes differ between train and test: disruption rate by year is 2019 21.7%, 2020 10.5%, 2021 17.0%, 2022 19.0%. **MEM has a higher disruption rate than ORD in every 2022 month** (Apr 22.3% vs 18.6%, Jul 25.8% vs 23.2%, Oct 14.2% vs 12.6%), whereas over the full data ORD is higher. Airport-level rates therefore shift between train and test.

---

## Check 4 - Class balance
**Result: FLAG - action for Person 2 (unchanged)**

Disruption = 17.86% (66,213 of 370,729); class 0 = 304,516. `scale_pos_weight` is about 4.6. By airport: ORD 18.06%, MEM 15.48%.

---

## Check 5 - Airport coverage
**Result: PASS (wording corrected)**

ORD 341,399 rows and MEM 29,330 rows, both well above 500. Both airports have data on the same **492 dates**: January, April, July and October of 2019-2022. The calendar span is 1,400 days, but only 35% of those days are present, so `data_period_days` overstates coverage. v2 adds `days_with_data = 492` to the resilience inputs.

---

## Check 6 - Column name consistency
**Result: PASS**

v2 has the same 38 columns, in the same order, with the same names and dtypes as the original. Only the four rolling columns differ in value. No nulls in any column.

---

## Check 7 - Null values in target
**Result: PASS (re-verified)** - zero nulls in `disruption`.

---

## Check 8 (new) - Weather / flight time alignment
**Result: FAIL - OPEN, needs the raw data**

Mean temperature by `hour_of_day` has its minimum at hours 11-13 and its maximum at hours 19-22 in every month (Jan, Apr, Jul, Oct) at both airports (see section 10 of `notebooks/01_eda_v2.ipynb`). Local-time expectations are a minimum near 06:00 and a maximum near 15:00. The offset of roughly 5-6 hours matches UTC vs US Central. Supporting evidence: aggregate disruption correlates most strongly with weather labelled 2-6 hours *after* the flight hour.

**Likely cause:** NOAA ISD times (UTC) merged onto BTS local scheduled hours without timezone conversion.
**Fix written:** `src/data_prep.py` converts ISD times to `America/Chicago` before the merge and matches each flight to the latest observation at or before its scheduled minute. It is covered by synthetic-data tests but **has not been run on the real raw files**; they must be run through it, and the resulting dataset must be re-checked (expected temperature minimum around 05-07 and maximum around 14-16).
**Impact until regenerated:** all weather columns and weather-derived features describe conditions about 5-6 hours before departure. Weather sensitivity scores and weather conclusions from SHAP are provisional.

---

## Check 9 (new) - Cancelled flights
**Result: FLAG - decision needed**

Only 295 cancelled flights remain (0.08% of rows). Rows with null `dep_delay` were dropped, which removes most cancellations - often the most severe weather outcomes. `data_prep.clean_labels` now has an explicit `cancelled_policy` ('drop' keeps the old behaviour; 'disrupt' keeps cancelled flights and labels them disrupted). The team must choose; KPIs 1-3 and 5 are lower bounds until then.

---

## Check 10 (new) - Redundant or misleading features
**Result: FLAG - documented**

| Feature | Finding |
|---|---|
| `humidity_stress` | Constant 0 |
| `hub_size_enc` | Constant 1 |
| `is_holiday_season` | Identical to `month == 10` (only Jan/Apr/Jul/Oct present) |
| `wind_category` | `severe` never occurs (max wind 30.3 kn); `storm_flag` never fires on wind |
| `runway_count` | Perfect proxy for `airport_code` (ORD 8, MEM 4) |
| `is_peak_hour` | No premium: peak 17.59% vs off-peak 18.23% (ratio 0.97, correlation -0.008); covers 57.7% of rows |

---

## Check 11 (new) - Columns that leak the label
**Result: FLAG - action for Person 2**

`dep_delay`, `arr_delay`, `delay_minutes`, `weather_delay`, `carrier_delay`, `nas_delay`, `late_aircraft_delay` and `cancelled` are post-departure information; `dep_delay > 15` *is* the label. The task guide's instruction to drop only `airport_code`, `timestamp` and `delay_minutes` is not enough. Use the explicit allowed-feature list in `feature_dictionary_v2.md`, Section 9, and add a test that fails if a banned column enters the feature matrix.

---

## Check 12 (new) - Code and tests
**Result: PASS for what could be tested; one item untested**

- `tests/test_features.py` (3 tests) and `tests/test_data_prep.py` (5 tests): 8 passed.
- `src/features.py`: reproduces the delivered data exactly (see Check 1).
- `src/resilience.py`: reproduces `airport_resilience_inputs_v2.csv` exactly.
- `src/data_prep.py`: ISD parsing, UTC -> local conversion, backward weather merge, 2400 midnight handling, label rules and the end-to-end flow into `build_features` are tested on synthetic data. **Not tested on the real raw files**; the ISD `AA1` precipitation scaling (depth in mm x 10) is an assumption taken from the ISD format and should be checked against the original data on the first real run.

---

## Summary

| Check | Original | v2 | Notes |
|---|---|---|---|
| 1 Data leakage | PASS (invalid test) | PASS | Rolling features rebuilt; verified |
| 2 Label integrity | PASS | PASS | Cancelled flights not in label (Check 9) |
| 3 Train/test split | PASS | PASS with caveats | Shared date; no winter in test; MEM > ORD in 2022 |
| 4 Class balance | FLAG | FLAG | `scale_pos_weight` about 4.6 |
| 5 Airport coverage | PASS | PASS | 492 days with data, not 1,399 |
| 6 Column consistency | PASS | PASS | Names unchanged |
| 7 Null target | PASS | PASS | |
| 8 Weather time alignment | - | FAIL (open) | Needs a run on the raw data |
| 9 Cancelled flights | - | FLAG | Policy decision needed |
| 10 Redundant features | - | FLAG | Documented |
| 11 Label-leaking columns | - | FLAG | Explicit feature list provided |
| 12 Code and tests | - | PASS (data_prep untested on raw data) | 8 tests pass |

**Overall: 7 PASS, 1 FAIL (open: weather timezone), 4 FLAG.**
