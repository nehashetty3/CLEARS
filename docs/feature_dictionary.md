# CLEARS — Feature Dictionary
**File:** `data/processed/features_final.csv`
**Rows:** 358,594 (ORD 332,108, MEM 26,486) | **Airports:** ORD, MEM | **Period:** Jan 2019 – Oct 2022, **only the months Jan, Apr, Jul, Oct of each year (492 days with data per airport)**
**Last Updated:** November 2026 | **Owner:** Person 1 — Data & Analytics Lead
**Source:** regenerated from the raw NOAA and BTS files with `src/data_prep.py` + `src/features.py`. Same 38 columns and names as the original `archive/original_delivery/features_final.csv`. Differences from the original: (1) weather is in correct local time (the original was UTC-labelled, 5-6 h early); (2) 12,135 duplicated flight records removed; (3) the four rolling features rebuilt without leakage (v2).

> This is the team-wide reference. Column names here are frozen. Any change requires updating this file and notifying all team members.

---

## Section 1 — Identifiers

| Column | Type | Range / Values | Nulls | Description |
|---|---|---|---|---|
| `airport_code` | string | ORD, MEM | 0 | IATA airport code. ORD = Chicago O'Hare, MEM = Memphis International |
| `timestamp` | datetime | 2019-01-01 to 2022-10-31 | 0 | Scheduled departure datetime (FlightDate + scheduled departure hour), BTS local time (US Central). **Hourly resolution** — all flights in the same hour share one timestamp. |
| `flight_date` | date | 2019-01-01 to 2022-10-31 | 0 | Date of the flight, sourced from BTS FL_DATE field |

---

## Section 2 — Raw Weather Variables
*Source: NOAA Integrated Surface Database (ISD), Station 72530094846 (KORD) and 72334013893 (KMEM). Parsed from encoded ISD fields.*

> **Timezone — fixed in v3.** NOAA ISD times are UTC. `data_prep.load_weather` converts them to `America/Chicago` (daylight saving included) before matching, and each flight receives the latest observation made at or before its scheduled departure minute (at most 1 h old; missing values carried forward at most 6 h, then the column median). Daily/monthly summary records (`SOD`, `SOM`) are excluded. Evidence: the original temperature at hour *t* equals the new local temperature 6 h earlier in January and 5 h earlier in April/July/October (correlation 0.9997-0.9999, identical values in 98% of hours); after the fix the mean temperature bottoms out at 06-07 h and peaks at 16 h. The original weather columns were therefore about 5-6 hours stale.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `wind_speed_knots` | float | knots | 0.0 – 35.0 | 0 | Wind speed at departure airport. Parsed from NOAA WND field (original m/s × 10), converted to knots (× 1.94384). Missing code 9999 replaced with NaN (as are ISD values flagged suspect/erroneous, quality 2/3/6/7); gaps are then carried forward for at most 6 h and any remainder filled with the column median. |
| `visibility_km` | float | km | 0.2 – 16.09 | 0 | Horizontal visibility at airport. Parsed from NOAA VIS field (original metres), divided by 1000. Missing code 999999 replaced with NaN (as are ISD values flagged suspect/erroneous, quality 2/3/6/7); gaps are then carried forward for at most 6 h and any remainder filled with the column median. |
| `temperature_c` | float | °C | -30.6 – 38.9 | 0 | Air temperature. Parsed from NOAA TMP field (original °C × 10), divided by 10. Missing code 9999 replaced with NaN (as are ISD values flagged suspect/erroneous, quality 2/3/6/7); gaps are then carried forward for at most 6 h and any remainder filled with the column median. |
| `humidity_pct` | float | % | 12.3 – 100.0 | 0 | Relative humidity. Derived from temperature and dewpoint using the Magnus formula: 100 × exp(17.625×Td/(243.04+Td)) / exp(17.625×T/(243.04+T)). Clipped to [0, 100]. |
| `precip_mm_hr` | float | mm/hr | 0.0 – 41.4 | 0 | Precipitation intensity. Parsed from NOAA AA1 field (ISD depth is in tenths of mm, so depth/10 mm divided by the period in hours). No report -> 0. Summary records (`SOD`/`SOM`) are excluded. |

---

## Section 3 — Engineered Weather Features
*Derived from raw weather variables in src/features.py. All thresholds aligned with ICAO aviation standards.*

| Column | Type | Unit | Range | Nulls | Formula | Rationale |
|---|---|---|---|---|---|---|
| `wind_category` | string | category | calm, moderate, strong (`severe` is defined but has **0 rows** — max wind is 35.0 kn) | 0 | pd.cut(wind_speed_knots, bins=[0,10,25,50,9999], labels=['calm','moderate','strong','severe'], include_lowest=True) (include_lowest keeps wind = 0 in 'calm') | Bins wind into operationally meaningful categories aligned with ICAO crosswind limits |
| `wind_category_enc` | int | ordinal | 0–2 | 0 | calm=0, moderate=1, strong=2, severe=3 | Ordinal encoding of wind_category for ML model input |
| `visibility_risk` | int | binary | 0, 1 | 0 | 1 if visibility_km < 3 | ICAO Category I ILS minimum is 3 km. Below this, instrument approaches are required and disruption risk rises sharply |
| `rain_severity` | string | category | none, light, moderate, heavy | 0 | pd.cut(precip_mm_hr, bins=[-0.001,0,2.5,7.5,9999], labels=['none','light','moderate','heavy']) | Standard WMO precipitation intensity classification |
| `rain_severity_enc` | int | ordinal | 0–3 | 0 | none=0, light=1, moderate=2, heavy=3 | Ordinal encoding of rain_severity for ML model input |
| `temp_stress` | int | binary | 0, 1 | 0 | 1 if temperature_c > 35 OR temperature_c < -10 | Extreme heat causes tarmac softening and equipment stress; extreme cold triggers de-icing delays |
| `storm_flag` | int | binary | 0, 1 | 0 | 1 if (wind_speed_knots > 50) OR (visibility_km < 1) OR (rain_severity == 'heavy') | Composite high-danger indicator. Captures rare but severe disruption events (1,729 records = 0.48% of dataset). Here it fires on visibility < 1 km (1,216 rows) and heavy rain (522 rows); the wind > 50 kn condition never occurs (max wind 35.0 kn) |
| `humidity_stress` | int | binary | 0, 1 | 0 | 1 if humidity_pct > 90 AND temperature_c > 28 | Combined heat-humidity fog risk condition. **Constant 0 in this dataset — exclude from modelling** (both airports in temperate climate during study period) |

---

## Section 4 — Time-Based Features
*Derived from timestamp column in src/features.py.*

| Column | Type | Unit | Range | Nulls | Formula | Rationale |
|---|---|---|---|---|---|---|
| `hour_of_day` | int | hour | 0–23 | 0 | timestamp.dt.hour | Raw departure hour. Captures intraday disruption patterns |
| `day_of_week` | int | day | 0–6 | 0 | timestamp.dt.dayofweek (0=Monday) | Monday=0, Sunday=6. Captures weekly traffic patterns |
| `month` | int | month | {1, 4, 7, 10} only | 0 | timestamp.dt.month | Month of year. Captures seasonal weather patterns |
| `is_peak_hour` | int | binary | 0, 1 | 0 | 1 if hour_of_day in [6–9] OR [16–20] | Standard aviation peak departure/arrival windows. **Data note:** not associated with higher disruption here — peak 17.59% vs off-peak 18.20% (corr −0.008). Disruption rises through the day (≈8.5% at 05:00 to about 21–25% between 15:00 and 22:00), so the 06–09 part of the window is the *lowest-risk* period. Flag covers 57.8% of rows. Prefer `hour_of_day` for modelling |
| `is_weekend` | int | binary | 0, 1 | 0 | 1 if day_of_week >= 5 | Weekend flights have different delay patterns than weekday operations |
| `is_holiday_season` | int | binary | 0, 1 | 0 | 1 if month in [3, 10, 11, 12] | Intended to flag high-travel months, but only Jan/Apr/Jul/Oct are in the data, so here it is **identical to `month == 10`** (redundant with `month`; no March/November/December rows) |

---

## Section 5 — Operational History Features (Rolling/Lag) — **rebuilt in v2, unchanged in v3**
*Derived in `data/processed/features_final.csv` by `src/features.py`. `timestamp` has hourly resolution, so every flight scheduled in the same hour gets identical history values.*

**Availability rule (leakage fix).** A flight's disrupted / not-disrupted status is known roughly 15 minutes after its scheduled departure. With hourly timestamps, a flight scheduled in hour *h* is only guaranteed known by the end of hour *h+1*. For a flight scheduled in hour *t*, history windows therefore end at hour *t−2*; flights scheduled in hours *t−1* and *t* are never used. (v1 used a 1-hour window after `shift(1)`, which at hourly resolution contained only earlier-listed flights from the **same** hour — outcomes not yet known at prediction time — and `congestion_level` partly encoded row order inside the hour.)

| Column | Type | Unit | Range | Nulls | Formula | Rationale |
|---|---|---|---|---|---|---|
| `delay_last_1h` | float | rate (0–1) | 0.0 – 1.0 | 0 | Per-airport share of disrupted flights among flights scheduled in hour *t−2* | Very recent disruption state. Empty window (e.g. 05:00–06:00, no overnight traffic) → last known rate carried forward up to 12 h, else 0.0 |
| `delay_last_3h` | float | rate (0–1) | 0.0 – 1.0 | 0 | Per-airport disruption rate over flights scheduled in hours *t−4 … t−2* | Medium-term disruption waves (weather events, ground stops). Same empty-window rule |
| `delay_last_24h` | float | rate (0–1) | 0.0 – 1.0 | 0 | Per-airport disruption rate over flights scheduled in hours *t−25 … t−2* | Chronic congestion / extended weather events. Same empty-window rule |
| `congestion_level` | float | count | 0 – 160 | 0 | Per-airport number of scheduled departures in hours *t−2* and *t−1* | How busy the airport was just before this flight. Uses the schedule only (known in advance) |

**Effect of the fix** (correlation with `disruption`, original file -> v3): `delay_last_1h` 0.302 → 0.264, `delay_last_3h` 0.347 → 0.267, `delay_last_24h` 0.254 → 0.227, `congestion_level` 0.035 → 0.076. The delay-cascade signal is real but smaller than the original suggested. Verified by brute-force recomputation of random rows and by `tests/test_features.py` (flipping the outcomes of hours *t-1* and *t* leaves the features of hour *t* unchanged).

---

## Section 6 — Target Labels

| Column | Type | Unit | Range | Nulls | Formula | Rationale |
|---|---|---|---|---|---|---|
| `disruption` | int | binary | 0, 1 | 0 | 1 if dep_delay > 15 minutes, else 0 | Standard FAA/ICAO threshold for a reportable delay. Class balance: 0=294,598 (82.2%), 1=63,996 (17.8%) |
| `delay_minutes` | float | minutes | -59 – 300 | 0 | dep_delay capped at 300 minutes | Numeric delay target for severity model. Capped at 300 to reduce outlier influence |

---

## Section 7 — Raw BTS Flight Columns (Reference)
*Source: BTS Reporting Carrier On-Time Performance. Retained for analysis. **These columns must NOT be used as model inputs** — `dep_delay` alone reconstructs `disruption` exactly (see Section 9).*

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `dep_delay` | float | minutes | -59 – 1596 | 0 | Raw departure delay in minutes. Negative = early departure |
| `arr_delay` | float | minutes | -79 – 1597 | 0 | Arrival delay in minutes. Filled with dep_delay where null |
| `cancelled` | float | binary | 0, 1 | 0 | 1 if flight was cancelled. Only 287 cancelled rows remain: 15,875 cancelled flights have no departure delay and were removed by `clean_labels(cancelled_policy='drop')`; 52.6% of the remaining ones have `disruption = 1`. Most cancellations are absent, which understates disruption (`cancelled_policy='disrupt'` keeps them labelled as disrupted) |
| `weather_delay` | float | minutes | 0 – 1095 | 0 | Minutes of delay attributed to weather by carrier. Null filled with 0 |
| `carrier_delay` | float | minutes | 0 – 1499 | 0 | Minutes of delay attributed to carrier. Null filled with 0 |
| `nas_delay` | float | minutes | 0 – 1288 | 0 | Minutes of delay attributed to National Airspace System (ATC, congestion) |
| `late_aircraft_delay` | float | minutes | 0 – 1514 | 0 | Minutes of delay from a late-arriving aircraft on a prior leg |

---

## Section 8 — Airport Context Features
*Source: OurAirports airports.csv and runways.csv.*

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `runway_count` | int | count | 4–8 | 0 | Number of active (non-closed) runways. ORD=8, MEM=4. More runways = greater capacity under weather stress |
| `hub_size` | string | category | large_hub | 0 | FAA hub classification from OurAirports type field. Both airports are large_airport type |
| `hub_size_enc` | int | ordinal | 1 | 0 | Numeric encoding of hub_size. large_hub=1 (both airports in this study). **Constant — exclude from modelling.** `runway_count` is a perfect proxy for `airport_code` (ORD=8, MEM=4) |

---

## Section 9 — Modelling Feature Set (added in v2)
Use an explicit list; do not just "drop a few columns".

**Allowed inputs:** `wind_speed_knots`, `visibility_km`, `temperature_c`, `humidity_pct`, `precip_mm_hr`, `wind_category_enc`, `visibility_risk`, `rain_severity_enc`, `temp_stress`, `storm_flag`, `hour_of_day`, `day_of_week`, `month`, `is_peak_hour`, `is_weekend`, `delay_last_1h`, `delay_last_3h`, `delay_last_24h`, `congestion_level`, `runway_count` (optionally `is_holiday_season`, which equals `month == 10` here).

**Never use as inputs (label or post-departure information):** `disruption`, `delay_minutes`, `dep_delay`, `arr_delay`, `cancelled`, `weather_delay`, `carrier_delay`, `nas_delay`, `late_aircraft_delay`.

**Identifiers / non-numeric (drop or encode):** `airport_code`, `timestamp`, `flight_date`, `wind_category`, `rain_severity`, `hub_size`.

**Constant (no information):** `humidity_stress`, `hub_size_enc`.

**Severity label note:** 63% of `delay_minutes` is negative. Build the severity class as 0 if `disruption == 0`, else `pd.cut(delay_minutes, [15, 45, 120, 9999], labels=[1, 2, 3])`; binning from 0 turns negatives into NaN.

---

## Appendix — Data Sources

| Source | Dataset | URL | Station/Filter | Period |
|---|---|---|---|---|
| BTS | Reporting Carrier On-Time Performance | transtats.bts.gov | Illinois + Tennessee (ORD, MEM filtered) | Jan/Apr/Jul/Oct 2019–2022 (492 days with data) |
| NOAA ISD | Global Hourly Surface Observations | ncei.noaa.gov/access/search/data-search/global-hourly | Station 72530094846 (KORD), 72334013893 (KMEM) | 2019-01-01 to 2022-12-31 |
| OurAirports | airports.csv + runways.csv | davidmegginson.github.io/ourairports-data | ORD, MEM | Current (downloaded Sep 2026) |

---

## Appendix B — How v3 was regenerated (and what the original file contained)
Raw files (public): NOAA ISD hourly observations for stations 72530094846 (KORD) and 72334013893 (KMEM), years 2019-2022, from `noaa-global-hourly-pds.s3.amazonaws.com`; BTS On-Time Performance monthly zips for Jan/Apr/Jul/Oct 2019-2022, from `transtats.bts.gov/PREZIP`. Pipeline: `data_prep.run_pipeline` -> `features.build_features`.

Reconciliation with the original `archive/original_delivery/features_final.csv` (see `data_quality_report.md`, Checks 8 and 13):
- Every one of the 358,594 flights in v3 exists in the original with identical delay values.
- The original contains **12,135 additional rows** (ORD 9,291, MEM 2,844) that have no counterpart in the BTS data: they are exact copies of other flights' records (the number of excess duplicate records equals 12,135). MEM even had more rows (29,330) than BTS lists MEM departures including cancelled ones (27,423).
- After shifting by 5-6 h the original weather equals the regenerated weather (temperature, wind, visibility, precipitation identical in 96-98% of hours; humidity 84% identical, correlation 0.9992).
