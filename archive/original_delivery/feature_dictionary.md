# CLEARS — Feature Dictionary
**File:** `data/processed/features_final.csv`
**Rows:** 370,729 | **Airports:** ORD, MEM | **Period:** Jan 2019 – Oct 2022
**Last Updated:** September 2026 | **Owner:** Person 1 — Data & Analytics Lead

> This is the team-wide reference. Column names here are frozen. Any change requires updating this file and notifying all team members.

---

## Section 1 — Identifiers

| Column | Type | Range / Values | Nulls | Description |
|---|---|---|---|---|
| `airport_code` | string | ORD, MEM | 0 | IATA airport code. ORD = Chicago O'Hare, MEM = Memphis International |
| `timestamp` | datetime | 2019-01-01 to 2022-10-31 | 0 | Scheduled departure datetime (FlightDate + scheduled departure hour) |
| `flight_date` | date | 2019-01-01 to 2022-10-31 | 0 | Date of the flight, sourced from BTS FL_DATE field |

---

## Section 2 — Raw Weather Variables
*Source: NOAA Integrated Surface Database (ISD), Station 72530094846 (KORD) and 72334013893 (KMEM). Parsed from encoded ISD fields.*

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `wind_speed_knots` | float | knots | 0.0 – 30.3 | 0 | Wind speed at departure airport. Parsed from NOAA WND field (original m/s × 10), converted to knots (× 1.94384). Missing code 9999 replaced with NaN then forward-filled. |
| `visibility_km` | float | km | 0.0 – 16.09 | 0 | Horizontal visibility at airport. Parsed from NOAA VIS field (original metres), divided by 1000. Missing code 999999 replaced with NaN then forward-filled. |
| `temperature_c` | float | °C | -30.6 – 38.9 | 0 | Air temperature. Parsed from NOAA TMP field (original °C × 10), divided by 10. Missing code 9999 replaced with NaN then forward-filled. |
| `humidity_pct` | float | % | 12.3 – 100.0 | 0 | Relative humidity. Derived from temperature and dewpoint using the Magnus formula: 100 × exp(17.625×Td/(243.04+Td)) / exp(17.625×T/(243.04+T)). Clipped to [0, 100]. |
| `precip_mm_hr` | float | mm/hr | 0.0 – 24.4 | 0 | Precipitation intensity. Parsed from NOAA AA1 field (depth_mm / period_hours). Null where no precipitation reported — filled with 0. |

---

## Section 3 — Engineered Weather Features
*Derived from raw weather variables in src/features.py. All thresholds aligned with ICAO aviation standards.*

| Column | Type | Unit | Range | Nulls | Formula | Rationale |
|---|---|---|---|---|---|---|
| `wind_category` | string | category | calm, moderate, strong | 0 | pd.cut(wind_speed_knots, bins=[0,10,25,50,9999], labels=['calm','moderate','strong','severe']) | Bins wind into operationally meaningful categories aligned with ICAO crosswind limits |
| `wind_category_enc` | int | ordinal | 0–2 | 0 | calm=0, moderate=1, strong=2, severe=3 | Ordinal encoding of wind_category for ML model input |
| `visibility_risk` | int | binary | 0, 1 | 0 | 1 if visibility_km < 3 | ICAO Category I ILS minimum is 3 km. Below this, instrument approaches are required and disruption risk rises sharply |
| `rain_severity` | string | category | none, light, moderate, heavy | 0 | pd.cut(precip_mm_hr, bins=[-0.001,0,2.5,7.5,9999], labels=['none','light','moderate','heavy']) | Standard WMO precipitation intensity classification |
| `rain_severity_enc` | int | ordinal | 0–3 | 0 | none=0, light=1, moderate=2, heavy=3 | Ordinal encoding of rain_severity for ML model input |
| `temp_stress` | int | binary | 0, 1 | 0 | 1 if temperature_c > 35 OR temperature_c < -10 | Extreme heat causes tarmac softening and equipment stress; extreme cold triggers de-icing delays |
| `storm_flag` | int | binary | 0, 1 | 0 | 1 if (wind_speed_knots > 50) OR (visibility_km < 1) OR (rain_severity == 'heavy') | Composite high-danger indicator. Captures rare but severe disruption events (1,679 records = 0.45% of dataset) |
| `humidity_stress` | int | binary | 0, 1 | 0 | 1 if humidity_pct > 90 AND temperature_c > 28 | Combined heat-humidity fog risk condition. No instances in this dataset (both airports in temperate climate during study period) |

---

## Section 4 — Time-Based Features
*Derived from timestamp column in src/features.py.*

| Column | Type | Unit | Range | Nulls | Formula | Rationale |
|---|---|---|---|---|---|---|
| `hour_of_day` | int | hour | 0–23 | 0 | timestamp.dt.hour | Raw departure hour. Captures intraday disruption patterns |
| `day_of_week` | int | day | 0–6 | 0 | timestamp.dt.dayofweek (0=Monday) | Monday=0, Sunday=6. Captures weekly traffic patterns |
| `month` | int | month | 1–10 | 0 | timestamp.dt.month | Month of year. Captures seasonal weather patterns |
| `is_peak_hour` | int | binary | 0, 1 | 0 | 1 if hour_of_day in [6–9] OR [16–20] | Standard aviation peak departure/arrival windows with highest congestion |
| `is_weekend` | int | binary | 0, 1 | 0 | 1 if day_of_week >= 5 | Weekend flights have different delay patterns than weekday operations |
| `is_holiday_season` | int | binary | 0, 1 | 0 | 1 if month in [3, 10, 11, 12] | Flags high-travel months: spring break (March), fall travel (October), Thanksgiving/Christmas (November/December) |

---

## Section 5 — Operational History Features (Rolling/Lag)
*Derived in src/features.py. All rolling features apply shift(1) before the rolling window to prevent data leakage — the current row is never included in its own feature.*

| Column | Type | Unit | Range | Nulls | Formula | Rationale |
|---|---|---|---|---|---|---|
| `delay_last_1h` | float | rate (0–1) | 0.0 – 1.0 | 0 | Per-airport rolling mean of disruption over prior 1 hour (shift(1) applied) | Captures very recent disruption cascades. A disrupted flight 30 mins ago strongly predicts the next one |
| `delay_last_3h` | float | rate (0–1) | 0.0 – 1.0 | 0 | Per-airport rolling mean of disruption over prior 3 hours (shift(1) applied) | Captures medium-term disruption waves from weather events or ground stops |
| `delay_last_24h` | float | rate (0–1) | 0.0 – 1.0 | 0 | Per-airport rolling mean of disruption over prior 24 hours (shift(1) applied) | Captures chronic congestion periods or extended weather events |
| `congestion_level` | float | count | 0 – 165 | 0 | Per-airport rolling flight count over prior 2 hours (shift(1) applied) | Measures how busy the airport was before this flight. High congestion amplifies weather-related disruption |

---

## Section 6 — Target Labels

| Column | Type | Unit | Range | Nulls | Formula | Rationale |
|---|---|---|---|---|---|---|
| `disruption` | int | binary | 0, 1 | 0 | 1 if dep_delay > 15 minutes, else 0 | Standard FAA/ICAO threshold for a reportable delay. Class balance: 0=304,516 (82.1%), 1=66,213 (17.9%) |
| `delay_minutes` | float | minutes | -59 – 300 | 0 | dep_delay capped at 300 minutes | Numeric delay target for severity model. Capped at 300 to reduce outlier influence |

---

## Section 7 — Raw BTS Flight Columns (Reference)
*Source: BTS Reporting Carrier On-Time Performance. Retained for analysis but not used as ML features directly.*

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `dep_delay` | float | minutes | -59 – 1596 | 0 | Raw departure delay in minutes. Negative = early departure |
| `arr_delay` | float | minutes | -79 – 1597 | 0 | Arrival delay in minutes. Filled with dep_delay where null |
| `cancelled` | float | binary | 0, 1 | 0 | 1 if flight was cancelled |
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
| `hub_size_enc` | int | ordinal | 1 | 0 | Numeric encoding of hub_size. large_hub=1 (both airports in this study) |

---

## Appendix — Data Sources

| Source | Dataset | URL | Station/Filter | Period |
|---|---|---|---|---|
| BTS | Reporting Carrier On-Time Performance | transtats.bts.gov | Illinois + Tennessee (ORD, MEM filtered) | Jan/Apr/Jul/Oct 2019–2022 |
| NOAA ISD | Global Hourly Surface Observations | ncei.noaa.gov/access/search/data-search/global-hourly | Station 72530094846 (KORD), 72334013893 (KMEM) | 2019-01-01 to 2022-12-31 |
| OurAirports | airports.csv + runways.csv | davidmegginson.github.io/ourairports-data | ORD, MEM | Current (downloaded Sep 2026) |
