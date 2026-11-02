# CLEARS - Data Sources (T-1.1)

**Downloaded:** 2 November 2026 | **Study airports:** Chicago O'Hare (IATA ORD, ICAO KORD), Memphis International (IATA MEM, ICAO KMEM) | **Period:** January, April, July and October of 2019, 2020, 2021 and 2022 (492 days with data per airport)

| # | Source | What | URL pattern | Files | Rows downloaded |
|---|---|---|---|---|---|
| 1 | NOAA Integrated Surface Database (ISD), "Global Hourly", public AWS copy of NCEI data | Hourly + special weather observations (wind, visibility, temperature, dewpoint, precipitation) for the airport stations; **times are UTC** | `https://noaa-global-hourly-pds.s3.amazonaws.com/<year>/<station>.csv` | 8 CSV files: stations 72530094846 (KORD) and 72334013893 (KMEM) x 2019-2022, 66.5 MB | KORD 51,636; KMEM 58,642 observations (all months of each year; only observations before flights are used) |
| 2 | US Bureau of Transportation Statistics (BTS), Reporting Carrier On-Time Performance | Flight-level scheduled/actual departure and arrival delays, cancellations and delay causes for all US flights; **times are local** | `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_<year>_<month>.zip` | 16 zip files (months 1, 4, 7, 10 x 2019-2022), 417.2 MB | 8,338,532 US flights in total; **374,469 departures from ORD or MEM** (ORD 347,046; MEM 27,423), of which 15,875 cancelled flights have no departure delay and are removed, leaving 358,594 |
| 3 | OurAirports (`airports.csv`, `runways.csv`) | Airport metadata: runway count, hub class, coordinates, elevation | `https://davidmegginson.github.io/ourairports-data/` | summarised in `airport_context.csv` (from the original delivery; not re-downloaded) | 2 airports |

**Where the raw files are:** `data/raw/weather/` (`<station>_<year>.csv`) and `data/raw/flights/` (`ontime_<year>_<month>.zip`). They are not needed to run the models and can be deleted (462 MB); they are needed to re-run `src/data_prep.py`. Do not commit them to git.

**Processing:** `src/data_prep.py` (load, UTC -> local time, backward match, label) -> `data/processed/clean_data.csv` -> `src/features.py` -> `data/processed/features_final.csv` (358,594 rows x 38 columns). Reconciliation with the original `archive/original_delivery/features_final.csv` (which contained 12,135 duplicated flights and UTC-labelled weather) is in `data_quality_report.md`, Checks 8 and 13.

**Known coverage limits:** only four months per year; two airports; one station per airport; BTS reporting carriers only (not all general aviation); cancelled flights excluded by default.

**Licensing/attribution:** NOAA ISD and BTS data are US public-domain government data; OurAirports data is public domain.
