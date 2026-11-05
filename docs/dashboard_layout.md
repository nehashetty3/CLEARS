# CLEARS dashboard: layout and design (task T-3.1)

**Owner:** Person 3 · **App:** `app/streamlit_app.py` · **Run:** `streamlit run app/streamlit_app.py` (from the project root)

The dashboard describes **one airport-hour at a time** (the "hour analysed") inside a **date range** chosen in the sidebar. The hour is either the highest-risk hour of the range or one the user picks.

## Wireframe

```
+-------------------+--------------------------------------------------------------------------------+
| SIDEBAR           |  [logo] CLEARS - Airport Resilience Dashboard      [airport] [hour] [flights] [period]
|  Airport          |                                                                                |
|  Date range       |  +------------------+  +------------------+  +------------------+              |
|  Hour to analyse  |  | 1  RISK GAUGE    |  | 1  SEVERITY      |  | 1  RESILIENCE    |   above the  |
|  About            |  | 95%  HIGH        |  | Severe + bars    |  | 50  Moderate     |   fold       |
|                   |  +------------------+  +------------------+  +------------------+              |
|                   |  [ Disruption rate ][ Avg delay ][ Weather sens. ][ Peak premium ][ Flights ]  |
|                   |  ----------------------------------------------------------------------------  |
|                   |  Risk Overview | Scenario Simulator | Resilience Rankings | Recommendations | Model & Data
|                   |  +-----------+ +-----------------------------------------------+               |
|                   |  | Conditions| | 2  Trend: actual, 7-day, model, storm days    |  tab 1        |
|                   |  +-----------+ +-----------------------------------------------+               |
|                   |  +-------------------------------+ +------------------------------+            |
|                   |  | 3  SHAP: this hour / overall  | | When disruptions happen      |            |
|                   |  +-------------------------------+ +------------------------------+            |
+-------------------+--------------------------------------------------------------------------------+
```

The numbers 1 to 6 follow the panels listed in the task guide.

| Guide panel | Where it is | Source |
|---|---|---|
| Header row: airport, date range, risk badge | Sidebar controls + header chips + risk card | `dashboard_data.py`, `predict.risk_level` |
| 1. Risk meter / KPI scorecard (probability, severity, resilience) | Three cards above the fold | `predict_disruption`, `predict_severity`, `resilience_scores.csv` |
| 2. Weather trend chart | Risk Overview, wide card | `dashboard_data.daily_series` |
| 3. SHAP feature importance | Risk Overview, switch between "This hour" and "Overall" | `explain.explain_prediction`, `explain.get_global_importance` |
| 4. Resilience ranking table | Resilience Rankings tab (table, bar chart, map, four inputs) | `resilience_scores.csv`, `airport_resilience_inputs.csv` |
| 5. Scenario simulator | Scenario Simulator tab (sliders, presets, before/after, sensitivity curves) | `explain.scenario_risk`, `explain.apply_overrides` |
| 6. Recommendation card | Recommendations tab | `recommend.build_recommendations` |
| (added) Model and data transparency | Model & Data tab | `metrics.json`, `shap_global_importance.csv` |

## Answers to the sketch questions

- **Above the fold:** the three hero cards (risk, severity, resilience) and the KPI strip. A viewer sees the answer before they scroll.
- **Tabs vs scrolled:** the five tabs hold the detail panels; inside a tab the panels scroll.
- **Sidebar content:** airport, date range (test / validation / training / all / custom), which hour to analyse, a short data note.
- **Global vs per-panel filters:** airport, date range and the analysed hour are global. Per-panel controls: trend toggles, SHAP view, scenario sliders and model view, recommendation source.

## What the dashboard needs from the other roles

| From | What | Contract |
|---|---|---|
| Person 1 | `features_final.csv` (rebuilt into `dashboard_hourly.csv`), KPI definitions | one row per airport-hour, the 20 model inputs, flight and disruption counts |
| Person 2 | `predict_disruption`, `predict_severity`, `explain_prediction`, `get_global_importance`, `scenario_risk`, three `.pkl` models | signatures unchanged; `scenario_risk` also exposes `apply_overrides` |
| Person 4 | Screenshots and the demo script | `docs/demo_script.md` |

## Design decisions

- **Coloured but calm theme** (`app/theme.py`): a navy-to-blue brand gradient on the banner, sidebar and active tab; a softly tinted page; white cards with blue-tinted shadows; the three hero cards wash in the colour of their state (risk level, severity, resilience tier); each KPI tile carries one accent colour. Status colours (green / amber / red) are reserved for risk and always come with a text label and a dot, never colour alone; body text never takes a data colour.
- **Chart colours follow the data-visualisation method in the dataviz skill:** magnitude uses one blue ramp (heatmap, map); the two airports use blue and orange, a pair that passes the colour-blind and contrast checks (CVD delta-E 24.7, normal vision 33.6); SHAP uses red for "raises risk" and blue for "lowers risk". Bars are thin with a 4px rounded end, lines are 2px, gridlines are solid hairlines, and a legend is shown whenever there are two or more series.
- **Meters instead of gauges:** the risk and resilience values are a thin meter with band ticks, which reads faster and prints cleanly. Severity is a segmented bar; the risk-level mix is a segmented bar too.
- **Honesty built in:** the unseen-test badge, the calibration chart, an "upper bound" warning for extreme what-ifs, the two-airport warning on the resilience index and a limitations list. The model is a modest signal and the interface says so.
- **Risk bands** (30% / 65%) are read from `config.RISK_BANDS`, the same constant the model code uses. The team has not finalised them; a change there flows to the risk meter, the bands in the charts and the recommendations.
- **Performance:** one 16,000-row table, models and SHAP behind `st.cache_data`, `st.spinner("Running prediction...")` around inference.
- **Responsive:** the layout works from a phone to a wide desktop; the sidebar collapses on small screens.
