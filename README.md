# CLEARS - Climate-Linked Evaluation and Analytics for Resilience in Airport Systems

Predicts weather-driven flight disruptions at Chicago O'Hare (ORD) and Memphis (MEM), explains them (SHAP), estimates delay severity, scores airport resilience and supports what-if scenarios. This repository holds the **data pipeline (Person 1)** and the **models and explainability (Person 2)**; the dashboard (Person 3) and final report/slides (Person 4) are still to be built on top of it.

**Status:** complete and verified for Person 1 and Person 2; not yet reviewed by Person 1 as the team handoff rule requires. Data version: **v3** = rebuilt from the raw NOAA and BTS files (the original delivery, v1, is in `archive/`).

## Directory map
```
README.md                    this file
requirements.txt             pinned packages (Python 3.11)
.gitignore

data/
  airport_context.csv        runway count, hub size, coordinates (OurAirports)
  raw/                       NOAA weather + BTS flight files (462 MB; only needed to re-run data_prep; git-ignored)
    weather/  flights/
  processed/
    clean_data.csv           merged flights + weather (data_prep.py output)        [git-ignored, regenerable]
    features_final.csv       feature matrix, 358,594 rows x 38 columns             [git-ignored, regenerable]
    airport_resilience_inputs.csv, resilience_scores.csv      per-airport inputs and the 0-100 index
  models/                    disruption_model.pkl, severity_model.pkl, scenario_model.pkl,
                             metrics.json, classification_report.txt              [pkl git-ignored]

src/                         all code
  config.py                  feature list, split dates, risk bands, paths
  data_prep.py               raw NOAA + BTS -> clean_data.csv   (UTC -> local time, backward weather match, labels)
  features.py                clean data -> 38-column feature matrix (leak-free rolling features)
  resilience.py              per-airport inputs + resilience index -> resilience_scores.csv
  model_train.py             trains the disruption, severity and weather-only models, writes metrics.json
  predict.py                 predict_disruption(), predict_severity(), predict_weather_risk(), load_model()
  explain.py                 explain_prediction(), get_global_importance(), scenario_risk(), SHAP plot
  verify_models.py           final verification checks -> docs/model_verification.md
  make_model_report.py       generates docs/model_report.md from metrics.json
tests/                       10 unit tests (features, leakage, data prep)
notebooks/                   01_eda.ipynb, 02_baseline_model.ipynb (both run end to end)
tools/                       reconcile_with_original.py - reproduces the evidence about the original delivery
assets/                      eda_key_chart.png, shap_summary.png, shap_global_importance.csv

docs/
  feature_dictionary.md      every column, ranges, which columns may be model inputs
  kpi_definitions.md         dashboard KPIs
  data_quality_report.md     13 checks with evidence
  data_sources.md            sources, URLs, file and row counts
  model_report.md            model evaluation (generated)
  model_verification.md      verification log (generated)
  reference/                 team task guide, project report (PDF), project overview (docx)

archive/                     superseded material, kept for traceability - safe to delete
  original_delivery/         the files delivered before the audit (v1), incl. the original features_final.csv
  v2_intermediate/           intermediate results (original data with rolling features rebuilt only)
```


## Results 
| | |
|---|---|
| Disruption model (XGBoost, calibrated) | **F1 0.365** (precision 0.323, recall 0.420), ROC-AUC 0.662, PR-AUC 0.336 |
| Logistic regression baseline | F1 0.351, ROC-AUC 0.652 |
| Project target F1 >= 0.70 | **Not met.** Even an easier date-grouped cross-validation reaches only F1 0.40 / ROC-AUC 0.71 |
| Severity model | macro-F1 0.290 (weak) |
| Resilience index | ORD 50 / MEM 50 (both Moderate); the earlier "ORD 30 / MEM 70" was an artefact of the two-airport scaling plus the data problems |
| Tests / verification | 10 unit tests pass; 15 of 16 verification checks pass (the failing one is the guide's flag-only "high risk" example, which scores 37%) |

Details: `docs/model_report.md`, `docs/model_verification.md`.

## Quick start
```bash
python -m venv clears_env
clears_env\Scripts\activate          # Windows;  source clears_env/bin/activate on Mac/Linux
pip install -r requirements.txt
python -m pytest tests -q
```
Retrain and regenerate every artefact from the current data (a few minutes):
```bash
python src/model_train.py
python src/resilience.py
python src/explain.py
python src/verify_models.py
python src/make_model_report.py
```
Rebuild the dataset from the raw files first, if needed (about 3 minutes; `data/raw/` must exist; see `docs/data_sources.md`):
```bash
python -c "import sys; sys.path.insert(0,'src'); import data_prep as d, features as f; df=d.run_pipeline(); out=f.build_features(df); out['timestamp']=out['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'); out['flight_date']=out['flight_date'].dt.strftime('%Y-%m-%d'); out.to_csv('data/processed/features_final.csv', index=False)"
```
`run_pipeline(cancelled_policy='drop')` removes cancelled flights (default); `'disrupt'` keeps them labelled as disrupted. Set `CLEARS_DATA=<path>` to train on a different feature file. The virtual environment used during development lives outside OneDrive (`C:\Users\shrey\.venvs\clears_env`) because a venv inside OneDrive syncs tens of thousands of files.



## Decisions and open items
1. **Cancelled flights:** 15,875 are removed (4.2% of departures). Counting them as disrupted would raise the disruption rate from 17.8% to about 21.3%. The team must choose, then regenerate and retrain.
2. **Risk bands:** with the calibrated probability 91.5% of test flights are "Low" (< 0.30) and 1.7% "High" (> 0.65); the KPI doc (0.30 / 0.65) and the project report (0.50 / 0.80) also disagree. Pick one set.
3. **Resilience index:** meaningless with two airports (50 / 50). It needs >= 4 airports or fixed reference ranges.
4. **F1 >= 0.70** is not reachable with these inputs (weather, time, short delay history); report the honest numbers.
5. **Scenario simulator:** `scenario_risk()` uses the full model (base value = dashboard gauge). Combined extreme overrides go beyond the data (about 70 historical flights) and are an upper bound. `mode='weather'` gives an optional weather-only view.
6. **Coverage:** four months per year, two airports; the test set has no winter month; in the test months MEM is more disrupted than ORD (the reverse of the full period).
7. **Project report PDF (Person 4):** the abstract's "F1 exceeding 0.70" and "substantial improvement over logistic regression", the claim that SHAP confirmed `visibility_risk` / `storm_flag` / `wind_category` as leading predictors (they contribute under 0.1% each), the recommendation bands, the academic year, the citation mismatches and the blank student-details page need correcting. The PDF itself was not changed.
8. **Approval:** Person 1 should review `src/` and the dataset before the numbers go into the report or slides.
