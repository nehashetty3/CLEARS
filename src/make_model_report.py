"""Generate model_report.md (T-2.7) from data/models/metrics.json so every number matches the code output.

    python src/make_model_report.py          # after model_train.py and explain.py have been run
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import sklearn
import xgboost

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

M = json.loads((C.MODEL_DIR / 'metrics.json').read_text(encoding='utf-8'))
D, T, TR = M['data'], M['test_tuned_threshold'], M['test_raw_0.5']
BASE, ABL, SEV, WX = M['baselines'], M['baselines']['ablation'], M['severity'], M['weather_model']
LR, DIAG = BASE['logistic_regression'], M['diagnostics']
RES = pd.read_csv(C.ROOT / 'data' / 'processed' / 'resilience_scores.csv').merge(
    pd.read_csv(C.ROOT / 'data' / 'processed' / 'airport_resilience_inputs.csv'), on='airport_code', suffixes=('', '_in'))
IMP = pd.read_csv(C.ASSETS_DIR / 'shap_global_importance.csv', index_col=0).iloc[:, 0]
_d = pd.read_csv(C.DATA_PATH, usecols=['airport_code', 'visibility_km', 'precip_mm_hr', 'wind_speed_knots', 'storm_flag', 'disruption'])
_OBS_DEF = {
    'low visibility (0.5 km)': ('visibility < 1 km', _d['visibility_km'] < 1),
    'heavy rain (10 mm/h)': ('rain >= 7.5 mm/h', _d['precip_mm_hr'] >= 7.5),
    'strong wind (28 kn)': ('wind >= 25 kn', _d['wind_speed_knots'] >= 25),
    'all three together': ('visibility < 1 km and rain >= 2.5 mm/h', (_d['visibility_km'] < 1) & (_d['precip_mm_hr'] >= 2.5)),
    'clear day (16 km, dry, 5 kn)': ('visibility >= 10 km, dry, wind < 10 kn',
                                      (_d['visibility_km'] >= 10) & (_d['precip_mm_hr'] == 0) & (_d['wind_speed_knots'] < 10)),
}
OBS = {k: (label, int(m.sum()), float(_d.loc[m, 'disruption'].mean())) for k, (label, m) in _OBS_DEF.items()}
STORM_SHARE = float(_d['storm_flag'].mean())
VIS3_RATE, VIS3_REST = float(_d.loc[_d['visibility_km'] < 3, 'disruption'].mean()), float(_d.loc[_d['visibility_km'] >= 3, 'disruption'].mean())
_ARCH = C.ROOT / 'archive' / 'v2_intermediate' / 'resilience_scores.csv'
OLD_RES = pd.read_csv(_ARCH).set_index('airport_code')['resilience_index'].to_dict() if _ARCH.exists() else {}


def f(x, n=3):
    return f'{x:.{n}f}'


def pc(x, n=1):
    return f'{100 * x:.{n}f}%'


MEANING = {
    'delay_last_3h': ('Share of flights disrupted in the 3-hour window ending two hours before this flight',
                      'Delays cascade: a disrupted airport tends to stay disrupted. The strongest and most intuitive signal.'),
    'delay_last_1h': ('Disruption share in the single hour ending two hours before', 'Same cascade effect, shorter memory.'),
    'delay_last_24h': ('Disruption share over the previous 24 hours', 'Captures a bad day (weather system, ground stop) rather than a bad hour.'),
    'hour_of_day': ('Scheduled departure hour', 'Delays accumulate through the day, so late hours are riskier (about 8.5% at 05:00, 21-25% between 15:00 and 22:00).'),
    'congestion_level': ('Scheduled departures in the two previous hours', 'A busier airport has less slack; plausible, and it also proxies the time of day.'),
    'temperature_c': ('Air temperature', 'Acts mostly as a season proxy (hot July afternoons with thunderstorms and high traffic) rather than a direct cause.'),
    'humidity_pct': ('Relative humidity', 'Also season/convective-weather proxy; not a direct operational driver.'),
    'visibility_km': ('Horizontal visibility', 'Low visibility forces instrument procedures and spacing; sensible, but a small effect once delay history is known.'),
    'wind_speed_knots': ('Wind speed', 'Strong wind restricts runway use; sensible, small effect (max wind in the data is 35 kn).'),
    'runway_count': ('Number of runways (ORD 8, MEM 4)', 'Only an airport identifier here (two airports); do not read it as a capacity effect.'),
    'day_of_week': ('Day of week', 'Weekly traffic pattern (Sunday highest, Tuesday lowest).'),
    'precip_mm_hr': ('Precipitation intensity', 'Rain slows operations; very small effect in this data.'),
    'month': ('Month (1, 4, 7 or 10)', 'Seasonality; only four distinct values exist.'),
}
imp_share = IMP / IMP.sum()
top10 = ''.join(f"| {i + 1} | `{n}` | {f(IMP[n], 4)} | {pc(imp_share[n])} | "
                f"{MEANING.get(n, ('', ''))[0]} | {MEANING.get(n, ('', ''))[1]} |\n" for i, n in enumerate(IMP.index[:10]))

cm = T['confusion_matrix']
cm5 = TR['confusion_matrix']
sev_cm = SEV['confusion_matrix']
names = ['none', 'minor', 'moderate', 'severe']
sev_rows = ''.join(f"| {n} | {SEV['class_counts_test'][n]:,} | {f(SEV['report'][n]['precision'])} | "
                   f"{f(SEV['report'][n]['recall'])} | {f(SEV['report'][n]['f1-score'])} |\n" for n in names)
sev_cm_rows = ''.join(f"| true {names[i]} | " + ' | '.join(f'{v:,}' for v in sev_cm[i]) + ' |\n' for i in range(4))
abl_rows = ''.join(f"| {k.replace('_', ' ')} | {v['n_features']} | {f(v['roc_auc'])} | {f(v['pr_auc'])} | {f(v['f1_1'])} |\n"
                   for k, v in ABL.items())
airport_rows = ''.join(f"| {a} | {v['n']:,} | {pc(v['positive_rate'])} | {f(v['roc_auc'])} | {f(v['pr_auc'])} | "
                       f"{f(v['precision_1'])} | {f(v['recall_1'])} | {f(v['f1_1'])} |\n" for a, v in M['test_by_airport'].items())
month_names = {'4': 'April 2022', '7': 'July 2022', '10': 'October 2022'}
month_rows = ''.join(f"| {month_names.get(m, m)} | {v['n']:,} | {pc(v['positive_rate'])} | {f(v['roc_auc'])} | {f(v['pr_auc'])} | "
                     f"{f(v['f1_1'])} |\n" for m, v in M['test_by_month'].items())
band_hi, band_lo = M['band_observed_rate']['high'], M['band_observed_rate']['low']
share = M['risk_band_share_test']
ST = M['scenario_table']
sc_rows = ''.join(
    f"| {k} (`{', '.join(f'{a}={b}' for a, b in v['override'].items())}`) | {OBS[k][0]}: n = {OBS[k][1]:,}, {pc(OBS[k][2])} disrupted | "
    f"{pc(v['full_prob'])} ({v['full_delta'] * 100:+.0f} pts) | {pc(v['weather_prob'])} ({v['weather_delta'] * 100:+.0f} pts) |\n"
    for k, v in ST.items() if k != 'baseline')
res_rows = ''.join(f"| {r.airport_code} | {f(r.disruption_rate / 100, 4)} | {f(r.avg_recovery_time, 1)} | {f(r.weather_sensitivity, 4)} | "
                   f"{f(r.congestion_sensitivity, 4)} | **{r.resilience_index:.1f}** | {r.tier} |\n" for r in RES.itertuples())
target_met = T['f1_1'] >= 0.70
bp = M['best_params']

report = f"""# CLEARS - Model Evaluation Report
**Owner:** Person 2 - ML & Explainability Lead | **Generated by:** `src/make_model_report.py` from `data/models/metrics.json`
**Data:** `{D['file']}` (see `docs/data_quality_report.md`) | **Library versions:** XGBoost {xgboost.__version__}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}

> **Headline.** On a time-ordered, out-of-sample test set the disruption model reaches **F1 = {f(T['f1_1'])}** for the disruption class
> (precision {f(T['precision_1'])}, recall {f(T['recall_1'])}) and **ROC-AUC = {f(T['roc_auc'])}**. The project target of F1 >= 0.70 is
> **{'met' if target_met else 'NOT met'}**. A date-grouped cross-validation, which is easier than a time-ordered test, reaches only
> F1 = {f(DIAG['date_grouped_cv']['f1_1_at_base_rate_threshold'])} and ROC-AUC = {f(DIAG['date_grouped_cv']['roc_auc'])}, so the gap to 0.70 is not caused by the
> split. The data were regenerated from the raw NOAA and BTS files (weather in correct local time, each flight once); they still cover only four months a year and two airports.

## 1. Model architecture
| Item | Value |
|---|---|
| Disruption model | `XGBClassifier`, binary, `scale_pos_weight` = {f(M['scale_pos_weight'], 2)} (negatives / positives in training), histogram trees |
| Selected hyper-parameters | max_depth {bp['max_depth']}, learning_rate {bp['learning_rate']}, {bp['best_iteration'] + 1} trees (early stopping, 30 rounds, validation PR-AUC), subsample 0.8, colsample 0.8, min_child_weight 5 |
| Search | guide grid (depth 3/5/7 x learning rate 0.05/0.1), n_estimators by early stopping; chosen on the **validation** slice by PR-AUC. All six configurations scored within 0.002 PR-AUC of each other |
| Probability calibration | isotonic regression fitted on the validation slice, so `probability` is a real probability (the raw `scale_pos_weight` output is inflated: Brier {f(M['test_brier_raw'], 4)} raw vs {f(M['test_brier_calibrated'], 4)} calibrated vs {f(M['test_brier_constant_baseline'], 4)} for a constant) |
| Decision threshold | {f(T['threshold'])} on the calibrated probability, the F1-optimal value on the validation slice |
| Inputs ({len(D['features'])}) | {', '.join('`' + x + '`' for x in D['features'])} |
| Excluded | every label-derived column (`dep_delay`, `arr_delay`, `delay_minutes`, `cancelled`, `*_delay`), identifiers and constants - see `docs/feature_dictionary.md` Section 9 |
| Severity model | `XGBClassifier(objective='multi:softprob', num_class=4)`, balanced sample weights, depth {SEV['best_params']['max_depth']}, learning rate {SEV['best_params']['learning_rate']}, {SEV['best_params']['best_iteration'] + 1} trees |
| Scenario (what-if) model | XGBoost on weather + time + airport only (no delay history), monotone constraints, depth {WX['best_params']['max_depth']}, learning rate {WX['best_params']['learning_rate']}, {WX['best_params']['best_iteration'] + 1} trees, isotonic-calibrated |
| Saved files | `data/models/disruption_model.pkl`, `severity_model.pkl`, `scenario_model.pkl` (dicts holding the model, calibrator, threshold, feature list and training medians; load with the same library versions) |

## 2. Training data
| Slice | Dates | Rows | Disruption rate |
|---|---|---|---|
| Train | {D['train_range'][0]} to {D['train_range'][1]} | {D['train_rows']:,} | {pc(D['train_rate'])} |
| Validation (early stopping, calibration, threshold) | {D['val_range'][0]} to {D['val_range'][1]} | {D['val_rows']:,} | {pc(D['val_rate'])} |
| **Test (used once)** | {D['test_range'][0]} to {D['test_range'][1]} | {D['test_rows']:,} | {pc(D['test_rate'])} |

Airports ORD and MEM, only January/April/July/October of each year (492 days). Split on whole dates, strictly time-ordered; the test set
contains **no winter month** (April, July and October 2022 only). Class ratio about {f(M['scale_pos_weight'], 1)} : 1.

## 3. Test metrics - disruption model (threshold {f(T['threshold'])}, chosen on validation)
| Metric | Value |
|---|---|
| Accuracy | {f(T['accuracy'])} |
| Precision / recall / **F1** (disruption class) | {f(T['precision_1'])} / {f(T['recall_1'])} / **{f(T['f1_1'])}** |
| Precision / recall / F1 (no-disruption class) | {f(cm['tn'] / (cm['tn'] + cm['fn']))} / {f(cm['tn'] / (cm['tn'] + cm['fp']))} / {f(T['f1_0'])} |
| ROC-AUC | {f(T['roc_auc'])} |
| PR-AUC (a random model scores {f(D['test_rate'])}) | {f(T['pr_auc'])} |

Full `classification_report` output for both models: `data/models/classification_report.txt`.

Confusion matrix (test, rows = truth):

| | predicted no disruption | predicted disruption |
|---|---|---|
| **actual no disruption** | {cm['tn']:,} | {cm['fp']:,} |
| **actual disruption** | {cm['fn']:,} | {cm['tp']:,} |

At the uncalibrated 0.5 cut-off the same model gives precision {f(TR['precision_1'])}, recall {f(TR['recall_1'])}, F1 {f(TR['f1_1'])}
(TP {cm5['tp']:,}, FP {cm5['fp']:,}, FN {cm5['fn']:,}, TN {cm5['tn']:,}); the validation-chosen threshold is not better on the test months because the
disruption regime differs (validation {pc(D['val_rate'])} vs test {pc(D['test_rate'])}).

**By airport**

| Airport | Rows | Disruption rate | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
{airport_rows}
**By month** (October 2022 is a low-disruption month and scores far worse - the threshold fitted on a winter validation slice is wrong for it)

| Month | Rows | Disruption rate | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|---|
{month_rows}
## 4. Comparison with baselines
| Model | ROC-AUC | PR-AUC | F1 (disruption) |
|---|---|---|---|
| Flag every flight | 0.500 | {f(BASE['base_rate'])} | {f(2 * BASE['base_rate'] / (1 + BASE['base_rate']))} |
| Delay-history signal alone (`delay_last_3h` as a score) | {f(BASE['rolling_rate_only']['roc_auc'])} | {f(BASE['rolling_rate_only']['pr_auc'])} | - |
| Logistic regression (same 20 inputs, same protocol) | {f(LR['roc_auc'])} | {f(LR['pr_auc'])} | {f(LR['f1_1'])} |
| **XGBoost (this model)** | **{f(T['roc_auc'])}** | **{f(T['pr_auc'])}** | **{f(T['f1_1'])}** |

XGBoost improves on logistic regression by {f(T['f1_1'] - LR['f1_1'])} F1 ({pc((T['f1_1'] - LR['f1_1']) / LR['f1_1'])} relative) and {f(T['roc_auc'] - LR['roc_auc'])} ROC-AUC. That is a small
gain, not the "substantial improvement" the project report expects. A weather-only XGBoost scores ROC-AUC {f(ABL['weather_only']['roc_auc'])} and
F1 {f(ABL['weather_only']['f1_1'])}; flagging every flight scores F1 {f(2 * BASE['base_rate'] / (1 + BASE['base_rate']))}.

**Which inputs matter (retrained on feature groups, same protocol)**

| Feature group | Inputs | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|
{abl_rows}
Delay history is worth about +{f(ABL['full']['roc_auc'] - ABL['no_history_features']['roc_auc'])} ROC-AUC on top of weather + time; weather alone scores ROC-AUC {f(ABL['weather_only']['roc_auc'])}, time alone {f(ABL['time_only']['roc_auc'])}.

**Is the low score a split artefact?** No.

| Evaluation | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Time-ordered test (this report) | {f(T['roc_auc'])} | {f(T['pr_auc'])} | {f(T['f1_1'])} |
| Train before 2022, test 2022 (whole year) | {f(DIAG['train_before_2022_test_2022']['roc_auc'])} | {f(DIAG['train_before_2022_test_2022']['pr_auc'])} | - |
| Train before 2021, test 2021 | {f(DIAG['train_before_2021_test_2021']['roc_auc'])} | {f(DIAG['train_before_2021_test_2021']['pr_auc'])} | - |
| Date-grouped 5-fold CV (optimistic, not time-ordered) | {f(DIAG['date_grouped_cv']['roc_auc'])} | {f(DIAG['date_grouped_cv']['pr_auc'])} | {f(DIAG['date_grouped_cv']['f1_1_at_base_rate_threshold'])} |

## 5. Top 10 SHAP features (mean |SHAP| on the test set, log-odds of the raw model)
SHAP values are exact TreeSHAP from XGBoost (`pred_contribs`), identical to `shap.TreeExplainer`; they sum to the model output (checked in `model_verification.md`). Plot: `assets/shap_summary.png`.

| # | Feature | Mean abs SHAP | Share | What it measures | Does it make sense? |
|---|---|---|---|---|---|
{top10}
**The weather flags the project report names as leading predictors contribute almost nothing:** `visibility_risk` {pc(imp_share['visibility_risk'], 2)}, `storm_flag` {pc(imp_share['storm_flag'], 2)},
`wind_category_enc` {pc(imp_share['wind_category_enc'], 2)} of total importance; only `delay_last_3h` ({pc(imp_share['delay_last_3h'])}) is a leading predictor. The flags are rare (storm flag {pc(STORM_SHARE, 2)} of flights) and the model
reads the same information from the raw weather columns instead (`visibility_km`, `precip_mm_hr`, `wind_speed_knots` together {pc(imp_share[['visibility_km', 'precip_mm_hr', 'wind_speed_knots']].sum())} of total importance). Weather clearly matters in the raw data:
flights with visibility < 3 km are disrupted {pc(VIS3_RATE)} of the time vs {pc(VIS3_REST)} otherwise, but those flights are few and delay history explains most of the rest.

## 6. Delay severity model
Classes: none (not disrupted), minor (15-45 min), moderate (45-120 min), severe (> 120 min). Balanced class weights.

| Class | Test flights | Precision | Recall | F1 |
|---|---|---|---|---|
{sev_rows}
Accuracy {f(SEV['accuracy'])}, **macro-F1 {f(SEV['macro_f1'])}** (always predicting "none" would score accuracy {f(SEV['baseline_always_none']['accuracy'])} but macro-F1 about 0.2). Confusion matrix (rows = truth, columns = prediction):

| | none | minor | moderate | severe |
|---|---|---|---|---|
{sev_cm_rows}
The severity classes are only weakly separable from the available inputs (F1 {f(min(SEV['report'][n]['f1-score'] for n in names[1:]))}-{f(max(SEV['report'][n]['f1-score'] for n in names[1:]))} for the three delay classes). Use the severity output as a coarse indication and show the class probabilities, not the label alone.

## 7. Probability bands and the dashboard
With the calibrated probability, KPI 4 bands (Low < {C.RISK_BANDS[0]}, Medium {C.RISK_BANDS[0]}-{C.RISK_BANDS[1]}, High > {C.RISK_BANDS[1]}) split the test flights as:

| Band | Share of flights | Observed disruption rate |
|---|---|---|
| Low | {pc(share[list(share)[0]])} | {pc(band_lo['observed_disruption_rate'])} |
| Medium | {pc(share['medium'])} | - |
| High | {pc(share[list(share)[2]])} | {pc(band_hi['observed_disruption_rate'])} |

The probabilities are well calibrated at the top (High-band flights are disrupted {pc(band_hi['observed_disruption_rate'], 0)} of the time), but with an {pc(D['test_rate'], 0)} base rate {pc(share[list(share)[0]], 0)} of flights are "Low" and only {pc(share[list(share)[2]], 1)} are "High", so the gauge will
rarely leave green. The team should choose bands with this in mind (for example by relative risk: base rate x 1.5, x 2).

## 8. Scenario simulator
`scenario_risk(base, overrides)` re-scores a flight with changed weather. By default it uses the **full model**, so `base_probability` equals the dashboard gauge. Raw weather overrides are
turned into the flags and encodings the model reads (a scenario row cannot contradict itself); the flight's delay-history inputs stay as they are, so this answers "what if the weather were different and nothing else changed".
`mode='weather'` uses a second model without delay history (monotone: worse weather never lowers the risk; test ROC-AUC {f(WX['test']['roc_auc'])}), i.e. "the risk under these conditions before delays have accumulated".

Mean predicted probability on {5000:,} test flights; "historical" is the observed disruption rate of all flights in the data under comparable conditions.

| Scenario | Historical flights in comparable conditions | Full model: probability (change) | Weather-only model: probability (change) |
|---|---|---|---|
| no override | all flights: {pc(D['test_rate'])} (test) | {pc(ST['baseline']['full_prob'])} | {pc(ST['baseline']['weather_prob'])} |
{sc_rows}
Single conditions land in the right range (heavy rain: {pc(OBS['heavy rain (10 mm/h)'][2], 0)} observed; the models say about {pc(ST['heavy rain (10 mm/h)']['full_prob'], 0)} / {pc(ST['heavy rain (10 mm/h)']['weather_prob'], 0)}).
**The combined "all three" scenario goes beyond the data**: only {OBS['all three together'][1]} historical flights had visibility < 1 km together with rain >= 2.5 mm/h ({pc(OBS['all three together'][2], 0)} disrupted), yet the models extrapolate far higher.
Treat combined extremes as an upper bound, not a forecast. Wind above the training maximum (35 kn) is treated like ~35 kn.

*History:* on the earlier data version, whose weather was 5-6 hours out of sync with departure times, the full model barely reacted to weather overrides (about +4 points for the same severe override); the weather-only model was added to fix that.
The reaction came back once the weather was aligned, which is why the full model is now the default and the weather-only model is optional.

## 9. Resilience index methodology
Inputs per airport (from `data/processed/airport_resilience_inputs.csv`): disruption rate, average delay of disrupted flights ("recovery time"), weather sensitivity = correlation(`storm_flag`, `disruption`), congestion sensitivity = correlation(`congestion_level`, `disruption`).
Formula: min-max normalise each across airports; vulnerability = 0.35 x disruption rate + 0.30 x recovery time + 0.20 x weather sensitivity + 0.15 x congestion sensitivity; index = 100 - 100 x vulnerability; tiers <= 40 Vulnerable, 40-70 Moderate, > 70 Resilient.
Weights are a design choice: frequency (0.35) and severity of delays (0.30) are what operations experience most directly; the two correlation terms (0.20 / 0.15) describe sensitivity rather than outcomes and are weaker, noisier measures. The literature does not fix the weights.

| Airport | Disruption rate | Avg delay of disrupted flights (min) | Weather sens. | Congestion sens. | Resilience index | Tier |
|---|---|---|---|---|---|---|
{res_rows}
**Limitation that matters:** with two airports min-max scaling maps every dimension to exactly 0 or 1, so the scores only say which airport is higher on each dimension, not by how much. This makes the index unstable:
on the earlier data version (weather misaligned, duplicated flights) it was {', '.join(f'{a} {v:.0f}' for a, v in sorted(OLD_RES.items(), reverse=True)) or 'n/a'}; on the corrected data it is {', '.join(f'{r.airport_code} {r.resilience_index:.0f}' for r in RES.sort_values('airport_code', ascending=False).itertuples())}.
The change comes from one reversed ordering (weather sensitivity: MEM {f(RES.set_index('airport_code').loc['MEM', 'weather_sensitivity'], 4)} vs ORD {f(RES.set_index('airport_code').loc['ORD', 'weather_sensitivity'], 4)} after the correction), not from any real change in the airports.
In the test months MEM's disruption rate ({pc(M['test_by_airport']['MEM']['positive_rate'])}) is higher than ORD's ({pc(M['test_by_airport']['ORD']['positive_rate'])}), the opposite of the full-period ranking. Do not present the index as more than a two-airport ranking; it needs >= 4 airports or fixed reference ranges.

## 10. Where everything is (task guide file map)
| Task guide path | Location in this project | Note |
|---|---|---|
| `data/models/disruption_model.pkl` | `data/models/disruption_model.pkl` | a dict {{model, calibrator, threshold, features, medians}}; use `predict.load_model()` |
| `data/models/severity_model.pkl` | `data/models/severity_model.pkl` | same structure, 4 classes |
| (extra) | `data/models/scenario_model.pkl` | optional weather-only model for `scenario_risk(mode='weather')` |
| `data/processed/resilience_scores.csv` | `data/processed/resilience_scores.csv` | airport_code, resilience_index, tier (+ disruption_rate, lat, lon for the map) |
| `src/model_train.py`, `predict.py`, `explain.py`, `resilience.py` | `src/...` | |
| `notebooks/02_baseline_model.ipynb` | `notebooks/02_baseline_model.ipynb` | |
| `assets/shap_summary.png` | `assets/shap_summary.png` | |
| `docs/model_report.md`, `docs/model_verification.md` | `docs/model_report.md`, `docs/model_verification.md` | generated by `src/make_model_report.py` and `src/verify_models.py` |

## 11. Known limitations
1. **Weather alignment was fixed in this version** (NOAA UTC times converted to local time; the earlier data was 5-6 hours out of sync - data quality report, Check 8). What remains: one weather station per airport, observations up to an hour old at departure time, and precipitation only from the first ISD precipitation group.
2. **Modest predictive power.** F1 {f(T['f1_1'])} / ROC-AUC {f(T['roc_auc'])}; even the optimistic date-grouped CV only reaches F1 {f(DIAG['date_grouped_cv']['f1_1_at_base_rate_threshold'])}. The inputs (weather, time, short delay history) do not determine disruption well; schedule, airline and route data would be needed.
3. **Small, unrepresentative coverage:** four months per year, two airports, no winter month in the test set, 2020 is an outlier year. 15,875 cancelled flights (4.2% of scheduled departures) have no departure delay and were removed, so disruption is understated.
4. **Regime shift:** thresholds tuned on the October-January validation slice do not transfer to October 2022 (F1 {f(M['test_by_month']['10']['f1_1'])} that month).
5. **Extreme and combined weather scenarios extrapolate** beyond the data (section 8).
6. **No real-time feed:** inputs are historical; delay-history features need a live feed in production.
7. **Severity model is weak** (macro-F1 {f(SEV['macro_f1'])}).
8. **Statements in the project report that this evidence does not support** (for Person 4 to correct): the abstract's "F1 score exceeding 0.70" and "substantial improvement over logistic regression"; Chapter 5's claim that SHAP confirmed `visibility_risk`, `storm_flag` and `wind_category` as leading predictors (they contribute under 0.1% each); the recommendation bands (0.50/0.80) which do not match the KPI bands used here.
"""
(C.DOCS_DIR / 'model_report.md').write_text(report, encoding='utf-8')
print('wrote', C.DOCS_DIR / 'model_report.md', len(report), 'chars')
