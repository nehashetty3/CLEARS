"""Final model verification (T-2.8). Runs every check from the task guide plus a few extra ones and writes
model_verification.md with the real inputs, outputs and PASS / FAIL.

    python src/verify_models.py

Pass criteria are the ones in the task guide (high-risk probability > 0.60, benign < 0.30, severity not
'none' for a high-risk input, ...). They are NOT adjusted to fit the model; a FAIL is reported as a FAIL.
"""
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

GUIDE_HIGH = {'wind_category': 'severe', 'visibility_risk': 1, 'storm_flag': 1, 'is_peak_hour': 1}
REALISTIC_STORM = {'wind_speed_knots': 28, 'visibility_km': 0.5, 'precip_mm_hr': 10, 'temperature_c': 20,
                   'humidity_pct': 95, 'hour_of_day': 18, 'day_of_week': 4, 'month': 7, 'delay_last_1h': 0.7,
                   'delay_last_3h': 0.65, 'delay_last_24h': 0.5, 'congestion_level': 110, 'runway_count': 8}
BENIGN = {'wind_speed_knots': 4, 'visibility_km': 16, 'precip_mm_hr': 0, 'temperature_c': 15, 'humidity_pct': 50,
          'hour_of_day': 10, 'day_of_week': 2, 'month': 4, 'delay_last_1h': 0.05, 'delay_last_3h': 0.05,
          'delay_last_24h': 0.1, 'congestion_level': 40, 'runway_count': 8}


def fresh_session(expr: str):
    """Evaluate `expr` in a brand-new Python process that only imports src/predict.py."""
    code = ("import sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, %r); "
            "import predict as P, explain as E; print('@@' + json.dumps(%s))" % (str(HERE), expr))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd='/')
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-500:])
    return json.loads(out.stdout.split('@@')[-1])


def short(d, nd=3):
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items() if k != 'probabilities'}


results = []


def record(name, inp, out, ok, note=''):
    results.append({'check': name, 'input': inp, 'output': out, 'status': 'PASS' if ok else 'FAIL', 'note': note})


# 1) fresh load + high risk (guide input, then a realistic stormy row)
r = fresh_session('{"guide": P.predict_disruption(%r), "storm": P.predict_disruption(%r)}' % (GUIDE_HIGH, REALISTIC_STORM))
record('1a Fresh-session load; guide high-risk input (probability > 0.60)', GUIDE_HIGH, short(r['guide']),
       r['guide']['probability'] > 0.60,
       'Partial input: delay-history features default to the training median, so the model sees no active delays.')
record('1b Fresh-session; realistic storm row with high delay history (probability > 0.60)', REALISTIC_STORM,
       short(r['storm']), r['storm']['probability'] > 0.60)

# 2) benign
r = fresh_session('P.predict_disruption(%r)' % BENIGN)
record('2 Benign conditions (probability < 0.30)', BENIGN, short(r), r['probability'] < 0.30)

# 3) severity consistency
sev_storm = fresh_session('{"p": P.predict_disruption(%r), "s": P.predict_severity(%r)}' % (REALISTIC_STORM, REALISTIC_STORM))
record("3a Severity for the high-risk storm row is 'moderate' or 'severe', not 'none'", REALISTIC_STORM,
       {'probability': round(sev_storm['p']['probability'], 3), 'severity': sev_storm['s']['severity'],
        'probabilities': {k: round(v, 3) for k, v in sev_storm['s']['probabilities'].items()}},
       sev_storm['s']['severity'] in ('moderate', 'severe'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import model_train as T  # noqa: E402
import predict as P  # noqa: E402
import explain as E  # noqa: E402
import xgboost as xgb  # noqa: E402

df = T.load_data()
_, _, test = T.split(df)
sample = test.sample(8000, random_state=0)
sev_b, dis_b = P._severity_bundle(), P._disruption_bundle()
cal = dis_b['calibrator'].predict(dis_b['model'].predict_proba(sample[C.FEATURES])[:, 1])
sev_pred = sev_b['model'].predict_proba(sample[C.FEATURES]).argmax(axis=1)
top = cal >= np.quantile(cal, 0.95)
share_none_top = float((sev_pred[top] == 0).mean())
share_none_all = float((sev_pred == 0).mean())
record("3b Across 8,000 test flights, the 5% highest-risk flights are rarely predicted severity 'none'",
       'top 5% by probability', {'share predicted none (top 5%)': round(share_none_top, 3),
                                 'share predicted none (all)': round(share_none_all, 3)},
       share_none_top < share_none_all, 'Compared with the overall share; high probability should shift towards worse classes.')

# 4) SHAP consistency
doc = (C.DOCS_DIR / 'feature_dictionary.md').read_text(encoding='utf-8')
dictionary_cols = set(re.findall(r'`([a-z_0-9]+)`', doc))
sh = E.explain_prediction_detail(dis_b, REALISTIC_STORM)
X = P.build_row(REALISTIC_STORM)
margin = float(dis_b['model'].get_booster().predict(xgb.DMatrix(X), output_margin=True)[0])
additive = abs(sum(sh['shap'].values()) + sh['base_value'] - margin) < 1e-4
names_ok = set(sh['shap']) == set(C.FEATURES) and set(sh['shap']) <= dictionary_cols
record('4 explain_prediction returns exactly the model features, all named in feature_dictionary.md, and SHAP values sum to the model output',
       REALISTIC_STORM, {'n_features': len(sh['shap']), 'names_in_dictionary': names_ok,
                         'sum(shap)+base - margin': round(sum(sh['shap'].values()) + sh['base_value'] - margin, 7)},
       names_ok and additive)

# 5) scenario
benign_full = {**BENIGN}
sw = E.scenario_risk(benign_full, {'visibility_km': 0.5, 'precip_mm_hr': 10, 'wind_speed_knots': 28})      # default = full model
sf = E.scenario_risk(benign_full, {'visibility_km': 0.5, 'precip_mm_hr': 10, 'wind_speed_knots': 28}, mode='weather')
sg = E.scenario_risk(benign_full, {'visibility_risk': 1, 'rain_severity': 'heavy', 'wind_category': 'severe'})
sh = E.scenario_risk(benign_full, {'precip_mm_hr': 10})
record('5a scenario_risk (default = full model, base value equals the gauge), severe raw-weather override: scenario > base',
       'visibility 0.5 km, 10 mm/h, 28 kn', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sw.items()},
       sw['scenario_probability'] > sw['base_probability'] and sw['mode'] == 'full',
       'Combined extremes go beyond the historical data (about 70 flights had visibility < 1 km with rain >= 2.5 mm/h); treat as an upper bound.')
record("5b scenario_risk with the dashboard guide's flag overrides (visibility_risk, rain_severity, wind_category)", 'flags',
       {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sg.items()}, sg['scenario_probability'] > sg['base_probability'])
record("5c scenario_risk(mode='weather'), the weather + time model without delay history: scenario > base", 'visibility 0.5 km, 10 mm/h, 28 kn',
       {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sf.items()}, sf['scenario_probability'] > sf['base_probability'])
record('5d Single-condition override (10 mm/h rain only) moves the probability by a plausible amount (observed disruption rate in heavy-rain flights is ~46% vs ~16% on clear days)',
       'precip 10 mm/h', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sh.items()},
       0.05 < sh['delta'] < 0.60)

# extra checks
bad = sorted(set(C.FEATURES) & set(C.BANNED_FEATURES))
record('6 No label-leaking column among the model inputs', C.FEATURES, {'banned columns used': bad}, not bad)
grid = {'visibility_km': [16, 8, 3, 1, 0.5], 'wind_speed_knots': [0, 10, 20, 30], 'precip_mm_hr': [0, 1, 3, 8, 15]}
mono = {}
for var, vals in grid.items():
    p = [P.predict_weather_risk({**BENIGN, var: v})['probability'] for v in vals]
    inc = all(b >= a - 1e-12 for a, b in zip(p, p[1:]))
    mono[var] = {'values': vals, 'probability': [round(x, 3) for x in p], 'non_decreasing_with_worse_weather': inc}
record('7 Scenario (weather) model: worse weather never lowers risk (visibility down, wind up, rain up)', grid,
       {k: v['non_decreasing_with_worse_weather'] for k, v in mono.items()}, all(v['non_decreasing_with_worse_weather'] for v in mono.values()))
rng_ok = all(0 <= P.predict_disruption(d)['probability'] <= 1 for d in (GUIDE_HIGH, REALISTIC_STORM, BENIGN, {}))
record('8 Probabilities in [0, 1]; empty dict does not crash (defaults to training medians)', '{} and the inputs above',
       {'ok': rng_ok, 'empty_dict': short(P.predict_disruption({}))}, rng_ok)
big_wind = P.predict_weather_risk({**BENIGN, 'wind_speed_knots': 60})['probability']
at30 = P.predict_weather_risk({**BENIGN, 'wind_speed_knots': 35})['probability']
record('9 Wind beyond the training range (max 35.0 kn) is handled without error', 'wind 60 kn vs 35 kn',
       {'p(60 kn)': round(big_wind, 3), 'p(35 kn)': round(at30, 3)}, big_wind >= at30 - 1e-12,
       'Trees extrapolate flat: 60 kn gives the same answer as ~35 kn because no such wind exists in the data.')

# --- plan items added after the first audit
scores_path = ROOT / 'data' / 'processed' / 'resilience_scores.csv'
if scores_path.exists():
    sc = pd.read_csv(scores_path)
    cols_ok = {'airport_code', 'resilience_index', 'tier'} <= set(sc.columns)
    range_ok = bool(sc['resilience_index'].between(0, 100).all())
    tier_ok = all(t == ('Vulnerable' if v <= 40 else 'Moderate' if v <= 70 else 'Resilient')
                  for v, t in zip(sc['resilience_index'], sc['tier']))
    record('10 data/processed/resilience_scores.csv has airport_code, resilience_index, tier; scores in 0-100; tiers match the 0-40/41-70/71-100 rule',
           str(scores_path.name), {'rows': sc[['airport_code', 'resilience_index', 'tier']].to_dict('records')},
           cols_ok and range_ok and tier_ok)
else:
    record('10 data/processed/resilience_scores.csv exists', str(scores_path), {'exists': False}, False)

dash = ("import sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, %r); "
        "from src import predict, explain; "
        "imp = explain.get_global_importance(); b = predict.load_model(); "
        "d = predict.predict_disruption({}); s = predict.predict_severity({}); "
        "print('@@' + json.dumps({'n_importance': len(imp), 'sorted_desc': bool((imp.diff().dropna() <= 1e-12).all()), "
        "'top': list(imp.index[:3]), 'bundle_keys': sorted(b), 'disruption_keys': sorted(d), 'severity_keys': sorted(s), "
        "'severity_in_labels': s['severity'] in ('none', 'minor', 'moderate', 'severe')}))" % str(ROOT))
proc = subprocess.run([sys.executable, '-c', dash], capture_output=True, text=True, cwd='/')
if proc.returncode == 0:
    o = json.loads(proc.stdout.split('@@')[-1])
    ok = (o['n_importance'] == len(C.FEATURES) and o['sorted_desc'] and {'disruption', 'probability'} <= set(o['disruption_keys'])
          and {'severity', 'severity_code'} <= set(o['severity_keys']) and o['severity_in_labels'])
    record("11 Dashboard-style use (`from src import predict, explain`; `explain.get_global_importance()` with no arguments; `predict.load_model()`); "
           "interface keys 'disruption', 'probability', 'severity', 'severity_code' present", 'fresh process, sys.path.insert(0, project root)', o, ok)
else:
    record('11 Dashboard-style imports work', 'fresh process', {'error': proc.stderr[-300:]}, False)

# write the log
n_pass = sum(r['status'] == 'PASS' for r in results)
lines = ['# CLEARS - Model Verification Log (T-2.8)', '',
         'Generated by `src/verify_models.py`. Pass criteria are the ones in the task guide and were **not** adjusted.',
         f'Data file: `{C.DATA_PATH.name}` | models: `data/models/*.pkl` | **{n_pass} of {len(results)} checks passed.**', '',
         '| # | Check | Result |', '|---|---|---|']
for r in results:
    lines.append(f"| {r['check'].split(' ')[0]} | {r['check'][len(r['check'].split(' ')[0]) + 1:]} | **{r['status']}** |")
lines.append('')
for r in results:
    lines += [f"## {r['check']}", f"**{r['status']}**", '', '- Input: `' + json.dumps(r['input'], default=str)[:600] + '`',
              '- Output: `' + json.dumps(r['output'], default=str)[:900] + '`']
    if r['note']:
        lines.append('- Note: ' + r['note'])
    lines.append('')
(C.DOCS_DIR / 'model_verification.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'{n_pass}/{len(results)} passed')
for r in results:
    print(r['status'], '-', r['check'][:100], '|', json.dumps(r['output'], default=str)[:160])
