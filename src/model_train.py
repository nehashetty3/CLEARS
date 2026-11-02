"""Train the CLEARS disruption and severity models (Person 2, tasks T-2.3 and T-2.4).

    python src/model_train.py                      # uses config.DATA_PATH
    CLEARS_DATA=path/to/regenerated.csv python src/model_train.py

Design decisions (each one fixes a problem found in the data audit):
  * explicit FEATURES list -- dep_delay & friends would leak the label
  * whole-date, time-ordered split: train_core < 2021-10-01 <= validation < 2022-02-01 <= test
  * the TEST set is touched exactly once, after every choice has been made on the validation slice
    (hyper-parameters, early stopping, probability calibration, decision threshold)
  * scale_pos_weight (~4.6) as in the project report, then isotonic calibration so that `probability`
    is a real probability (needed for the Low / Medium / High bands of KPI 4)
  * severity label: 0 for non-disrupted flights, otherwise bins of delay_minutes (negative delays would
    be NaN with pd.cut from 0)
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from . import config as C
    from . import features as F
except ImportError:                 # run as a script: python src/model_train.py
    import config as C
    import features as F

warnings.filterwarnings('ignore', category=UserWarning)

WEATHER_ONLY = ['wind_speed_knots', 'visibility_km', 'temperature_c', 'humidity_pct', 'precip_mm_hr',
                'wind_category_enc', 'visibility_risk', 'rain_severity_enc', 'temp_stress', 'storm_flag']
TIME_ONLY = ['hour_of_day', 'day_of_week', 'month', 'is_peak_hour', 'is_weekend', 'runway_count']
HISTORY = ['delay_last_1h', 'delay_last_3h', 'delay_last_24h', 'congestion_level']
GRID = [(d, lr) for d in (3, 5, 7) for lr in (0.05, 0.1)]      # n_estimators is chosen by early stopping


# ----------------------------------------------------------------------------------------------- data
def severity_label(df: pd.DataFrame) -> pd.Series:
    """0 none (not disrupted), 1 minor (15,45], 2 moderate (45,120], 3 severe (>120)."""
    sev = pd.cut(df['delay_minutes'], bins=C.SEVERITY_BINS, labels=[1, 2, 3])
    return sev.astype('float').fillna(0).astype(int).where(df['disruption'] == 1, 0)


def load_data(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or C.DATA_PATH, low_memory=False, parse_dates=['timestamp', 'flight_date'])
    missing = set(C.FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f'data file lacks model features: {sorted(missing)}')
    df['severity'] = severity_label(df)
    return df.sort_values('timestamp', kind='mergesort').reset_index(drop=True)


def split(df: pd.DataFrame):
    d = df['flight_date']
    train = df[d < C.VAL_START]
    val = df[(d >= C.VAL_START) & (d < C.TEST_START)]
    test = df[d >= C.TEST_START]
    assert train['timestamp'].max() < val['timestamp'].min() <= val['timestamp'].max() < test['timestamp'].min()
    return train, val, test


# ------------------------------------------------------------------------------------------- metrics
def binary_metrics(y, p, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold), 'n': int(len(y)), 'positive_rate': float(np.mean(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'precision_1': float(precision_score(y, pred, zero_division=0)),
        'recall_1': float(recall_score(y, pred, zero_division=0)),
        'f1_1': float(f1_score(y, pred, zero_division=0)),
        'f1_0': float(f1_score(y, pred, pos_label=0, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        'pr_auc': float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


def best_f1_threshold(y, p) -> float:
    """Threshold that maximises F1 of the disruption class, searched on VALIDATION data only."""
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ------------------------------------------------------------------------------------ disruption model
def make_xgb(depth, lr, spw) -> XGBClassifier:
    return XGBClassifier(n_estimators=1000, max_depth=depth, learning_rate=lr, scale_pos_weight=spw,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5, tree_method='hist',
                         eval_metric='aucpr', early_stopping_rounds=30, random_state=C.RANDOM_STATE, n_jobs=-1)


def fit_xgb(train, val, features, depth, lr):
    spw = float((train['disruption'] == 0).sum() / (train['disruption'] == 1).sum())
    m = make_xgb(depth, lr, spw)
    m.fit(train[features], train['disruption'], eval_set=[(val[features], val['disruption'])], verbose=False)
    return m, spw


def train_disruption(train, val, test, log=print):
    # 1) hyper-parameter search on the validation slice (guide grid; n_estimators by early stopping)
    results = []
    for depth, lr in GRID:
        m, _ = fit_xgb(train, val, C.FEATURES, depth, lr)
        ap = average_precision_score(val['disruption'], m.predict_proba(val[C.FEATURES])[:, 1])
        results.append({'max_depth': depth, 'learning_rate': lr, 'best_iteration': int(m.best_iteration),
                        'val_pr_auc': float(ap)})
        log(f'  depth={depth} lr={lr:<4} trees={m.best_iteration:>4}  val PR-AUC={ap:.4f}')
    best = max(results, key=lambda r: r['val_pr_auc'])
    model, spw = fit_xgb(train, val, C.FEATURES, best['max_depth'], best['learning_rate'])

    # 2) calibrate probabilities and pick the decision threshold on the validation slice
    raw_val = model.predict_proba(val[C.FEATURES])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(raw_val, val['disruption'])
    cal_val = calibrator.predict(raw_val)
    threshold = best_f1_threshold(val['disruption'], cal_val)

    # 3) single evaluation on the test slice
    raw_test = model.predict_proba(test[C.FEATURES])[:, 1]
    cal_test = calibrator.predict(raw_test)
    y = test['disruption'].values
    metrics = {
        'best_params': best, 'search': results, 'scale_pos_weight': spw,
        'validation': binary_metrics(val['disruption'].values, cal_val, threshold),
        'test_tuned_threshold': binary_metrics(y, cal_test, threshold),
        'test_raw_0.5': binary_metrics(y, raw_test, 0.5),   # what the uncalibrated model says at 0.5
        'test_brier_calibrated': float(brier_score_loss(y, cal_test)),
        'test_brier_raw': float(brier_score_loss(y, raw_test)),
        'test_brier_constant_baseline': float(brier_score_loss(y, np.full(len(y), train['disruption'].mean()))),
        'risk_band_share_test': {
            'low (<%.2f)' % C.RISK_BANDS[0]: float((cal_test < C.RISK_BANDS[0]).mean()),
            'medium': float(((cal_test >= C.RISK_BANDS[0]) & (cal_test <= C.RISK_BANDS[1])).mean()),
            'high (>%.2f)' % C.RISK_BANDS[1]: float((cal_test > C.RISK_BANDS[1]).mean()),
        },
    }
    for a, g in test.assign(p=cal_test).groupby('airport_code'):
        metrics.setdefault('test_by_airport', {})[a] = binary_metrics(g['disruption'].values, g['p'].values, threshold)
    for mth, g in test.assign(p=cal_test).groupby('month'):
        metrics.setdefault('test_by_month', {})[int(mth)] = binary_metrics(g['disruption'].values, g['p'].values, threshold)

    # high-risk-band precision: how often is a flight flagged High / Low really disrupted?
    for name, mask in [('high', cal_test > C.RISK_BANDS[1]), ('low', cal_test < C.RISK_BANDS[0])]:
        metrics.setdefault('band_observed_rate', {})[name] = (
            {'share': float(mask.mean()), 'observed_disruption_rate': float(y[mask].mean())} if mask.any() else None)

    bundle = {'model': model, 'calibrator': calibrator, 'threshold': threshold, 'features': C.FEATURES,
              'medians': train[C.FEATURES].median().to_dict(),
              'metadata': {'train_end': str(train['flight_date'].max().date()),
                           'val_range': [str(val['flight_date'].min().date()), str(val['flight_date'].max().date())],
                           'data_file': C.DATA_PATH.name, 'params': best}}
    return bundle, metrics, (raw_val, cal_val, raw_test, cal_test)


def baselines_and_ablations(train, val, test, best, threshold_fn=best_f1_threshold, log=print):
    out = {}
    y_val, y = val['disruption'].values, test['disruption'].values

    # logistic-regression baseline (same features, same protocol: threshold chosen on validation)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
    lr.fit(train[C.FEATURES], train['disruption'])
    pv, pt = lr.predict_proba(val[C.FEATURES])[:, 1], lr.predict_proba(test[C.FEATURES])[:, 1]
    cal = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(pv, y_val)
    thr = threshold_fn(y_val, cal.predict(pv))
    out['logistic_regression'] = binary_metrics(y, cal.predict(pt), thr)

    # trivial baselines
    p_const = np.full(len(y), train['disruption'].mean())
    out['always_disrupted'] = binary_metrics(y, np.ones(len(y)), 0.5)
    out['rolling_rate_only'] = {'roc_auc': float(roc_auc_score(y, test['delay_last_3h'])),
                                'pr_auc': float(average_precision_score(y, test['delay_last_3h']))}
    out['base_rate'] = float(y.mean())
    out['train_base_rate'] = float(p_const[0])

    # feature-group ablations (same XGBoost settings, retrained, early stopping on validation)
    groups = {'full': C.FEATURES, 'weather_only': WEATHER_ONLY, 'time_only': TIME_ONLY,
              'no_history_features': [f for f in C.FEATURES if f not in HISTORY]}   # = weather + time
    abl = {}
    for name, feats in groups.items():
        m, _ = fit_xgb(train, val, feats, best['max_depth'], best['learning_rate'])
        pv = m.predict_proba(val[feats])[:, 1]
        cal_a = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(pv, y_val)
        thr_a = threshold_fn(y_val, cal_a.predict(pv))
        mt = binary_metrics(y, cal_a.predict(m.predict_proba(test[feats])[:, 1]), thr_a)
        abl[name] = {k: mt[k] for k in ('roc_auc', 'pr_auc', 'f1_1', 'precision_1', 'recall_1')}
        abl[name]['n_features'] = len(feats)
        log(f'  ablation {name:<22} ROC-AUC={mt["roc_auc"]:.3f}  PR-AUC={mt["pr_auc"]:.3f}  F1={mt["f1_1"]:.3f}')
    out['ablation'] = abl
    return out


# --------------------------------------------------------------------------------------- severity model
def diagnostics(df, best) -> dict:
    """Is the modest test score caused by the time shift? Compare with a date-grouped cross-validation
    (NOT time-ordered, so it is optimistic) and with expanding-window yearly tests."""
    from sklearn.model_selection import GroupKFold
    spw = float((df['disruption'] == 0).sum() / (df['disruption'] == 1).sum())
    n_trees = max(best['best_iteration'], 40)

    def fit_predict(tr, te):
        m = make_xgb(best['max_depth'], best['learning_rate'], spw)
        m.set_params(early_stopping_rounds=None, n_estimators=n_trees)
        m.fit(tr[C.FEATURES], tr['disruption'])
        return m.predict_proba(te[C.FEATURES])[:, 1]

    aucs, aps, f1s = [], [], []
    for tr_i, te_i in GroupKFold(n_splits=5).split(df, groups=df['flight_date']):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        p = fit_predict(tr, te)
        thr = np.quantile(p, 1 - te['disruption'].mean())          # flag the same share as the base rate
        aucs.append(roc_auc_score(te['disruption'], p))
        aps.append(average_precision_score(te['disruption'], p))
        f1s.append(f1_score(te['disruption'], p >= thr))
    out = {'date_grouped_cv': {'roc_auc': float(np.mean(aucs)), 'pr_auc': float(np.mean(aps)),
                               'f1_1_at_base_rate_threshold': float(np.mean(f1s))}}
    for yr in (2021, 2022):
        tr, te = df[df['flight_date'].dt.year < yr], df[df['flight_date'].dt.year == yr]
        p = fit_predict(tr, te)
        out[f'train_before_{yr}_test_{yr}'] = {'roc_auc': float(roc_auc_score(te['disruption'], p)),
                                               'pr_auc': float(average_precision_score(te['disruption'], p)),
                                               'base_rate': float(te['disruption'].mean())}
    return out


def train_severity(train, val, test, log=print):
    best, best_loss, model = None, np.inf, None
    for depth, lr in [(3, 0.1), (5, 0.1), (5, 0.05)]:
        m = XGBClassifier(objective='multi:softprob', num_class=4, n_estimators=800, max_depth=depth,
                          learning_rate=lr, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          tree_method='hist', eval_metric='mlogloss', early_stopping_rounds=30,
                          random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(train[C.FEATURES], train['severity'], sample_weight=compute_sample_weight('balanced', train['severity']),
              eval_set=[(val[C.FEATURES], val['severity'])],
              sample_weight_eval_set=[compute_sample_weight('balanced', val['severity'])], verbose=False)
        loss = m.best_score
        log(f'  severity depth={depth} lr={lr}: trees={m.best_iteration} val weighted mlogloss={loss:.4f}')
        if loss < best_loss:
            best, best_loss, model = {'max_depth': depth, 'learning_rate': lr, 'best_iteration': int(m.best_iteration)}, loss, m
    proba = model.predict_proba(test[C.FEATURES])
    pred = proba.argmax(axis=1)
    y = test['severity'].values
    names = [C.SEVERITY_LABELS[i] for i in range(4)]
    report = classification_report(y, pred, labels=[0, 1, 2, 3], target_names=names, output_dict=True, zero_division=0)
    metrics = {
        'best_params': best, 'class_counts_test': {n: int((y == i).sum()) for i, n in enumerate(names)},
        'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro')),
        'baseline_always_none': {'accuracy': float((y == 0).mean())},
        'report': report,
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
        'mean_expected_severity_by_true_class': {n: float((proba[y == i] @ np.arange(4)).mean())
                                                 for i, n in enumerate(names)},
    }
    bundle = {'model': model, 'features': C.FEATURES, 'labels': C.SEVERITY_LABELS,
              'medians': train[C.FEATURES].median().to_dict(), 'metadata': {'params': best}}
    return bundle, metrics


# directional (monotone) constraints for the scenario model: worse weather never lowers the risk
MONOTONE = {'wind_speed_knots': 1, 'visibility_km': -1, 'precip_mm_hr': 1, 'wind_category_enc': 1,
            'rain_severity_enc': 1, 'visibility_risk': 1, 'storm_flag': 1, 'temp_stress': 1}
SCENARIOS = {'low visibility (0.5 km)': {'visibility_km': 0.5}, 'heavy rain (10 mm/h)': {'precip_mm_hr': 10.0},
             'strong wind (28 kn)': {'wind_speed_knots': 28.0},
             'all three together': {'visibility_km': 0.5, 'precip_mm_hr': 10.0, 'wind_speed_knots': 28.0},
             'clear day (16 km, dry, 5 kn)': {'visibility_km': 16.0, 'precip_mm_hr': 0.0, 'wind_speed_knots': 5.0}}


def train_weather_model(train, val, test, log=print):
    """Disruption model WITHOUT the delay-history features: an optional second view for the what-if simulator.

    It sees weather, time and airport only, with monotone constraints, and answers: 'what is the disruption risk
    under these conditions before any delay history has built up?'. Chosen on validation PR-AUC like the main model.
    (It was introduced when the weather columns were ~5-6 h out of sync with the flights, which made the main
    model nearly insensitive to weather; with correctly aligned weather the main model responds too, so the
    simulator now defaults to the main model.)"""
    feats = WEATHER_ONLY + TIME_ONLY
    mono = tuple(MONOTONE.get(f, 0) for f in feats)
    spw = float((train['disruption'] == 0).sum() / (train['disruption'] == 1).sum())
    best, best_ap, best_model = None, -1, None
    for depth, lr in [(3, 0.03), (3, 0.05), (4, 0.03), (4, 0.05), (5, 0.05)]:
        m = XGBClassifier(n_estimators=8000, max_depth=depth, learning_rate=lr, scale_pos_weight=spw, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, tree_method='hist', eval_metric='logloss',
                          early_stopping_rounds=100, monotone_constraints=mono, random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(train[feats], train['disruption'], eval_set=[(val[feats], val['disruption'])], verbose=False)
        ap = average_precision_score(val['disruption'], m.predict_proba(val[feats])[:, 1])
        log(f'  weather model depth={depth} lr={lr}: trees={m.best_iteration + 1:>4} val PR-AUC={ap:.4f}')
        if ap > best_ap:
            best, best_ap, best_model = {'max_depth': depth, 'learning_rate': lr, 'best_iteration': int(m.best_iteration),
                                         'val_pr_auc': float(ap)}, ap, m
    pv = best_model.predict_proba(val[feats])[:, 1]
    cal = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(pv, val['disruption'])
    thr = best_f1_threshold(val['disruption'].values, cal.predict(pv))
    pt = cal.predict(best_model.predict_proba(test[feats])[:, 1])
    metrics = {'best_params': best, 'test': binary_metrics(test['disruption'].values, pt, thr)}
    bundle = {'model': best_model, 'calibrator': cal, 'threshold': thr, 'features': feats,
              'medians': train[C.FEATURES].median().to_dict(), 'metadata': {'params': best, 'purpose': 'what-if scenarios'}}
    return bundle, metrics


def scenario_table(full_bundle, weather_bundle, test, n=5000) -> dict:
    """Mean predicted probability under each what-if scenario, for both models, on a sample of test flights.

    The raw weather columns are overridden and the flags / encodings are re-derived from them (features.py), so a
    scenario row never contradicts itself. The delay-history inputs of each flight are left unchanged."""
    sample = test[C.FEATURES].sample(n, random_state=C.RANDOM_STATE)

    def prob(bundle, X):
        return bundle['calibrator'].predict(bundle['model'].predict_proba(X[bundle['features']])[:, 1])

    base = {'full': prob(full_bundle, sample), 'weather': prob(weather_bundle, sample)}
    out = {'baseline': {'full_prob': float(base['full'].mean()), 'weather_prob': float(base['weather'].mean())}}
    for name, change in SCENARIOS.items():
        s = sample.copy()
        for k, v in change.items():
            s[k] = v
        s = F.add_weather_features(s)[C.FEATURES]
        pf, pw = prob(full_bundle, s), prob(weather_bundle, s)
        out[name] = {'override': change, 'full_prob': float(pf.mean()), 'full_delta': float((pf - base['full']).mean()),
                     'weather_prob': float(pw.mean()), 'weather_delta': float((pw - base['weather']).mean())}
    return out


def D_RANGE(part):
    return str(part['flight_date'].min().date()), str(part['flight_date'].max().date())


def main():
    log = print
    df = load_data()
    train, val, test = split(df)
    log(f'data: {C.DATA_PATH.name}  rows={len(df):,}')
    for name, part in [('train', train), ('validation', val), ('test', test)]:
        log(f'  {name:<10} {part["flight_date"].min().date()} -> {part["flight_date"].max().date()}  '
            f'rows={len(part):>7,}  disruption rate={part["disruption"].mean():.3f}')
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    log('\nDisruption model: grid search on the validation slice')
    bundle, metrics, arrays = train_disruption(train, val, test, log)
    log('\nBaselines and ablations')
    metrics['baselines'] = baselines_and_ablations(train, val, test, metrics['best_params'], log=log)
    log('\nDiagnostics: time-ordered vs date-grouped evaluation')
    metrics['diagnostics'] = diagnostics(df, metrics['best_params'])
    log('  ', metrics['diagnostics'])
    log('\nSeverity model')
    sev_bundle, sev_metrics = train_severity(train, val, test, log)
    metrics['severity'] = sev_metrics
    metrics['data'] = {
        'file': C.DATA_PATH.name, 'features': C.FEATURES,
        'train_rows': len(train), 'val_rows': len(val), 'test_rows': len(test),
        'train_range': [str(train['flight_date'].min().date()), str(train['flight_date'].max().date())],
        'val_range': [str(val['flight_date'].min().date()), str(val['flight_date'].max().date())],
        'test_range': [str(test['flight_date'].min().date()), str(test['flight_date'].max().date())],
        'train_rate': float(train['disruption'].mean()), 'val_rate': float(val['disruption'].mean()),
        'test_rate': float(test['disruption'].mean()),
    }
    joblib.dump(bundle, C.MODEL_DIR / 'disruption_model.pkl')
    log('\nWeather scenario model (what-if simulator)')
    weather_bundle, metrics['weather_model'] = train_weather_model(train, val, test, log)
    metrics['scenario_table'] = scenario_table(bundle, weather_bundle, test)
    log('  scenario response (mean change in probability): full model',
        {k: round(v['full_delta'], 3) for k, v in metrics['scenario_table'].items() if 'full_delta' in v})
    log('                                                  weather model',
        {k: round(v['weather_delta'], 3) for k, v in metrics['scenario_table'].items() if 'weather_delta' in v})
    joblib.dump(weather_bundle, C.MODEL_DIR / 'scenario_model.pkl')
    joblib.dump(sev_bundle, C.MODEL_DIR / 'severity_model.pkl')
    (C.MODEL_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    t = metrics['test_tuned_threshold']
    y_test, cal_test = test['disruption'].values, arrays[3]
    cm_ = t['confusion_matrix']
    start, end = D_RANGE(test)
    report_text = '\n'.join([
        f"Disruption model - TEST ({start} to {end}), calibrated probability, threshold {t['threshold']:.3f} "
        'chosen on validation', '',
        classification_report(y_test, (cal_test >= t['threshold']).astype(int),
                              target_names=['no disruption', 'disruption'], digits=3),
        f"ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f}",
        f"Confusion matrix [[tn fp] [fn tp]]: [[{cm_['tn']}, {cm_['fp']}], [{cm_['fn']}, {cm_['tp']}]]", '',
        f'Severity model - TEST ({start} to {end})', '',
        classification_report(test['severity'].values,
                              sev_bundle['model'].predict_proba(test[C.FEATURES]).argmax(axis=1),
                              target_names=[C.SEVERITY_LABELS[i] for i in range(4)], digits=3, zero_division=0)])
    (C.MODEL_DIR / 'classification_report.txt').write_text(report_text, encoding='utf-8')
    log('\n' + report_text)
    log('\n=== TEST (2022-02-01 onward), threshold %.3f chosen on validation ===' % t['threshold'])
    log(f"  ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | F1(disruption) {t['f1_1']:.3f} | "
        f"precision {t['precision_1']:.3f} | recall {t['recall_1']:.3f} | accuracy {t['accuracy']:.3f}")
    lr = metrics['baselines']['logistic_regression']
    log(f"  logistic regression: ROC-AUC {lr['roc_auc']:.3f} | PR-AUC {lr['pr_auc']:.3f} | F1 {lr['f1_1']:.3f}")
    log(f"  severity: accuracy {sev_metrics['accuracy']:.3f} | macro-F1 {sev_metrics['macro_f1']:.3f}")
    log('  saved models and metrics.json to', C.MODEL_DIR)


if __name__ == '__main__':
    main()
