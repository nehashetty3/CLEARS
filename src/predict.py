"""Prediction interface used by the dashboard (Person 2, tasks T-2.2 / T-2.3 / T-2.4).

    predict_disruption(feature_dict) -> {'disruption', 'probability', 'raw_score', 'risk_level', 'threshold'}
    predict_severity(feature_dict)   -> {'severity', 'severity_code', 'probabilities'}

The function names and the two guaranteed keys of each result match the interface contract in the task
guide. Extra keys were added; callers may ignore them.

`feature_dict` may be a complete data row or a PARTIAL dict. Rules, in order:
  1. string categories are accepted: wind_category ('calm'|'moderate'|'strong'|'severe') and
     rain_severity ('none'|'light'|'moderate'|'heavy'); airport_code gives runway_count (ORD 8, MEM 4)
  2. if a derived flag / category is given but its raw weather value is not, a representative raw value is
     filled in (see _REPRESENTATIVE) so the model, which reads the raw values too, sees a consistent row
  3. when raw weather values are present, the flags and encodings are RE-DERIVED from them (raw wins)
     with the thresholds in features.py, so a row can never contradict itself
  4. is_peak_hour / is_weekend are derived from hour_of_day / day_of_week when those are given
  5. anything still missing is set to its training-data median
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

try:                                    # works as `from src import predict` and as a plain script import
    from . import config as C
    from . import features as F
except ImportError:                     # pragma: no cover
    import config as C
    import features as F

WIND_ENC = {'calm': 0, 'moderate': 1, 'strong': 2, 'severe': 3}
RAIN_ENC = {'none': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
RUNWAYS = {'ORD': 8, 'MEM': 4}
RAW_WEATHER = ['wind_speed_knots', 'visibility_km', 'temperature_c', 'humidity_pct', 'precip_mm_hr']

# raw values used when only a flag / category is known (chosen inside each category's range)
_REPRESENTATIVE = {
    'wind_speed_knots': {0: 5.0, 1: 17.0, 2: 35.0, 3: 55.0},        # by wind_category_enc
    'precip_mm_hr': {0: 0.0, 1: 1.0, 2: 5.0, 3: 10.0},              # by rain_severity_enc
}


def load_model(name: str = 'disruption') -> dict:
    """Saved model bundle: {'model', 'calibrator', 'threshold', 'features', 'medians', 'metadata'}.

    name: 'disruption' (main model), 'severity', or 'scenario' (weather what-if model). The .pkl files hold this
    dict rather than a bare XGBClassifier because the calibrator and decision threshold must travel with the model.
    Pass the bundle (or bundle['model']) to explain.explain_prediction / get_global_importance."""
    loaders = {'disruption': _disruption_bundle, 'severity': _severity_bundle, 'scenario': _scenario_bundle}
    if name not in loaders:
        raise ValueError(f"name must be one of {sorted(loaders)}")
    return loaders[name]()


@lru_cache(maxsize=None)
def _disruption_bundle():
    return joblib.load(C.MODEL_DIR / 'disruption_model.pkl')


@lru_cache(maxsize=None)
def _severity_bundle():
    return joblib.load(C.MODEL_DIR / 'severity_model.pkl')


def build_row(feature_dict: dict, medians: dict = None) -> pd.DataFrame:
    """One-row DataFrame with exactly C.FEATURES, in order (see module docstring for the rules)."""
    medians = medians or _disruption_bundle()['medians']
    d = {k: v for k, v in dict(feature_dict).items() if v is not None}

    if 'wind_category' in d and 'wind_category_enc' not in d:
        d['wind_category_enc'] = WIND_ENC[str(d['wind_category'])]
    if 'rain_severity' in d and 'rain_severity_enc' not in d:
        d['rain_severity_enc'] = RAIN_ENC[str(d['rain_severity'])]
    if 'runway_count' not in d and d.get('airport_code') in RUNWAYS:
        d['runway_count'] = RUNWAYS[d['airport_code']]

    # --- raw weather back-fill from flags / categories (rule 2)
    if 'wind_speed_knots' not in d and 'wind_category_enc' in d:
        d['wind_speed_knots'] = _REPRESENTATIVE['wind_speed_knots'][int(d['wind_category_enc'])]
    if 'precip_mm_hr' not in d and 'rain_severity_enc' in d:
        d['precip_mm_hr'] = _REPRESENTATIVE['precip_mm_hr'][int(d['rain_severity_enc'])]
    if 'visibility_km' not in d:
        if d.get('storm_flag') == 1:
            d['visibility_km'] = 0.5           # storm_flag fires on visibility < 1 km (or heavy rain / wind)
        elif d.get('visibility_risk') == 1:
            d['visibility_km'] = 2.0
        elif d.get('visibility_risk') == 0 and 'storm_flag' in d:
            d['visibility_km'] = 10.0
    if 'temperature_c' not in d and 'temp_stress' in d:
        d['temperature_c'] = 38.0 if d['temp_stress'] == 1 else 15.0

    # --- complete the raw columns, then re-derive flags and encodings from them (rule 3)
    row = {c: d.get(c, medians[c]) for c in RAW_WEATHER}
    wx = F.add_weather_features(pd.DataFrame([row]))
    for c in ['wind_category_enc', 'visibility_risk', 'rain_severity_enc', 'temp_stress', 'storm_flag']:
        d[c] = int(wx[c].iloc[0])
    for c in RAW_WEATHER:
        d[c] = float(row[c])

    # --- time flags (rule 4)
    if 'hour_of_day' in d:
        h = int(d['hour_of_day'])
        d['is_peak_hour'] = int(6 <= h <= 9 or 16 <= h <= 20)
    if 'day_of_week' in d:
        d['is_weekend'] = int(int(d['day_of_week']) >= 5)

    out = {c: float(d.get(c, medians[c])) for c in C.FEATURES}      # rule 5
    return pd.DataFrame([out], columns=C.FEATURES)


def risk_level(probability: float) -> str:
    low, high = C.RISK_BANDS
    return 'Low' if probability < low else ('High' if probability > high else 'Medium')


@lru_cache(maxsize=None)
def _scenario_bundle():
    return joblib.load(C.MODEL_DIR / 'scenario_model.pkl')


def _score(bundle: dict, feature_dict: dict) -> dict:
    X = build_row(feature_dict, _disruption_bundle()['medians'])[bundle['features']]
    raw = float(bundle['model'].predict_proba(X)[:, 1][0])
    prob = float(bundle['calibrator'].predict([raw])[0])
    return {'disruption': int(prob >= bundle['threshold']), 'probability': prob, 'raw_score': raw,
            'risk_level': risk_level(prob), 'threshold': float(bundle['threshold'])}


def predict_disruption(feature_dict: dict) -> dict:
    """Full model (weather + time + delay history). `probability` is calibrated; `disruption` applies the
    F1-optimal threshold chosen on validation data; `raw_score` is the uncalibrated model output."""
    return _score(_disruption_bundle(), feature_dict)


def predict_weather_risk(feature_dict: dict) -> dict:
    """Weather + time + airport only (no delay history), monotone in the weather inputs: risk under these
    conditions before any delay history has built up. Optional second view for the what-if simulator
    (explain.scenario_risk(..., mode='weather'))."""
    return _score(_scenario_bundle(), feature_dict)


def predict_severity(feature_dict: dict) -> dict:
    b = _severity_bundle()
    X = build_row(feature_dict, b['medians'])
    proba = b['model'].predict_proba(X)[0]
    code = int(np.argmax(proba))
    return {'severity': b['labels'][code], 'severity_code': code,
            'probabilities': {b['labels'][i]: float(p) for i, p in enumerate(proba)}}
