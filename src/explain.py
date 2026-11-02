"""SHAP explainability and scenario simulation (Person 2, task T-2.5).

SHAP values come from XGBoost's built-in exact TreeSHAP (`pred_contribs=True`), which gives the same
values as `shap.TreeExplainer` for tree models. The `shap` package is not required; if it is installed
it is used only for the beeswarm plot. Values are in log-odds of the RAW (scale_pos_weight) model output
-- positive pushes towards disruption, negative away from it. They explain the ranking of flights, not
the calibrated probability.

    explain_prediction(model, feature_dict) -> {feature: shap_value}
    get_global_importance(model, X) -> pd.Series (mean |SHAP|, descending)
    scenario_risk(base_features, overrides) -> {'base_probability', 'scenario_probability', 'delta'}
"""
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from . import config as C
    from . import predict as P
except ImportError:                     # pragma: no cover
    import config as C
    import predict as P

# raw weather columns that must be dropped from the base row when a flag / category is overridden, so the
# override reaches the raw values the model reads (predict.build_row back-fills them from the flag)
_OVERRIDE_CLEARS = {
    'visibility_risk': ['visibility_km'], 'storm_flag': ['visibility_km'],
    'wind_category': ['wind_speed_knots'], 'wind_category_enc': ['wind_speed_knots'],
    'rain_severity': ['precip_mm_hr'], 'rain_severity_enc': ['precip_mm_hr'],
    'temp_stress': ['temperature_c'],
}


def _booster(model):
    m = model['model'] if isinstance(model, dict) else model          # accepts the saved bundle or a model
    return m.get_booster() if hasattr(m, 'get_booster') else m


def _features(model) -> list:
    """Feature list of a saved bundle (the scenario model uses a subset); plain models use C.FEATURES."""
    return list(model['features']) if isinstance(model, dict) else list(C.FEATURES)


def _contribs(model, X: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_features + 1) SHAP matrix; the last column is the base value."""
    return _booster(model).predict(xgb.DMatrix(X[_features(model)]), pred_contribs=True)


def explain_prediction(model, feature_dict: dict) -> dict:
    """Signed SHAP value for every model feature for one prediction."""
    X = P.build_row(feature_dict)
    c = _contribs(model, X)[0]
    return {f: float(v) for f, v in zip(_features(model), c[:-1])}


def explain_prediction_detail(model, feature_dict: dict) -> dict:
    """Like explain_prediction, plus the base value and the feature values that were used."""
    X = P.build_row(feature_dict)
    c = _contribs(model, X)[0]
    return {'shap': {f: float(v) for f, v in zip(_features(model), c[:-1])}, 'base_value': float(c[-1]),
            'features': {f: float(X[f].iloc[0]) for f in _features(model)}}


def get_global_importance(model=None, X_test: pd.DataFrame = None, max_rows: int = 20000) -> pd.Series:
    """Mean absolute SHAP value per feature over (a sample of) X_test, sorted descending.

    Called with no arguments (as the dashboard does once at startup, task T-3.3) it uses the saved main model
    and returns the cached table assets/shap_global_importance.csv; the cache is rebuilt from the test split
    whenever the model file is newer than the cache."""
    if model is None:
        model = P.load_model('disruption')
        cache = C.ASSETS_DIR / 'shap_global_importance.csv'
        if X_test is None:
            if cache.exists() and cache.stat().st_mtime >= (C.MODEL_DIR / 'disruption_model.pkl').stat().st_mtime:
                return pd.read_csv(cache, index_col=0).iloc[:, 0].sort_values(ascending=False)
            X_test = _test_features()
            imp = get_global_importance(model, X_test, max_rows)
            C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            imp.rename('mean_abs_shap').to_csv(cache, header=True)
            return imp
    if X_test is None:
        X_test = _test_features()
    feats = _features(model)
    X = X_test[feats]
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=C.RANDOM_STATE)
    c = _contribs(model, X)[:, :-1]
    return pd.Series(np.abs(c).mean(axis=0), index=feats).sort_values(ascending=False)


def _test_features() -> pd.DataFrame:
    """Test-period rows of the configured data file (imported lazily; only needed for the no-argument calls)."""
    try:
        from . import model_train as T
    except ImportError:                 # pragma: no cover
        import model_train as T
    return T.split(T.load_data())[2]


def make_shap_summary_plot(model, X_test: pd.DataFrame, path=None, max_rows: int = 5000) -> str:
    """Beeswarm summary plot saved to assets/shap_summary.png (uses `shap` if installed)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = path or C.ASSETS_DIR / 'shap_summary.png'
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    feats = _features(model)
    X = X_test[feats]
    X = X.sample(max_rows, random_state=C.RANDOM_STATE) if len(X) > max_rows else X
    sv = _contribs(model, X)[:, :-1]
    try:
        import shap
        shap.summary_plot(sv, X, show=False)
        plt.title('SHAP summary: effect on disruption risk (log-odds)')
    except ImportError:               # own beeswarm: rows sorted by mean |SHAP|, colour = normalised value
        order = np.argsort(np.abs(sv).mean(axis=0))
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots(figsize=(8, 7))
        for rank, j in enumerate(order):
            col = X.iloc[:, j].to_numpy(dtype=float)
            lo, hi = np.nanpercentile(col, 1), np.nanpercentile(col, 99)
            color = np.clip((col - lo) / (hi - lo + 1e-12), 0, 1)
            ax.scatter(sv[:, j], rank + rng.uniform(-0.3, 0.3, len(col)), c=color, cmap='coolwarm', s=5, alpha=0.6)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([feats[j] for j in order])
        ax.axvline(0, color='grey', lw=0.8)
        ax.set_xlabel('SHAP value (log-odds; right = pushes towards disruption)')
        ax.set_title('SHAP summary (red = high feature value, blue = low)')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close('all')
    return str(path)


def scenario_risk(base_features: dict, overrides: dict, mode: str = 'full') -> dict:
    """Disruption probability before and after a what-if change.

    `overrides` may use RAW weather names (visibility_km, wind_speed_knots, precip_mm_hr, temperature_c --
    preferred) and/or the flag / category names from the dashboard guide (visibility_risk, wind_category,
    rain_severity, temp_stress, storm_flag, is_peak_hour). For a flag override the matching raw value in the
    base row is replaced by a representative one, otherwise the model would still see the old raw value.

    mode='full' (default): the main model, so base_probability equals the dashboard gauge value. The delay-history
        inputs of the base row are held fixed (a real storm would also raise them), so this is a conservative
        'weather changes, nothing else does' view.
    mode='weather': the weather + time model without delay history, 'risk under these conditions before delays
        have accumulated'. Its base_probability is NOT the gauge value.
    Combined extreme overrides (e.g. visibility 0.5 km AND 10 mm/h rain AND 28 kn wind) go beyond what the data
    contains (only ~70 historical flights had visibility < 1 km together with rain >= 2.5 mm/h, 34% disrupted);
    treat such results as an upper bound, not a forecast."""
    merged = {k: v for k, v in base_features.items() if v is not None}
    for key in overrides:
        for raw in _OVERRIDE_CLEARS.get(key, []):
            if raw not in overrides:
                merged.pop(raw, None)
    if overrides.get('visibility_risk') == 0 and 'storm_flag' not in overrides:
        merged['storm_flag'] = 0                  # visibility restored; do not keep an old storm flag alive
    merged.update(overrides)
    if 'is_peak_hour' in overrides and 'hour_of_day' not in overrides:
        # the model also reads hour_of_day, and build_row re-derives is_peak_hour from it, so move the
        # hour into / out of the peak windows (evening peak 17:00 / midday 12:00) instead of ignoring the override
        h = merged.get('hour_of_day')
        in_peak = h is not None and (6 <= int(h) <= 9 or 16 <= int(h) <= 20)
        if overrides['is_peak_hour'] and not in_peak:
            merged['hour_of_day'] = 17
        elif not overrides['is_peak_hour'] and in_peak:
            merged['hour_of_day'] = 12
    predict = P.predict_weather_risk if mode == 'weather' else P.predict_disruption
    base = predict(base_features)['probability']
    scen = predict(merged)['probability']
    return {'base_probability': float(base), 'scenario_probability': float(scen), 'delta': float(scen - base),
            'mode': mode}


if __name__ == '__main__':
    import json
    import model_train as T
    bundle = P._disruption_bundle()
    df = T.load_data()
    _, _, test = T.split(df)
    imp = get_global_importance(bundle, test)
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    imp.rename('mean_abs_shap').to_csv(C.ASSETS_DIR / 'shap_global_importance.csv', header=True)
    print(make_shap_summary_plot(bundle, test))
    print(json.dumps({k: round(float(v), 4) for k, v in imp.items()}, indent=1))
