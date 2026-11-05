"""Cached data and model access for the dashboard (T-3.3). Everything that reads a file or runs the models is
wrapped in st.cache_data so a widget click does not repeat the work."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))          # the model code imports itself as plain modules
import config as C            # noqa: E402
import dashboard_data as D    # noqa: E402
import explain as E           # noqa: E402
import predict as P           # noqa: E402
import recommend as R         # noqa: E402

FRIENDLY = {
    'wind_speed_knots': ('Wind speed', 'kt'), 'visibility_km': ('Visibility', 'km'), 'temperature_c': ('Temperature', '°C'),
    'humidity_pct': ('Humidity', '%'), 'precip_mm_hr': ('Rainfall', 'mm/h'), 'wind_category_enc': ('Wind category', ''),
    'visibility_risk': ('Low-visibility flag', ''), 'rain_severity_enc': ('Rain severity', ''), 'temp_stress': ('Temperature stress', ''),
    'storm_flag': ('Storm flag', ''), 'hour_of_day': ('Hour of day', 'h'), 'day_of_week': ('Day of week', ''),
    'month': ('Month', ''), 'is_peak_hour': ('Peak hour', ''), 'is_weekend': ('Weekend', ''),
    'delay_last_1h': ('Delays 2 h ago', ''), 'delay_last_3h': ('Delays, last 3 h', ''), 'delay_last_24h': ('Delays, last 24 h', ''),
    'congestion_level': ('Recent traffic', 'flights'), 'runway_count': ('Runways', ''),
}
GROUPS = {
    'Recent delay history': ['delay_last_1h', 'delay_last_3h', 'delay_last_24h'],
    'Time of day and season': ['hour_of_day', 'day_of_week', 'month', 'is_peak_hour', 'is_weekend'],
    'Weather': ['wind_speed_knots', 'visibility_km', 'temperature_c', 'humidity_pct', 'precip_mm_hr', 'wind_category_enc',
                'visibility_risk', 'rain_severity_enc', 'temp_stress', 'storm_flag'],
    'Traffic and airport': ['congestion_level', 'runway_count'],
}
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def label(feature: str) -> str:
    return FRIENDLY.get(feature, (feature, ''))[0]


def value_text(feature: str, v: float) -> str:
    unit = FRIENDLY.get(feature, ('', ''))[1]
    if feature in ('visibility_risk', 'storm_flag', 'temp_stress', 'is_peak_hour', 'is_weekend'):
        return 'yes' if v >= 0.5 else 'no'
    if feature == 'day_of_week':
        return DAYS[int(v)]
    if feature == 'wind_category_enc':
        return ['calm', 'moderate', 'strong', 'severe'][int(v)]
    if feature == 'rain_severity_enc':
        return ['none', 'light', 'moderate', 'heavy'][int(v)]
    if feature.startswith('delay_last'):
        return f'{v:.0%} of flights late'
    if feature == 'hour_of_day':
        return f'{int(v):02d}:00'
    if feature in ('month', 'runway_count'):
        return f'{int(v)}'
    return f'{v:.1f} {unit}'.strip()


@st.cache_data(show_spinner='Scoring airport-hours with the model...')
def load_scored() -> pd.DataFrame:
    return D.score_hourly(D.load_hourly())


@st.cache_data
def load_resilience() -> pd.DataFrame:
    scores = pd.read_csv(C.MODEL_DIR.parent / 'processed' / 'resilience_scores.csv')
    inputs = pd.read_csv(C.MODEL_DIR.parent / 'processed' / 'airport_resilience_inputs.csv')
    ctx = pd.read_csv(C.CONTEXT_PATH)[['airport_code', 'name', 'municipality', 'hub_size']]
    out = scores.merge(inputs.drop(columns=['disruption_rate', 'hub_size', 'municipality'], errors='ignore'), on='airport_code')
    out = out.merge(ctx, on='airport_code', how='left')
    return out.sort_values('resilience_index', ascending=False).reset_index(drop=True)


@st.cache_data
def airport_names() -> dict:
    ctx = pd.read_csv(C.CONTEXT_PATH)
    return {r.airport_code: (r.name.replace('International Airport', '').replace('Frederick W. Smith', 'Memphis').strip(),
                             r.municipality) for r in ctx.itertuples()}


@st.cache_data
def load_metrics() -> dict:
    return json.loads((C.MODEL_DIR / 'metrics.json').read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False)
def global_importance() -> pd.Series:
    return E.get_global_importance()


def group_importance(imp: pd.Series) -> pd.Series:
    return pd.Series({g: float(imp.reindex(cols).sum()) for g, cols in GROUPS.items()}).sort_values(ascending=False)


def slice_range(hourly: pd.DataFrame, airport: str, start, end) -> pd.DataFrame:
    m = (hourly['airport_code'] == airport) & (hourly['date'] >= pd.Timestamp(start)) & (hourly['date'] <= pd.Timestamp(end))
    return hourly[m]


def kpis(hourly: pd.DataFrame, airport: str, start, end) -> dict:
    """Query-time KPIs of docs/kpi_definitions.md (1, 2 and 6), with the prior period of equal length for the trend arrow."""
    def rate(df):
        n = df['n_flights'].sum()
        return df['n_disrupted'].sum() / n * 100 if n else np.nan

    cur = slice_range(hourly, airport, start, end)
    span = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    prior = slice_range(hourly, airport, pd.Timestamp(start) - pd.Timedelta(days=span), pd.Timestamp(start) - pd.Timedelta(days=1))
    disrupted = cur['n_disrupted'].sum()
    peak, off = rate(cur[cur['is_peak_hour'] == 1]), rate(cur[cur['is_peak_hour'] == 0])
    return {'rate': rate(cur), 'prior_rate': rate(prior) if len(prior) else np.nan, 'flights': int(cur['n_flights'].sum()),
            'disrupted': int(disrupted), 'avg_delay': cur['delay_sum_disrupted'].sum() / disrupted if disrupted else np.nan,
            'peak_premium': peak / off if off and not np.isnan(off) else np.nan, 'hours': len(cur)}


def pick_snapshot(hourly: pd.DataFrame, airport: str, start, end, mode: str, date=None, hour=None):
    """The airport-hour the dashboard describes: the riskiest hour of the range or one the user chose."""
    rng = slice_range(hourly, airport, start, end)
    if rng.empty:
        return None
    if mode == 'pick' and date is not None:
        day = rng[rng['date'] == pd.Timestamp(date)]
        if not day.empty:
            hrs = day['hour_of_day'].astype(int)
            return day.iloc[int((hrs - (hour if hour is not None else 12)).abs().argmin())]
    return rng.sort_values(['probability', 'n_flights']).iloc[-1]


def feature_dict(row: pd.Series) -> dict:
    d = {c: float(row[c]) for c in C.FEATURES}
    d['airport_code'] = row['airport_code']
    return d


def conditions_of(features: dict) -> dict:
    return {k: features[k] for k in ['visibility_km', 'wind_speed_knots', 'precip_mm_hr', 'temperature_c', 'storm_flag']}


@st.cache_data(show_spinner=False, max_entries=512)
def score_conditions(features: dict, mode: str = 'full') -> dict:
    """Probability, severity and per-feature SHAP of one feature dict (a snapshot or a what-if)."""
    pred = P.predict_disruption(features) if mode == 'full' else P.predict_weather_risk(features)
    sev = P.predict_severity(features)
    model = P.load_model('disruption' if mode == 'full' else 'scenario')
    shap = E.explain_prediction(model, features)
    return {'pred': pred, 'sev': sev, 'shap': shap, 'used': {c: float(v) for c, v in P.build_row(features).iloc[0].items()}}


@st.cache_data(show_spinner=False, max_entries=256)
def run_scenario(features: dict, overrides: dict, mode: str = 'full') -> dict:
    """explain.scenario_risk (the T-2.5 function the guide names) behind a cache."""
    return E.scenario_risk(features, overrides, mode=mode)


def top_drivers(shap: dict, n: int = 4) -> list:
    """Features pushing the prediction TOWARDS disruption, strongest first."""
    return [k for k, v in sorted(shap.items(), key=lambda kv: -kv[1])[:n] if v > 0]


@st.cache_data(show_spinner=False, max_entries=256)
def sensitivity(features: dict, overrides: dict, name: str, values: tuple, mode: str = 'full') -> list:
    """Disruption probability as one weather input sweeps `values` while everything else stays as in the what-if."""
    out = []
    for v in values:
        merged = E.apply_overrides(features, {**overrides, name: float(v)})
        out.append((P.predict_disruption(merged) if mode == 'full' else P.predict_weather_risk(merged))['probability'])
    return out


def risk_band_of(p: float) -> str:
    return P.risk_level(p)


def base_rate(hourly: pd.DataFrame, airport: str) -> float:
    """Average disruption rate (0-1) of the airport over the whole history, the 'normal' to compare a risk against."""
    h = hourly[hourly['airport_code'] == airport]
    return float(h['n_disrupted'].sum() / h['n_flights'].sum())
