"""Data layer for the dashboard (Person 3, tasks T-3.3 / T-3.4).

The dashboard works on one row per airport-hour (16,000 rows) instead of 358,594 flights: all model inputs
are hour-level, so the hourly table loses nothing the models read, loads instantly and is small enough to
live in the repository (data/processed/dashboard_hourly.csv, built from features_final.csv by `python
src/dashboard_data.py`).

    build_hourly(flights)   flight-level feature matrix -> airport-hour table
    load_hourly()           reads the committed hourly table (rebuilds it from features_final.csv if absent)
    score_hourly(hourly)    adds the calibrated disruption probability of the saved model to every hour
    daily_series(hourly)    one row per airport-day for the trend chart (calendar gaps kept as NaN)

Within an hour the weather differs slightly between flights (the backward match picks the latest observation
before each departure), so the hourly weather is the mean of those observations and the flags / categories are
re-derived from it with the same thresholds as features.py. The delay-history inputs are identical for every
flight of an hour.
"""
import numpy as np
import pandas as pd

try:
    from . import config as C
    from . import features as F
    from . import predict as P
except ImportError:                     # pragma: no cover
    import config as C
    import features as F
    import predict as P

HOURLY_PATH = C.ROOT / 'data' / 'processed' / 'dashboard_hourly.csv'
RAW_WEATHER = ['wind_speed_knots', 'visibility_km', 'temperature_c', 'humidity_pct', 'precip_mm_hr']
HISTORY = ['delay_last_1h', 'delay_last_3h', 'delay_last_24h', 'congestion_level']
PERIODS = ['train', 'validation', 'test']


def period_of(dates: pd.Series) -> pd.Series:
    """Train / validation / test label of a date column (the split in config.py)."""
    d = pd.to_datetime(dates)
    return pd.Series(np.where(d >= C.TEST_START, 'test', np.where(d >= C.VAL_START, 'validation', 'train')),
                     index=dates.index)


def build_hourly(flights: pd.DataFrame = None) -> pd.DataFrame:
    """Aggregate the flight-level feature matrix to one row per airport-hour."""
    if flights is None:
        cols = ['airport_code', 'timestamp', 'disruption', 'delay_minutes'] + RAW_WEATHER + HISTORY + ['runway_count']
        flights = pd.read_csv(C.DATA_PATH, usecols=cols)
    f = flights.copy()
    f['timestamp'] = pd.to_datetime(f['timestamp'])
    f['delay_if_disrupted'] = f['delay_minutes'].where(f['disruption'] == 1, 0.0)
    agg = {'n_flights': ('disruption', 'size'), 'n_disrupted': ('disruption', 'sum'),
           'delay_sum_disrupted': ('delay_if_disrupted', 'sum'), 'runway_count': ('runway_count', 'first')}
    agg.update({c: (c, 'mean') for c in RAW_WEATHER})
    agg.update({c: (c, 'first') for c in HISTORY})
    h = f.groupby(['airport_code', 'timestamp']).agg(**agg).reset_index()
    h = F.add_weather_features(h)
    h = F.add_time_features(h)
    h['date'] = h['timestamp'].dt.normalize()
    h['period'] = period_of(h['date'])
    keep = (['airport_code', 'timestamp', 'date', 'period', 'n_flights', 'n_disrupted', 'delay_sum_disrupted']
            + C.FEATURES)
    return h[keep].sort_values(['airport_code', 'timestamp']).reset_index(drop=True)


def save_hourly(path=HOURLY_PATH) -> pd.DataFrame:
    h = build_hourly()
    out = h.copy()
    out['timestamp'] = out['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.round(4).to_csv(path, index=False)
    return h


def load_hourly() -> pd.DataFrame:
    """The committed hourly table; rebuilt from features_final.csv when the table is missing."""
    if not HOURLY_PATH.exists():
        if not C.DATA_PATH.exists():
            raise FileNotFoundError(f'Neither {HOURLY_PATH} nor {C.DATA_PATH} exists. Run `python src/dashboard_data.py` '
                                    'after building the dataset (see README, Quick start).')
        return save_hourly()
    h = pd.read_csv(HOURLY_PATH, parse_dates=['timestamp', 'date'])
    return h


def score_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add `probability` (calibrated, the value the dashboard gauge shows) and `risk_level` to every hour."""
    b = P.load_model('disruption')
    raw = b['model'].predict_proba(hourly[b['features']])[:, 1]
    out = hourly.copy()
    out['probability'] = b['calibrator'].predict(raw)
    low, high = C.RISK_BANDS
    out['risk_level'] = np.where(out['probability'] < low, 'Low', np.where(out['probability'] > high, 'High', 'Medium'))
    return out


def daily_series(hourly: pd.DataFrame, airport: str, start, end) -> pd.DataFrame:
    """Airport-day table over [start, end] with the actual disruption rate, mean predicted probability and a
    storm flag; days without data (the dataset covers four months per year) stay in as NaN rows so a line
    chart breaks at the gap instead of joining across it."""
    h = hourly[(hourly['airport_code'] == airport) & (hourly['date'] >= pd.Timestamp(start))
               & (hourly['date'] <= pd.Timestamp(end))]
    g = h.groupby('date').apply(lambda x: pd.Series({
        'flights': x['n_flights'].sum(), 'disrupted': x['n_disrupted'].sum(),
        'actual_rate': x['n_disrupted'].sum() / x['n_flights'].sum() * 100,
        'predicted_rate': np.average(x['probability'], weights=x['n_flights']) * 100,
        'storm': int(x['storm_flag'].max()), 'low_vis': int(x['visibility_risk'].max())}), include_groups=False)
    full = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq='D')
    g = g.reindex(full)
    g.index.name = 'date'
    return g.reset_index()


if __name__ == '__main__':
    hh = save_hourly()
    print(f'wrote {HOURLY_PATH}  rows={len(hh):,}  size={HOURLY_PATH.stat().st_size / 1e6:.1f} MB')
