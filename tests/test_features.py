"""Leakage and consistency tests for src/features.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import features as F  # noqa: E402

HISTORY = ['delay_last_1h', 'delay_last_3h', 'delay_last_24h', 'congestion_level']


def synthetic(seed=0, hours=60, per_hour=6):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(hours):
        for _ in range(per_hour):
            delay = float(rng.choice([-5, 0, 10, 30, 90]))
            rows.append({'airport_code': 'ORD', 'timestamp': pd.Timestamp('2022-04-01 05:00') + pd.Timedelta(hours=h),
                         'dep_delay': delay})
    df = pd.DataFrame(rows)
    df['flight_date'] = df['timestamp'].dt.normalize()
    df['wind_speed_knots'], df['visibility_km'], df['temperature_c'] = 5.0, 10.0, 15.0
    df['humidity_pct'], df['precip_mm_hr'] = 60.0, 0.0
    df['arr_delay'] = df['dep_delay']
    for c in ['cancelled', 'weather_delay', 'carrier_delay', 'nas_delay', 'late_aircraft_delay']:
        df[c] = 0.0
    df['disruption'] = (df['dep_delay'] > 15).astype(int)
    df['delay_minutes'] = df['dep_delay'].clip(upper=300)
    df['runway_count'], df['hub_size'] = 8, 'large_hub'
    return df


def test_features_do_not_use_outcomes_of_current_or_previous_hour():
    df = synthetic()
    base = F.build_features(df)
    target_hour = pd.Timestamp('2022-04-02 12:00')
    flip = df['timestamp'].isin([target_hour, target_hour - pd.Timedelta(hours=1)])
    changed = df.copy()
    changed.loc[flip, 'disruption'] = 1 - changed.loc[flip, 'disruption']      # flip outcomes of hours t-1 and t
    after = F.build_features(changed)
    rows = base['timestamp'] == target_hour
    pd.testing.assert_frame_equal(base.loc[rows, HISTORY], after.loc[rows, HISTORY])


def test_features_do_change_when_older_outcomes_change():
    df = synthetic()
    base = F.build_features(df)
    target_hour = pd.Timestamp('2022-04-02 12:00')
    flip = df['timestamp'] == target_hour - pd.Timedelta(hours=2)
    changed = df.copy()
    assert df.loc[flip, 'disruption'].mean() < 1          # so setting them all to 1 really changes the rate
    changed.loc[flip, 'disruption'] = 1                   # make every flight of hour t-2 disrupted
    after = F.build_features(changed)
    rows = base['timestamp'] == target_hour
    assert not np.allclose(base.loc[rows, 'delay_last_1h'], after.loc[rows, 'delay_last_1h'])


def test_history_is_identical_within_an_hour_and_has_expected_schema():
    out = F.build_features(synthetic())
    assert list(out.columns) == F.FEATURE_COLUMNS
    assert out.groupby('timestamp')[HISTORY].nunique().max().max() == 1
    assert out[HISTORY].notna().all().all() and out['delay_last_1h'].between(0, 1).all()
