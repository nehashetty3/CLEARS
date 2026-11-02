"""Feature engineering for CLEARS (Person 1).

build_features(df) takes the clean merged flight+weather dataframe produced by
data_prep.run_pipeline and returns the 38-column feature matrix documented in
docs/feature_dictionary.md (v2).

Leakage rule for the history features
-------------------------------------
`timestamp` has hourly resolution, so every flight scheduled in the same hour shares one
timestamp. A flight's disrupted / not-disrupted status is known about 15 minutes after its
scheduled departure, i.e. by the end of the NEXT hour. For a flight scheduled in hour t the
history windows therefore end at hour t-2; flights scheduled in hours t-1 and t are never used.
(The v1 version used shift(1) + rolling('1h'), which at hourly resolution only contained
earlier-listed flights from the same hour -- outcomes not yet known at prediction time.)
"""
import numpy as np
import pandas as pd

WIND_BINS = [0, 10, 25, 50, 9999]
WIND_LABELS = ['calm', 'moderate', 'strong', 'severe']
RAIN_BINS = [-0.001, 0, 2.5, 7.5, 9999]
RAIN_LABELS = ['none', 'light', 'moderate', 'heavy']
HUB_ENC = {'large_hub': 1}          # both study airports are large hubs; any other class -> 0
FFILL_LIMIT_HOURS = 12              # how long a last-known disruption rate may be carried forward

FEATURE_COLUMNS = [
    'airport_code', 'timestamp', 'flight_date',
    'wind_speed_knots', 'visibility_km', 'temperature_c', 'humidity_pct', 'precip_mm_hr',
    'wind_category', 'wind_category_enc', 'visibility_risk', 'rain_severity', 'rain_severity_enc',
    'temp_stress', 'storm_flag', 'humidity_stress',
    'hour_of_day', 'day_of_week', 'month', 'is_peak_hour', 'is_weekend', 'is_holiday_season',
    'delay_last_1h', 'delay_last_3h', 'delay_last_24h', 'congestion_level',
    'disruption', 'delay_minutes', 'dep_delay', 'arr_delay', 'cancelled',
    'weather_delay', 'carrier_delay', 'nas_delay', 'late_aircraft_delay',
    'runway_count', 'hub_size', 'hub_size_enc',
]


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['wind_category'] = pd.cut(df['wind_speed_knots'], bins=WIND_BINS, labels=WIND_LABELS,
                                 include_lowest=True).astype(str)
    df['wind_category_enc'] = df['wind_category'].map({l: i for i, l in enumerate(WIND_LABELS)}).astype(int)
    df['visibility_risk'] = (df['visibility_km'] < 3).astype(int)
    df['rain_severity'] = pd.cut(df['precip_mm_hr'], bins=RAIN_BINS, labels=RAIN_LABELS).astype(str)
    df['rain_severity_enc'] = df['rain_severity'].map({l: i for i, l in enumerate(RAIN_LABELS)}).astype(int)
    df['temp_stress'] = ((df['temperature_c'] > 35) | (df['temperature_c'] < -10)).astype(int)
    df['storm_flag'] = ((df['wind_speed_knots'] > 50) | (df['visibility_km'] < 1)
                        | (df['rain_severity'] == 'heavy')).astype(int)
    df['humidity_stress'] = ((df['humidity_pct'] > 90) & (df['temperature_c'] > 28)).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df['timestamp']
    df['hour_of_day'] = ts.dt.hour
    df['day_of_week'] = ts.dt.dayofweek
    df['month'] = ts.dt.month
    df['is_peak_hour'] = (df['hour_of_day'].between(6, 9) | df['hour_of_day'].between(16, 20)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday_season'] = df['month'].isin([3, 10, 11, 12]).astype(int)
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """delay_last_1h/3h/24h and congestion_level, per airport, on an hourly grid (see module docstring).

    - delay_last_1h : disruption rate of flights scheduled in hour t-2
    - delay_last_3h : ... in hours t-4 .. t-2
    - delay_last_24h: ... in hours t-25 .. t-2
    - congestion_level: number of scheduled departures in hours t-2 and t-1
    Windows with no flights carry the last known rate forward (<= FFILL_LIMIT_HOURS), else 0.0.
    """
    df = df.copy()
    out = {c: pd.Series(np.nan, index=df.index) for c in
           ['delay_last_1h', 'delay_last_3h', 'delay_last_24h', 'congestion_level']}
    for _, g in df.groupby('airport_code'):
        hourly = g.groupby('timestamp').agg(n=('disruption', 'size'), d=('disruption', 'sum'))
        grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
        hourly = hourly.reindex(grid, fill_value=0)

        def rate(w):
            d = hourly['d'].rolling(w, min_periods=w).sum().shift(2)
            n = hourly['n'].rolling(w, min_periods=w).sum().shift(2)
            return (d / n.where(n > 0)).ffill(limit=FFILL_LIMIT_HOURS)

        congestion = hourly['n'].rolling(2, min_periods=2).sum().shift(1)
        for name, series in [('delay_last_1h', rate(1)), ('delay_last_3h', rate(3)),
                             ('delay_last_24h', rate(24)), ('congestion_level', congestion)]:
            out[name][g.index] = series.reindex(g['timestamp']).values
    for name, s in out.items():
        df[name] = s.fillna(0.0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged dataframe -> 38-column feature matrix.

    Required input columns: airport_code, timestamp (hourly, local), flight_date, the five raw weather
    columns, dep_delay, arr_delay, cancelled, weather_delay, carrier_delay, nas_delay,
    late_aircraft_delay, disruption, delay_minutes, runway_count, hub_size.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['flight_date'] = pd.to_datetime(df['flight_date'])
    # stable sort keeps the original order of flights inside an hour
    df = df.sort_values(['airport_code', 'timestamp'], kind='mergesort').reset_index(drop=True)
    df = add_weather_features(df)
    df = add_time_features(df)
    df = add_history_features(df)
    df['hub_size_enc'] = df['hub_size'].map(HUB_ENC).fillna(0).astype(int)
    return df[FEATURE_COLUMNS]
