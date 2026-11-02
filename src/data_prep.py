"""Data loading, cleaning and merging for CLEARS (Person 1).

Pipeline: load_weather -> load_flights -> merge_datasets -> handle_missing -> clean_labels
          -> add_airport_context -> run_pipeline (writes data/processed/clean_data.csv)

Changes versus the v1 data (see docs/data_quality_report.md, Check 8):
  * NOAA ISD timestamps are UTC. They are converted to US Central local time BEFORE the merge; BTS
    scheduled departure times are local. v1 data shows weather ~5-6 h out of sync with the flights.
  * Weather is matched to each flight with direction='backward' on the scheduled minute: the flight
    gets the latest observation made at or before its scheduled departure (never a later one).
  * Cancelled flights: policy is explicit (`cancelled_policy`), see clean_labels.

Validated on the real raw files (8 NOAA ISD files, 16 BTS monthly zips; docs/data_quality_report.md, Checks 8, 12
and 13): every BTS flight is kept exactly once, and after the UTC shift the weather values equal those of the
original delivery in 96-98% of hours. The logic is also covered by synthetic-data tests (tests/test_data_prep.py).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'

LOCAL_TZ = 'America/Chicago'                       # ORD and MEM are both US Central
STATION_TO_AIRPORT = {'72530094846': 'ORD', '72334013893': 'MEM'}
AIRPORTS = sorted(STATION_TO_AIRPORT.values())
SUMMARY_REPORT_TYPES = {'SOD', 'SOM', 'SOS'}          # ISD daily / monthly / synoptic summaries, not observations
MS_TO_KNOTS = 1.94384
BAD_QUALITY = {'2', '3', '6', '7'}                # ISD quality codes: suspect / erroneous values
WEATHER_COLS = ['wind_speed_knots', 'visibility_km', 'temperature_c', 'humidity_pct', 'precip_mm_hr']
MAX_WEATHER_FFILL = pd.Timedelta('6h')            # do not carry weather forward further than this
DELAY_CAP_MINUTES = 300
DISRUPTION_THRESHOLD_MIN = 15


# ----------------------------------------------------------------------------- ISD parsing
def _part(series: pd.Series, i: int) -> pd.Series:
    # a missing / short field (e.g. an empty AA1) yields '' instead of NaN so the .str methods keep working
    return series.fillna('').astype(str).str.split(',').str[i].fillna('')


def _scaled(value: pd.Series, quality: pd.Series, missing: str, scale: float) -> pd.Series:
    """ISD numeric field -> float, NaN where the missing code or a bad quality flag is present."""
    v = pd.to_numeric(value.str.replace('+', '', regex=False), errors='coerce')
    v = v.mask(value.str.lstrip('+') == missing)
    v = v.mask(quality.isin(BAD_QUALITY))
    return v / scale


def parse_wind_knots(wnd: pd.Series) -> pd.Series:
    """WND = dir,dirQ,type,speed(m/s x10),speedQ. Missing speed 9999."""
    ms = _scaled(_part(wnd, 3), _part(wnd, 4), '9999', 10)
    return ms * MS_TO_KNOTS


def parse_visibility_km(vis: pd.Series) -> pd.Series:
    """VIS = distance(m),distQ,variability,varQ. Missing distance 999999."""
    return _scaled(_part(vis, 0), _part(vis, 1), '999999', 1000)


def parse_temperature_c(tmp: pd.Series) -> pd.Series:
    """TMP / DEW = value(degC x10),quality. Missing value 9999."""
    return _scaled(_part(tmp, 0), _part(tmp, 1), '9999', 10)


def magnus_humidity(temp_c: pd.Series, dew_c: pd.Series) -> pd.Series:
    """Relative humidity (%) from temperature and dewpoint (Magnus), clipped to [0, 100]."""
    rh = 100 * np.exp(17.625 * dew_c / (243.04 + dew_c)) / np.exp(17.625 * temp_c / (243.04 + temp_c))
    return rh.clip(0, 100)


def parse_precip_mm_hr(aa1: pd.Series) -> pd.Series:
    """AA1 = periodHours,depth(mm x10),condition,quality -> mm per hour. No report -> 0 (dictionary rule).

    Only the first liquid-precipitation group (AA1) is used."""
    period = pd.to_numeric(_part(aa1, 0), errors='coerce')
    period = period.mask((period <= 0) | (period == 99))
    depth_mm = _scaled(_part(aa1, 1), _part(aa1, 3), '9999', 10)
    return (depth_mm / period).fillna(0.0)


# ----------------------------------------------------------------------------- loaders
def _read_csvs(path, keep=None, chunksize=None, **kwargs) -> pd.DataFrame:
    """Read one CSV / zip, a list of them, or every *.csv and *.zip in a directory.

    BTS monthly zips hold ALL US flights (~600k rows x 100+ columns), so large files can be read in chunks
    (`chunksize`) and reduced on the fly with `keep(chunk) -> chunk`."""
    if isinstance(path, (list, tuple)):
        files = [Path(p) for p in path]
    else:
        path = Path(path)
        files = sorted(list(path.glob('*.csv')) + list(path.glob('*.zip'))) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f'no CSV or zip files found at {path}')
    frames = []
    for f in files:
        # BTS zips contain the CSV plus a readme.html; pandas refuses multi-member zips, so open the CSV member
        opener = zipfile.ZipFile(f) if f.suffix.lower() == '.zip' else None
        try:
            source = opener.open(next(n for n in opener.namelist() if n.lower().endswith('.csv'))) if opener else f
            if chunksize:
                frames += [keep(c) if keep else c for c in pd.read_csv(source, chunksize=chunksize, **kwargs)]
            else:
                frames.append(pd.read_csv(source, **kwargs))
        finally:
            if opener:
                opener.close()
    return pd.concat(frames, ignore_index=True)


def load_weather(filepath) -> pd.DataFrame:
    """NOAA ISD global-hourly CSV(s) -> tidy weather observations in LOCAL time.

    Output columns: airport_code, obs_time (naive US Central), and the five weather columns."""
    raw = _read_csvs(filepath, dtype={'STATION': str}, low_memory=False)
    raw.columns = [c.upper() for c in raw.columns]
    raw['STATION'] = raw['STATION'].str.strip()
    raw = raw[raw['STATION'].isin(STATION_TO_AIRPORT)].copy()
    if 'REPORT_TYPE' in raw:
        # SOD / SOM / SOS are daily / monthly / synoptic-summary records: every field but a 24 h rain total is
        # missing, so matching one to a flight would inject fake rain and empty weather. Keep real observations
        # (FM-15 METAR, FM-16 SPECI, FM-12 SYNOP, SY-MT, ...).
        raw = raw[~raw['REPORT_TYPE'].str.strip().isin(SUMMARY_REPORT_TYPES)].copy()

    out = pd.DataFrame({'airport_code': raw['STATION'].map(STATION_TO_AIRPORT)})
    # ISD DATE is UTC -> convert to local time before anything is joined to BTS local times
    utc = pd.to_datetime(raw['DATE'], utc=True)
    out['obs_time'] = utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    out['wind_speed_knots'] = parse_wind_knots(raw['WND'])
    out['visibility_km'] = parse_visibility_km(raw['VIS'])
    out['temperature_c'] = parse_temperature_c(raw['TMP'])
    dew = parse_temperature_c(raw['DEW']) if 'DEW' in raw else pd.Series(np.nan, index=raw.index)
    out['humidity_pct'] = magnus_humidity(out['temperature_c'], dew)
    out['precip_mm_hr'] = parse_precip_mm_hr(raw['AA1']) if 'AA1' in raw else 0.0

    out = out.dropna(subset=['obs_time'])
    # an observation with no wind, visibility and temperature at all carries no weather information
    out = out.dropna(subset=['wind_speed_knots', 'visibility_km', 'temperature_c'], how='all')
    return out.sort_values(['airport_code', 'obs_time']).reset_index(drop=True)


_FLIGHT_COLUMNS = {   # normalised BTS header (lower case, no underscores) -> our name
    'fldate': 'flight_date', 'flightdate': 'flight_date',
    'origin': 'airport_code',
    'crsdeptime': 'crs_dep_time',
    'depdelay': 'dep_delay', 'arrdelay': 'arr_delay', 'cancelled': 'cancelled',
    'weatherdelay': 'weather_delay', 'carrierdelay': 'carrier_delay', 'nasdelay': 'nas_delay',
    'lateaircraftdelay': 'late_aircraft_delay',
}


def load_flights(filepath) -> pd.DataFrame:
    """BTS On-Time Performance CSV(s) -> departures from the study airports.

    Output: airport_code, flight_date, scheduled_departure (local, minute precision), timestamp
    (scheduled hour), dep_delay, arr_delay, cancelled and the four delay-cause columns."""
    def normalise(c):
        return c.replace('_', '').lower()

    def keep_study_airports(chunk):                       # reduce each chunk before it is stored
        chunk.columns = [normalise(c) for c in chunk.columns]
        return chunk[chunk['origin'].isin(AIRPORTS)]

    raw = _read_csvs(filepath, keep=keep_study_airports, chunksize=200_000,
                     usecols=lambda c: normalise(c) in _FLIGHT_COLUMNS, low_memory=False)
    raw = raw.rename(columns={c: _FLIGHT_COLUMNS[c] for c in raw.columns if c in _FLIGHT_COLUMNS})
    missing = {'flight_date', 'airport_code', 'crs_dep_time', 'dep_delay'} - set(raw.columns)
    if missing:
        raise ValueError(f'flight file is missing columns: {sorted(missing)}')

    df = raw[raw['airport_code'].isin(AIRPORTS)].copy()
    df['flight_date'] = pd.to_datetime(df['flight_date']).dt.normalize()
    hhmm = pd.to_numeric(df['crs_dep_time'], errors='coerce')
    df = df[hhmm.notna()].copy()
    hhmm = hhmm[hhmm.notna()].astype(int)
    # BTS uses 2400 for midnight -> hour 24 rolls into the next day
    df['scheduled_departure'] = (df['flight_date'] + pd.to_timedelta(hhmm // 100, unit='h')
                                 + pd.to_timedelta(hhmm % 100, unit='m'))
    df['timestamp'] = df['scheduled_departure'].dt.floor('h')
    for c in ['arr_delay', 'cancelled', 'weather_delay', 'carrier_delay', 'nas_delay', 'late_aircraft_delay']:
        if c not in df:
            df[c] = np.nan
    cols = ['airport_code', 'flight_date', 'scheduled_departure', 'timestamp', 'dep_delay', 'arr_delay',
            'cancelled', 'weather_delay', 'carrier_delay', 'nas_delay', 'late_aircraft_delay']
    return df[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------- merge / clean
def merge_datasets(weather_df: pd.DataFrame, flights_df: pd.DataFrame,
                   tolerance: str = '1h', direction: str = 'backward') -> pd.DataFrame:
    """Attach to each flight the latest weather observation at or before its scheduled departure.

    direction='backward' never uses an observation made after the scheduled time. (The task guide
    suggests 'nearest'; that can pull in an observation up to ~30 min later than the flight.)"""
    flights = flights_df.sort_values('scheduled_departure').reset_index(drop=True)
    weather = weather_df.sort_values('obs_time').reset_index(drop=True)
    merged = pd.merge_asof(flights, weather, left_on='scheduled_departure', right_on='obs_time',
                           by='airport_code', tolerance=pd.Timedelta(tolerance), direction=direction)
    return merged.drop(columns='obs_time')


def _limited_ffill(df: pd.DataFrame, col: str) -> pd.Series:
    """Forward fill within each airport, but only from observations at most MAX_WEATHER_FFILL old."""
    ts = df['scheduled_departure']
    filled = df.groupby('airport_code')[col].ffill()
    last_valid = ts.where(df[col].notna()).groupby(df['airport_code']).ffill()
    too_old = (ts - last_valid) > MAX_WEATHER_FFILL
    return filled.mask(too_old & df[col].isna())


def handle_missing(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Fill weather gaps (time-limited forward fill, then column median) and handle delay nulls.

    Rows whose departure delay is null are kept here (cancelled flights); clean_labels decides what to
    do with them according to `cancelled_policy`."""
    df = df.sort_values(['airport_code', 'scheduled_departure']).reset_index(drop=True)
    for col in WEATHER_COLS:
        before = int(df[col].isna().sum())
        df[col] = _limited_ffill(df, col)
        still = int(df[col].isna().sum())
        df[col] = df[col].fillna(df[col].median())
        if verbose:
            print(f'{col}: {before} missing after merge, {still} after time-limited ffill, 0 after median')
    for col in ['weather_delay', 'carrier_delay', 'nas_delay', 'late_aircraft_delay']:
        df[col] = df[col].fillna(0.0)
    df['cancelled'] = df['cancelled'].fillna(0.0)
    return df


def clean_labels(df: pd.DataFrame, cancelled_policy: str = 'drop') -> pd.DataFrame:
    """Add `disruption` (dep_delay > 15) and `delay_minutes` (dep_delay capped at 300).

    cancelled_policy:
      'drop'    : rows with null dep_delay are removed (what the v1 data did; understates disruption
                  because most cancellations vanish).
      'disrupt' : cancelled flights are kept, labelled disruption = 1 and given delay_minutes = 300.
    Either way arr_delay nulls are filled with dep_delay (documented dictionary rule)."""
    if cancelled_policy not in ('drop', 'disrupt'):
        raise ValueError("cancelled_policy must be 'drop' or 'disrupt'")
    df = df.copy()
    no_delay = df['dep_delay'].isna()
    if cancelled_policy == 'drop':
        df = df[~no_delay].copy()
        df['disruption'] = (df['dep_delay'] > DISRUPTION_THRESHOLD_MIN).astype(int)
        df['delay_minutes'] = df['dep_delay'].clip(upper=DELAY_CAP_MINUTES)
    else:
        df['disruption'] = ((df['dep_delay'] > DISRUPTION_THRESHOLD_MIN) | (df['cancelled'] == 1)
                            | no_delay).astype(int)
        df['delay_minutes'] = df['dep_delay'].clip(upper=DELAY_CAP_MINUTES)
        df.loc[df['disruption'].eq(1) & df['dep_delay'].isna(), 'delay_minutes'] = DELAY_CAP_MINUTES
    df['arr_delay'] = df['arr_delay'].fillna(df['dep_delay'])
    return df


def add_airport_context(df: pd.DataFrame, context_path) -> pd.DataFrame:
    """Add runway_count and hub_size from airport_context.csv (OurAirports)."""
    ctx = pd.read_csv(context_path)[['airport_code', 'runway_count', 'hub_size']]
    return df.merge(ctx, on='airport_code', how='left')


def run_pipeline(weather_path=None, flights_path=None, output_path=None, context_path=None,
                 cancelled_policy: str = 'drop') -> pd.DataFrame:
    """Raw NOAA + BTS files -> clean merged dataframe saved to data/processed/clean_data.csv.

    Paths default to the repository layout (relative to this file), never to absolute paths."""
    weather_path = weather_path or RAW_DIR / 'weather'
    flights_path = flights_path or RAW_DIR / 'flights'
    output_path = Path(output_path or PROCESSED_DIR / 'clean_data.csv')
    context_path = context_path or ROOT / 'data' / 'airport_context.csv'

    weather = load_weather(weather_path)
    flights = load_flights(flights_path)
    merged = merge_datasets(weather, flights)
    df = clean_labels(handle_missing(merged, verbose=True), cancelled_policy)
    df = add_airport_context(df, context_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'wrote {len(df):,} rows to {output_path}')
    return df


if __name__ == '__main__':
    run_pipeline()
