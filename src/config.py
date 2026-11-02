"""Shared settings for the CLEARS model code (Person 2).

The data file is a setting: DATA_PATH defaults to data/processed/features_final.csv (the dataset regenerated from the raw NOAA/BTS files
by data_prep.py + features.py); set CLEARS_DATA to use any other file. Nothing else needs to change.
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# default: the data regenerated from the raw NOAA/BTS files by data_prep.py + features.py
DATA_PATH = Path(os.environ.get('CLEARS_DATA', ROOT / 'data' / 'processed' / 'features_final.csv'))
CONTEXT_PATH = ROOT / 'data' / 'airport_context.csv'          # runway count, hub size, coordinates (OurAirports)
DOCS_DIR = ROOT / 'docs'
MODEL_DIR = ROOT / 'data' / 'models'
ASSETS_DIR = ROOT / 'assets'

# --- explicit model inputs (docs/feature_dictionary.md, Section 9). NEVER add dep_delay, arr_delay,
# delay_minutes, cancelled or the *_delay cause columns: dep_delay > 15 IS the label.
FEATURES = [
    'wind_speed_knots', 'visibility_km', 'temperature_c', 'humidity_pct', 'precip_mm_hr',
    'wind_category_enc', 'visibility_risk', 'rain_severity_enc', 'temp_stress', 'storm_flag',
    'hour_of_day', 'day_of_week', 'month', 'is_peak_hour', 'is_weekend',
    'delay_last_1h', 'delay_last_3h', 'delay_last_24h', 'congestion_level', 'runway_count',
]
BANNED_FEATURES = ['disruption', 'delay_minutes', 'dep_delay', 'arr_delay', 'cancelled', 'weather_delay',
                   'carrier_delay', 'nas_delay', 'late_aircraft_delay']
assert not set(FEATURES) & set(BANNED_FEATURES)

# --- time-ordered split on whole dates (docs/data_quality_report.md, Check 3)
TEST_START = pd.Timestamp('2022-02-01')      # test = 2022-02-01 onward (Apr/Jul/Oct 2022, no winter)
VAL_START = pd.Timestamp('2021-10-01')       # validation = last part of the training period
                                             # (Oct 2021 + Jan 2022), used for early stopping,
                                             # probability calibration and the decision threshold

# --- severity classes (disrupted flights are binned by delay_minutes; negative delays are class 0)
SEVERITY_LABELS = {0: 'none', 1: 'minor', 2: 'moderate', 3: 'severe'}
SEVERITY_BINS = [15, 45, 120, 99999]         # (15, 45] minor, (45, 120] moderate, > 120 severe

# --- KPI 4 risk bands on the CALIBRATED probability. The team has not settled these (KPI doc: 0.30/0.65,
# report: 0.50/0.80) -- change them here only; predict.py and the dashboard read this constant.
RISK_BANDS = (0.30, 0.65)

RANDOM_STATE = 42
