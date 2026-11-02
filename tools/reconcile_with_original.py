"""Reproduce the evidence in docs/data_quality_report.md (Checks 8 and 13): compare the regenerated dataset with
the original delivery kept in archive/original_delivery/.

    python tools/reconcile_with_original.py

Prints (1) row counts, (2) how many original rows have no counterpart in the regenerated (BTS) data and whether
they are duplicated flight records, (3) the lag at which the original weather matches the regenerated weather
(the original is UTC-labelled: 6 h in January, 5 h in April / July / October).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import config as C  # noqa: E402

ORIGINAL = C.ROOT / 'archive' / 'original_delivery' / 'features_final.csv'
o = pd.read_csv(ORIGINAL, low_memory=False, parse_dates=['timestamp'])
r = pd.read_csv(C.DATA_PATH, low_memory=False, parse_dates=['timestamp'])

print('=== 1. rows (original -> regenerated)')
print(pd.DataFrame({'original': o.airport_code.value_counts(), 'regenerated': r.airport_code.value_counts()}))

print('\n=== 2. original rows without a counterpart in the regenerated data')
key = ['airport_code', 'timestamp', 'dep_delay', 'arr_delay']
for d in (o, r):
    d['_n'] = d.groupby(key).cumcount()
m = o.merge(r[key + ['_n']].assign(_in=1), on=key + ['_n'], how='left')
extra = int(m['_in'].isna().sum())
print('original rows not found in regenerated data:', extra, '| regenerated rows not found in original:', int(len(r) - m['_in'].notna().sum()))
rec = key + ['weather_delay', 'carrier_delay', 'nas_delay', 'late_aircraft_delay', 'cancelled']
dup_o, dup_r = len(o) - len(o.drop_duplicates(rec)), len(r) - len(r.drop_duplicates(rec))
print(f'rows repeating an identical flight record: original {dup_o}, regenerated {dup_r} -> excess {dup_o - dup_r}'
      f' (equals the extra rows: {dup_o - dup_r == extra})')

print('\n=== 3. original weather = regenerated weather shifted by k hours (UTC label)')
hw = lambda df: df.drop_duplicates(['airport_code', 'timestamp']).set_index(['airport_code', 'timestamp']).temperature_c
O, R = hw(o), hw(r)
for month in (1, 4, 7, 10):
    om, res = O[O.index.get_level_values(1).month == month], {}
    for k in (-7, -6, -5, -4):
        rs = R.copy()
        rs.index = pd.MultiIndex.from_arrays([rs.index.get_level_values(0), rs.index.get_level_values(1) - pd.Timedelta(hours=k)])
        j = pd.concat([om.rename('o'), rs.rename('r')], axis=1).dropna()
        res[k] = round(j.o.corr(j.r), 4)
    print(f'month {month:2d}: best lag {max(res, key=res.get)} h  (correlations {res})')
