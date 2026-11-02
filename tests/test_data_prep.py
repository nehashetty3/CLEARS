"""Synthetic-data tests for src/data_prep.py (no raw NOAA/BTS files needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import data_prep as dp  # noqa: E402
import features as F  # noqa: E402


def isd_row(station, date_utc, wnd='290,5,N,0046,5', vis='016000,5,N,5', tmp='+0089,5', dew='+0030,5',
            aa1='01,0000,9,5'):
    return {'STATION': station, 'DATE': date_utc, 'WND': wnd, 'VIS': vis, 'TMP': tmp, 'DEW': dew, 'AA1': aa1}


def test_isd_field_parsing():
    s = pd.Series
    assert dp.parse_wind_knots(s(['290,5,N,0046,5'])).iloc[0] == pytest.approx(4.6 * 1.94384)
    assert np.isnan(dp.parse_wind_knots(s(['999,9,9,9999,9'])).iloc[0])
    assert dp.parse_visibility_km(s(['016000,5,N,5'])).iloc[0] == pytest.approx(16.0)
    assert np.isnan(dp.parse_visibility_km(s(['999999,9,9,9'])).iloc[0])
    assert dp.parse_temperature_c(s(['+0089,5'])).iloc[0] == pytest.approx(8.9)
    assert dp.parse_temperature_c(s(['-0306,5'])).iloc[0] == pytest.approx(-30.6)
    assert np.isnan(dp.parse_temperature_c(s(['+9999,9'])).iloc[0])
    assert np.isnan(dp.parse_temperature_c(s(['+0100,3'])).iloc[0])      # erroneous quality flag
    assert dp.parse_precip_mm_hr(s(['03,0075,9,5'])).iloc[0] == pytest.approx(2.5)  # 7.5 mm over 3 h
    assert dp.parse_precip_mm_hr(s([''])).iloc[0] == 0.0
    assert dp.magnus_humidity(s([20.0]), s([20.0])).iloc[0] == pytest.approx(100.0)
    assert 0 <= dp.magnus_humidity(s([30.0]), s([-5.0])).iloc[0] < 30


def test_weather_is_converted_from_utc_to_local(tmp_path):
    # 2019-07-01 23:53 UTC == 18:53 CDT (UTC-5);  2019-01-01 23:53 UTC == 17:53 CST (UTC-6)
    pd.DataFrame([isd_row('72530094846', '2019-07-01T23:53:00'),
                  isd_row('72530094846', '2019-01-01T23:53:00')]).to_csv(tmp_path / 'w.csv', index=False)
    w = dp.load_weather(tmp_path / 'w.csv')
    assert list(w['obs_time']) == [pd.Timestamp('2019-01-01 17:53'), pd.Timestamp('2019-07-01 18:53')]
    assert set(w['airport_code']) == {'ORD'}


def make_inputs(tmp_path):
    pd.DataFrame([isd_row('72530094846', '2019-07-01T22:53:00', tmp='+0100,5'),    # 17:53 local, 10.0 C
                  isd_row('72530094846', '2019-07-01T23:53:00', tmp='+0200,5'),    # 18:53 local, 20.0 C
                  isd_row('72334013893', '2019-07-01T23:53:00', tmp='+0300,5')]    # MEM 18:53 local, 30.0 C
                 ).to_csv(tmp_path / 'w.csv', index=False)
    pd.DataFrame({
        'FL_DATE': ['2019-07-01'] * 5, 'ORIGIN': ['ORD', 'ORD', 'ORD', 'MEM', 'JFK'],
        'CRS_DEP_TIME': [1830, 1900, 2400, 1900, 1900],
        'DEP_DELAY': [5.0, 40.0, 20.0, 0.0, 99.0],
        'ARR_DELAY': [1.0, np.nan, 3.0, 2.0, 5.0], 'CANCELLED': [0, 0, 0, 0, 0],
        'WEATHER_DELAY': [np.nan] * 5, 'CARRIER_DELAY': [np.nan] * 5,
        'NAS_DELAY': [np.nan] * 5, 'LATE_AIRCRAFT_DELAY': [np.nan] * 5}).to_csv(tmp_path / 'f.csv', index=False)
    pd.DataFrame({'airport_code': ['ORD', 'MEM'], 'runway_count': [8, 4],
                  'hub_size': ['large_hub'] * 2}).to_csv(tmp_path / 'ctx.csv', index=False)


def test_flights_loader_and_backward_merge(tmp_path):
    make_inputs(tmp_path)
    flights = dp.load_flights(tmp_path / 'f.csv')
    assert 'JFK' not in set(flights['airport_code'])
    # 2400 rolls to midnight of the next day
    assert (flights['scheduled_departure'] == pd.Timestamp('2019-07-02 00:00')).sum() == 1
    merged = dp.merge_datasets(dp.load_weather(tmp_path / 'w.csv'), flights)
    ord_rows = merged[merged['airport_code'] == 'ORD'].set_index('scheduled_departure')
    # 18:30 flight only sees the 17:53 observation (10 C); the 19:00 flight sees 18:53 (20 C)
    assert ord_rows.loc[pd.Timestamp('2019-07-01 18:30'), 'temperature_c'] == pytest.approx(10.0)
    assert ord_rows.loc[pd.Timestamp('2019-07-01 19:00'), 'temperature_c'] == pytest.approx(20.0)
    # no observation within 1 h before midnight -> NaN until handle_missing
    assert np.isnan(ord_rows.loc[pd.Timestamp('2019-07-02 00:00'), 'temperature_c'])


def test_full_pipeline_then_features(tmp_path):
    make_inputs(tmp_path)
    df = dp.run_pipeline(tmp_path / 'w.csv', tmp_path / 'f.csv', tmp_path / 'out' / 'clean.csv',
                         tmp_path / 'ctx.csv')
    assert df['disruption'].tolist().count(1) == 2 and 'JFK' not in set(df['airport_code'])
    assert df.loc[df['dep_delay'] == 40, 'disruption'].iloc[0] == 1
    assert df['arr_delay'].notna().all() and df[dp.WEATHER_COLS].notna().all().all()
    feats = F.build_features(df)
    assert list(feats.columns) == F.FEATURE_COLUMNS and len(feats) == len(df)


def test_daily_summary_records_are_not_used_as_weather(tmp_path):
    rows = [isd_row('72530094846', '2019-07-01T22:53:00', tmp='+0100,5'),
            isd_row('72530094846', '2019-07-01T23:59:00', wnd='999,9,9,9999,9', vis='999999,9,9,9', tmp='+9999,9',
                    dew='+9999,9', aa1='24,0218,9,5')]            # an SOD record: no weather, 24 h rain total
    rows[0]['REPORT_TYPE'], rows[1]['REPORT_TYPE'] = 'FM-15 ', 'SOD  '
    pd.DataFrame(rows).to_csv(tmp_path / 'w.csv', index=False)
    w = dp.load_weather(tmp_path / 'w.csv')
    assert len(w) == 1 and w['precip_mm_hr'].iloc[0] == 0.0 and w['temperature_c'].iloc[0] == pytest.approx(10.0)


def test_bts_zip_with_readme_member_is_read(tmp_path):
    import zipfile
    flights = pd.DataFrame({'FlightDate': ['2019-07-01'] * 3, 'Origin': ['ORD', 'JFK', 'MEM'],
                            'CRSDepTime': [900, 900, 1500], 'DepDelay': [3.0, 4.0, 20.0], 'Cancelled': [0, 0, 0],
                            'Year': [2019] * 3})                  # an extra column that must be ignored
    with zipfile.ZipFile(tmp_path / 'ontime_2019_7.zip', 'w') as z:
        z.writestr('readme.html', '<html>not a csv</html>')
        z.writestr('On_Time_2019_7.csv', flights.to_csv(index=False))
    out = dp.load_flights(tmp_path)                              # a directory containing the zip
    assert sorted(out['airport_code']) == ['MEM', 'ORD'] and 'Year' not in out.columns


def test_cancelled_policy(tmp_path):
    df = pd.DataFrame({'dep_delay': [np.nan, 5.0, 400.0], 'arr_delay': [np.nan] * 3, 'cancelled': [1.0, 0, 0]})
    dropped = dp.clean_labels(df, 'drop')
    assert len(dropped) == 2 and dropped['delay_minutes'].max() == 300
    kept = dp.clean_labels(df, 'disrupt')
    assert len(kept) == 3 and kept['disruption'].tolist() == [1, 0, 1] and kept['delay_minutes'].iloc[0] == 300
