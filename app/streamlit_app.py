"""CLEARS dashboard (Person 3, tasks T-3.2 - T-3.8).  Run from the project root:

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))          # app modules
st.set_page_config(page_title='CLEARS Dashboard', page_icon='✈️', layout='wide', initial_sidebar_state='auto')

import components as cp  # noqa: E402
import data as dt  # noqa: E402
import tabs  # noqa: E402
import theme  # noqa: E402

theme.inject(st)

try:
    hourly = dt.load_scored()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

C = dt.C
airports = sorted(hourly['airport_code'].unique())
names = dt.airport_names()
res = dt.load_resilience()
dmin, dmax = hourly['date'].min().date(), hourly['date'].max().date()
PERIODS = {'Test period (unseen data)': (C.TEST_START.date(), dmax),
           'Validation period': (C.VAL_START.date(), (C.TEST_START - pd.Timedelta(days=1)).date()),
           'Training period': (dmin, (C.VAL_START - pd.Timedelta(days=1)).date()),
           'All data': (dmin, dmax), 'Custom range': None}

# ------------------------------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.markdown('<div class="cl-side-title">Airport</div>', unsafe_allow_html=True)
airport = sb.selectbox('Airport', airports, index=airports.index('ORD') if 'ORD' in airports else 0,
                       format_func=lambda a: f'{a} · {names[a][1]}', label_visibility='collapsed',
                       help='Which airport to analyse. The dataset covers Chicago O\'Hare and Memphis.')
sb.markdown('<div class="cl-side-title">Date range</div>', unsafe_allow_html=True)
period = sb.selectbox('Period', list(PERIODS), label_visibility='collapsed',
                      help='The model was trained on the training period and tuned on the validation period. The test period is data it has never seen, '
                           'so risk estimates there are genuine out-of-sample predictions.')
if PERIODS[period] is None:
    picked = sb.date_input('Dates', value=(C.TEST_START.date(), dmax), min_value=dmin, max_value=dmax, label_visibility='collapsed')
    start, end = (picked[0], picked[-1]) if isinstance(picked, (tuple, list)) and len(picked) else (dmin, dmax)
else:
    start, end = PERIODS[period]
rng = dt.slice_range(hourly, airport, start, end)
if rng.empty:
    st.warning('There is no data for this airport in the selected period. The dataset covers January, April, July and October of 2019-2022.')
    st.stop()

sb.markdown('<div class="cl-side-title">Hour to analyse</div>', unsafe_allow_html=True)
mode = sb.radio('Hour', ['Highest-risk hour in range', 'Pick a date and hour'], label_visibility='collapsed',
                help='The gauge, drivers, scenario and recommendations all describe one airport-hour.')
date = hour = None
if mode == 'Pick a date and hour':
    avail = sorted(rng['date'].dt.date.unique())
    top_day = rng.sort_values('probability').iloc[-1]['date'].date()
    date = sb.date_input('Date', value=top_day, min_value=min(avail), max_value=max(avail), key=f'd_{airport}_{start}_{end}')
    day = rng[rng['date'].dt.date == date]
    if day.empty:
        sb.caption('No data on that date (the dataset covers four months a year); showing the nearest hour of the highest-risk day.')
    hour = sb.select_slider('Hour of day', options=list(range(24)), value=14, format_func=lambda h: f'{h:02d}:00')
snap = dt.pick_snapshot(hourly, airport, start, end, 'pick' if mode == 'Pick a date and hour' else 'top', date, hour)

sb.markdown('<div class="cl-side-title">About</div>', unsafe_allow_html=True)
sb.caption(f'Data: {dmin:%b %Y} to {dmax:%b %Y}, {len(hourly):,} airport-hours, {int(hourly["n_flights"].sum()):,} flights. '
           'Predictions come from the saved XGBoost models; the dashboard does not retrain anything.')

# ------------------------------------------------------------------------------------------ page
features = dt.feature_dict(snap)
with st.spinner('Running prediction...'):
    score = dt.score_conditions(features, 'full')
ctx = SimpleNamespace(hourly=hourly, airport=airport, start=start, end=end, rng=rng, snap=snap, features=features, score=score, res=res,
                      base=dt.base_rate(hourly, airport))

cp.header(airport, names[airport][0], names[airport][1], snap['timestamp'], snap['period'], int(snap['n_flights']))
tabs.hero(ctx)
cp.kpi_strip(dt.kpis(hourly, airport, start, end), float(res.loc[res['airport_code'] == airport, 'weather_sensitivity'].iloc[0]), airport,
             (pd.Timestamp(end) - pd.Timestamp(start)).days + 1)

t1, t2, t3, t4, t5 = st.tabs(['Risk Overview', 'Scenario Simulator', 'Resilience Rankings', 'Recommendations', 'Model & Data'])
with t1:
    tabs.risk_overview(ctx)
with t2:
    scen = tabs.scenario_tab(ctx)
with t3:
    tabs.resilience_tab(ctx)
with t4:
    tabs.recommendations_tab(ctx, scen)
with t5:
    tabs.model_tab(ctx)

st.markdown('<div class="cl-sub" style="text-align:center;margin-top:28px">CLEARS · decision support for weather-driven airport disruption · '
            'predictions are probabilities, not guarantees</div>', unsafe_allow_html=True)
