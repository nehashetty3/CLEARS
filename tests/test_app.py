"""End-to-end tests of the Streamlit dashboard (T-3.2 - T-3.8), run headlessly with streamlit.testing."""
import datetime as dt
from pathlib import Path

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / 'app' / 'streamlit_app.py')


def run_app():
    return AppTest.from_file(APP, default_timeout=180).run()


def pick_hour(at, airport='ORD', day=dt.date(2022, 7, 11), hour=14):
    at.sidebar.selectbox[0].set_value(airport)
    at.sidebar.radio[0].set_value('Pick a date and hour')
    at = at.run()
    at.sidebar.date_input[0].set_value(day)
    at.sidebar.select_slider[0].set_value(hour)
    return at.run()


def test_dashboard_loads_with_all_tabs_and_no_errors():
    at = run_app()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ['Risk Overview', 'Scenario Simulator', 'Resilience Rankings', 'Recommendations', 'Model & Data']


def test_both_airports_render():
    at = run_app()
    for airport in ['MEM', 'ORD']:
        at.sidebar.selectbox[0].set_value(airport)
        at = at.run()
        assert not at.exception, airport


def test_scenario_preset_raises_risk_and_recommendations_react():
    at = pick_hour(run_app())
    assert not at.exception
    before = [m for m in at.metric if m.label == 'Scenario risk'][0]
    assert before.delta in ('no change', '+0.0 pp')
    at.button(key='pre_Severe storm_ORD_2022071114').click()
    at = at.run()
    assert not at.exception
    after = [m for m in at.metric if m.label == 'Scenario risk'][0]
    assert float(after.value.strip('%')) > float(before.value.strip('%')) + 20      # about 30% -> 75% in the verified data
    assert after.delta.startswith('+')
    html = ' '.join(m.value for m in at.markdown)
    assert 'Initiate proactive departure delay programme' in html                  # High tier reached through the scenario


def test_reset_returns_to_selected_hour():
    at = pick_hour(run_app())
    at.button(key='pre_Severe storm_ORD_2022071114').click()
    at = at.run()
    at.button(key='reset_ORD_2022071114').click()
    at = at.run()
    after = [m for m in at.metric if m.label == 'Scenario risk'][0]
    assert after.delta == 'no change'


def test_custom_range_and_empty_selection_do_not_crash():
    at = run_app()
    at.sidebar.selectbox[1].set_value('Training period')
    at = at.run()
    assert not at.exception
