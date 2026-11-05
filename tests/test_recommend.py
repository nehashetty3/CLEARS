"""Rule tests for src/recommend.py (T-3.6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import config as C  # noqa: E402
import recommend as R  # noqa: E402

LOW, HIGH = C.RISK_BANDS


def texts(prob, severity='none', top=(), cond=None):
    return R.get_recommendation(prob, severity, list(top), cond)


def test_tiers_follow_config_bands():
    assert R.tier_for(LOW - 0.01) == 'Low' and R.tier_for(LOW) == 'Medium'
    assert R.tier_for(HIGH) == 'Medium' and R.tier_for(HIGH + 0.01) == 'High'


def test_each_tier_has_its_own_actions():
    assert 'Initiate proactive departure delay programme' in texts(HIGH + 0.05)
    assert 'Monitor weather feed every 15 minutes' in texts((LOW + HIGH) / 2)
    assert 'Conditions nominal - continue standard operations' in texts(LOW / 2)
    assert 'Initiate proactive departure delay programme' not in texts(LOW / 2)


def test_high_tier_actions_are_critical_low_are_info():
    assert {r['urgency'] for r in R.build_recommendations(HIGH + 0.1, 'none') if r['kind'] == 'tier'} == {'critical'}
    assert {r['urgency'] for r in R.build_recommendations(0.05, 'none') if r['kind'] == 'tier'} == {'info'}


def test_driver_rules_from_top_features_and_guide_names():
    assert any('low-visibility' in t for t in texts(0.4, top=['visibility_risk']))
    assert any('crosswind' in t for t in texts(0.4, top=['wind_category']))
    assert any('SIGMET' in t for t in texts(0.4, top=['storm_flag']))
    assert any('ground delay programme' in t for t in texts(0.4, top=['congestion_level']))
    assert not any('low-visibility' in t for t in texts(0.4, top=['hour_of_day']))


def test_driver_rules_from_current_weather_when_shap_misses_them():
    cond = {'visibility_km': 1.2, 'wind_speed_knots': 31, 'precip_mm_hr': 6, 'storm_flag': 1}
    out = texts(0.5, top=['delay_last_3h'], cond=cond)
    assert any('low-visibility' in t for t in out) and any('crosswind' in t for t in out)
    assert any('SIGMET' in t for t in out) and any('runway acceptance' in t for t in out)
    clear = {'visibility_km': 16, 'wind_speed_knots': 6, 'precip_mm_hr': 0, 'storm_flag': 0, 'temperature_c': 18}
    assert len(texts(0.5, top=['hour_of_day'], cond=clear)) == len(R.TIER_ACTIONS['Medium'])
    assert any('already building' in t for t in texts(0.5, top=['delay_last_3h'], cond=clear))


def test_known_weather_overrides_a_mild_shap_push():
    mild = {'visibility_km': 16, 'wind_speed_knots': 8, 'precip_mm_hr': 0, 'storm_flag': 0, 'temperature_c': 27}
    out = texts(0.4, top=['temperature_c', 'visibility_km', 'wind_speed_knots', 'precip_mm_hr'], cond=mild)
    assert not any(k in t for t in out for k in ('temperature extremes', 'low-visibility', 'crosswind', 'runway acceptance'))


def test_severity_adds_passenger_actions():
    assert any('rebooking' in t for t in texts(0.4, 'moderate'))
    assert any('irregular-operations' in t for t in texts(0.4, 'severe'))
    assert not any('rebooking' in t for t in texts(0.4, 'minor'))


def test_recommendations_react_to_inputs_and_sort_by_urgency():
    calm = R.build_recommendations(0.1, 'none')
    storm = R.build_recommendations(0.8, 'severe', ['storm_flag'], {'visibility_km': 0.5})
    assert [r['text'] for r in calm] != [r['text'] for r in storm]
    order = {'critical': 0, 'watch': 1, 'info': 2}
    ranks = [order[r['urgency']] for r in storm]
    assert ranks == sorted(ranks)
