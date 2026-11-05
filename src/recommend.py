"""Recommendation engine (Person 3, task T-3.6).

    get_recommendation(prob, severity, top_features, conditions=None) -> list[str]        (guide signature)
    build_recommendations(prob, severity, top_features, conditions=None) -> list[dict]    (what the dashboard shows)

Rules
  1. Risk tier from the calibrated probability, using the bands in config.RISK_BANDS (the single place the team
     changes them): High > band[1] -> proactive actions, Medium -> monitoring actions, Low -> nominal.
  2. Driver rules: an action is added when its factor is among the top SHAP drivers of the prediction
     (`top_features`, the guide's rule) OR the factor is plainly present in `conditions` (raw weather of the
     hour, using the thresholds of features.py). The second trigger is needed because the model leans mostly
     on delay history and time of day (docs/model_report.md), so weather features seldom rank in the top SHAP list
     even when the weather itself is bad. When `conditions` is given it decides the weather actions (a mild SHAP
     push from a 27 degree day must not produce a temperature-extreme warning); without it the guide's SHAP-only rule applies.
  3. Severity rules: moderate / severe predicted delays add passenger-facing actions.

The wording of the tier and driver actions follows the task guide; urgency maps to the colours red / amber / blue.
"""
try:
    from . import config as C
except ImportError:                     # pragma: no cover
    import config as C

# feature names that count as the same factor (model columns plus the names used in the guide)
FACTORS = {
    'visibility': {'visibility_risk', 'visibility_km'},
    'wind': {'wind_category', 'wind_category_enc', 'wind_speed_knots'},
    'storm': {'storm_flag'},
    'congestion': {'congestion_level'},
    'rain': {'precip_mm_hr', 'rain_severity', 'rain_severity_enc'},
    'temperature': {'temperature_c', 'temp_stress'},
    'delay_history': {'delay_last_1h', 'delay_last_3h', 'delay_last_24h'},
}
WEATHER_FACTORS = {'visibility', 'wind', 'storm', 'rain', 'temperature'}
ACTIONS = {
    'visibility': 'Coordinate low-visibility procedures with ATC - ILS Category II/III may be required',
    'wind': 'Review crosswind component limits for active runways - consider runway change if sustained gusts exceed 25kt',
    'storm': 'Issue SIGMET advisory to connected airlines if storm flag active',
    'congestion': 'Implement ground delay programme to reduce airborne stack congestion',
    'rain': 'Expect reduced runway acceptance rates in heavy rain - review spacing and braking-action reports',
    'temperature': 'Prepare for temperature extremes - de-icing capacity in the cold, apron heat and equipment checks in the heat',
    'delay_history': 'Departure delays are already building - review gate-hold and turnaround plans to stop them cascading',
}
TIER_ACTIONS = {
    'High': ['Initiate proactive departure delay programme', 'Alert all gate agents of imminent disruption',
             'Pre-position ground crew for recovery operations', 'Notify inbound airlines of potential slot disruptions'],
    'Medium': ['Monitor weather feed every 15 minutes', 'Prepare staffing buffers for departure terminals',
               'Review inbound aircraft positions and estimated arrival times'],
    'Low': ['Conditions nominal - continue standard operations', 'Schedule recommended: no proactive action required'],
}
SEVERITY_ACTIONS = {
    'moderate': 'Prepare passenger communications and rebooking support for delays of 45 minutes or more',
    'severe': 'Brief duty managers to expect delays above 2 hours and activate the irregular-operations plan',
}
URGENCY = {'High': 'critical', 'Medium': 'watch', 'Low': 'info'}      # colours: red / amber / blue


def tier_for(prob: float) -> str:
    low, high = C.RISK_BANDS
    return 'Low' if prob < low else ('High' if prob > high else 'Medium')


def _conditions_trigger(factor: str, c: dict) -> bool:
    """Is this factor plainly present in the raw weather of the hour? Thresholds as in features.py."""
    g = c.get
    if factor == 'visibility':
        return g('visibility_km', 99) < 3
    if factor == 'wind':
        return g('wind_speed_knots', 0) >= 25
    if factor == 'storm':
        return bool(g('storm_flag', 0))
    if factor == 'rain':
        return g('precip_mm_hr', 0) >= 2.5
    if factor == 'temperature':
        return g('temperature_c', 15) > 35 or g('temperature_c', 15) < -10
    return False


def build_recommendations(prob: float, severity: str, top_features=(), conditions: dict = None) -> list:
    """Ordered list of {'text', 'urgency' ('critical'|'watch'|'info'), 'kind' ('tier'|'driver'|'severity'),
    'trigger'} dicts, most urgent first."""
    tier = tier_for(prob)
    urgency = URGENCY[tier]
    top = set(top_features or ())
    cond = conditions or {}
    out = [{'text': t, 'urgency': urgency, 'kind': 'tier', 'trigger': f'{tier} risk ({prob:.0%})'}
           for t in TIER_ACTIONS[tier]]
    for factor, names in FACTORS.items():
        in_top = bool(top & names)
        present = _conditions_trigger(factor, cond)
        if factor in WEATHER_FACTORS and cond:
            fire = present                              # the weather itself decides when it is known (a SHAP push alone can be a mild effect)
        else:
            fire = in_top or present
        if fire:
            why = 'present in the current weather' if present else 'top model driver'
            out.append({'text': ACTIONS[factor], 'urgency': 'watch' if tier == 'Low' else urgency, 'kind': 'driver',
                        'trigger': f'{factor.replace("_", " ")}: {why}'})
    if severity in SEVERITY_ACTIONS:
        out.append({'text': SEVERITY_ACTIONS[severity], 'urgency': 'critical' if severity == 'severe' else 'watch',
                    'kind': 'severity', 'trigger': f'predicted severity: {severity}'})
    rank = {'critical': 0, 'watch': 1, 'info': 2}
    return sorted(out, key=lambda r: rank[r['urgency']])


def get_recommendation(prob: float, severity: str, top_features: list, conditions: dict = None) -> list:
    """Plain list of action strings (interface from the task guide)."""
    return [r['text'] for r in build_recommendations(prob, severity, top_features, conditions)]
