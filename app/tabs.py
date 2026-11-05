"""The hero row and the five tabs of the dashboard. Each function receives `ctx`, the selections made in the sidebar."""
import numpy as np
import streamlit as st

import charts as ch
import components as cp
import data as dt
import theme as T

E, R, C = dt.E, dt.R, dt.C
CHART = dict(config={'displayModeBar': False}, width='stretch')
RAIN_LEVELS = ['none', 'light', 'moderate', 'heavy']
RAIN_MM = {'none': 0.0, 'light': 1.0, 'moderate': 5.0, 'heavy': 10.0}          # representative values, as predict.py
PRESETS = {
    'Severe storm': dict(vis=0.5, wind=35, rain='heavy', peak=True),
    'Light rain + peak hour': dict(vis=8.0, wind=12, rain='light', peak=True),
    'Clear day': dict(vis=16.0, wind=6, rain='none', temp=20, peak=False),
}


def plot(fig, key=None):
    st.plotly_chart(fig, key=key, **CHART)


# ------------------------------------------------------------------------------------------------ hero row
def hero(ctx):
    pred, sev = ctx.score['pred'], ctx.score['sev']
    prob = pred['probability']
    c1, c2, c3 = st.columns(3)
    with c1.container(key='card_risk'):
        cp.show(cp.risk_hero(prob, pred['risk_level'], prob / ctx.base, bool(pred['disruption']), pred['threshold'], ctx.base))
    with c2.container(key='card_severity'):
        cp.show(cp.severity_hero(sev))
    with c3.container(key='card_resilience'):
        r = ctx.res[ctx.res['airport_code'] == ctx.airport].iloc[0]
        tied = (ctx.res['resilience_index'] == r['resilience_index']).sum() > 1
        others = ', '.join(f"{a} {v:.0f}" for a, v in zip(ctx.res['airport_code'], ctx.res['resilience_index']) if a != ctx.airport)
        txt = (f'Tied with {others}: a two-airport scale cannot separate them' if tied else
               f'Rank {int(ctx.res.index[ctx.res["airport_code"] == ctx.airport][0]) + 1} of {len(ctx.res)} · others: {others}')
        cp.show(cp.resilience_hero(r, txt))


# ------------------------------------------------------------------------------------------------ tab 1
def risk_overview(ctx):
    left, right = st.columns([1, 2.3])
    with left.container(key='card_conditions'):
        u = ctx.score['used']
        cp.card_title('Conditions this hour', 'Weather and recent traffic from the airport observations')
        cp.show(cp.condition_list(u))
        cp.show(f'<div style="margin-top:12px">{cp.flag_chips(u)}</div>')
        cp.show(cp.history_bars(u))
    with right.container(key='card_trend'):
        cp.card_title('Disruption trend', 'Share of flights delayed by more than 15 minutes, per day. The dataset covers January, April, July and October, so the line has gaps.')
        t1, t2, _ = st.columns([1, 1.5, 1.2])
        pred_on = t1.toggle('Model estimate', value=True, help='Average of the model\'s predicted probability per day, shown next to what actually happened.')
        ev_on = t2.toggle('Storm and low-visibility days', value=True, help='Triangles: storm flag (wind above 50 kt, visibility below 1 km or heavy rain). Diamonds: visibility below 3 km.')
        with st.spinner('Building trend...'):
            daily = dt.D.daily_series(ctx.hourly, ctx.airport, ctx.start, ctx.end)
        plot(ch.trend_chart(daily, ctx.snap['date'], pred_on, ev_on), key='trend')

    l2, r2 = st.columns([1.25, 1])
    with l2.container(key='card_shap'):
        cp.card_title('What drives the risk', 'Which factors push the prediction up or down (SHAP values)')
        view = st.radio('SHAP view', ['This hour', 'Overall (all flights)'], horizontal=True, label_visibility='collapsed', key='shap_view')
        if view == 'This hour':
            plot(ch.shap_local(ctx.score['shap'], ctx.score['used']), key='shap_local')
            cp.show('<div class="cl-mut">Red bars raise this hour\'s risk, blue bars lower it. The value of each factor in this hour follows its name.</div>')
        else:
            with st.spinner('Loading feature importance...'):
                imp = dt.global_importance()
            plot(ch.shap_global(imp), key='shap_global')
            cp.show('<div class="cl-mut">Average influence over thousands of test flights. Recent delays and time of day dominate; weather matters less than intuition suggests.</div>')
    with r2.container(key='card_heat'):
        cp.card_title('When disruptions happen', 'Share of flights delayed, by weekday and departure hour')
        plot(ch.hour_heatmap(ctx.rng), key='heat')


# ------------------------------------------------------------------------------------------------ tab 2
def _set(sid_keys, values):
    for k, v in values.items():
        st.session_state[sid_keys[k]] = v


def scenario_tab(ctx) -> dict:
    u = ctx.score['used']
    base = ctx.features
    sid = f'{ctx.airport}_{ctx.snap["timestamp"]:%Y%m%d%H}'
    K = {n: f'sc_{n}_{sid}' for n in ['vis', 'wind', 'rain', 'temp', 'peak', 'mode']}
    base_rain = RAIN_LEVELS[int(u['rain_severity_enc'])]
    d0 = dict(vis=float(min(16.0, round(u['visibility_km'] * 2) / 2)), wind=int(min(70, round(u['wind_speed_knots']))),
              rain=base_rain, temp=int(np.clip(round(u['temperature_c']), -20, 50)), peak=bool(u['is_peak_hour']))
    for n, v in d0.items():
        st.session_state.setdefault(K[n], v)
    saturated = ctx.score['pred']['probability'] > 0.9      # weather cannot move a risk that delays have already pushed to the top
    st.session_state.setdefault(K['mode'], 'Weather only' if saturated else 'Live model')

    left, right = st.columns([1, 1.5])
    with left.container(key='card_controls'):
        cp.card_title('What-if conditions', 'Start from a preset or move the sliders. Everything else stays as in the selected hour.')
        b = st.columns(3)
        for col, (name, vals) in zip(b, PRESETS.items()):
            col.button(name, width='stretch', on_click=_set, args=(K, {**d0, **vals}), key=f'pre_{name}_{sid}')
        st.button('Reset to the selected hour', on_click=_set, args=(K, d0), key=f'reset_{sid}', width='stretch', type='tertiary')
        vis = st.slider('Visibility (km)', 0.0, 16.0, step=0.5, key=K['vis'],
                        help='Aviation Category I minimum is 3 km. Below this, low-visibility (ILS) procedures may be required.')
        wind = st.slider('Wind speed (knots)', 0, 70, step=1, key=K['wind'],
                         help='Calm 0-10 kt, moderate 10-25, strong 25-50, severe above 50. The training data tops out at 35 kt.')
        rain = st.select_slider('Rainfall intensity', RAIN_LEVELS, key=K['rain'],
                                help='None 0 mm/h, light up to 2.5, moderate 2.5-7.5, heavy above 7.5 mm/h.')
        temp = st.slider('Temperature (°C)', -20, 50, step=1, key=K['temp'],
                         help='Heat stress above 35 °C and ice risk below -10 °C add to operational difficulty.')
        peak = st.checkbox('Peak-hour traffic', key=K['peak'], help='Moves the hour into (17:00) or out of (12:00) the 06-09 and 16-20 peak windows.')
        mode_label = st.radio('Model view', ['Live model', 'Weather only'], horizontal=True, key=K['mode'],
                              help='Live model: keeps the recent delay history of the selected hour (the value in the risk card). '
                                   'Weather only: risk from weather and time alone, before delays have built up.')
        mode = 'full' if mode_label == 'Live model' else 'weather'

    overrides = {}
    if vis != d0['vis']:
        overrides['visibility_km'] = float(vis)
    if wind != d0['wind']:
        overrides['wind_speed_knots'] = float(wind)
    if rain != d0['rain']:
        overrides['precip_mm_hr'] = RAIN_MM[rain]
    if temp != d0['temp']:
        overrides['temperature_c'] = float(temp)
    if peak != d0['peak']:
        overrides['is_peak_hour'] = int(peak)

    with st.spinner('Running prediction...'):
        res = dt.run_scenario(base, overrides, mode)
        merged = E.apply_overrides(base, overrides)
        base_s, scen_s = dt.score_conditions(base, mode), dt.score_conditions(merged, mode)
    p0, p1, delta = res['base_probability'], res['scenario_probability'], res['delta']
    l0, l1 = dt.risk_band_of(p0), dt.risk_band_of(p1)

    with right.container(key='card_scenario'):
        cp.card_title('Scenario result', 'Disruption risk before and after the change')
        m1, m2 = st.columns([1.5, 1])
        with m1:
            cp.show(cp.meter_block('Selected hour', p0))
            cp.show('<div style="height:10px"></div>')
            cp.show(cp.meter_block('What-if scenario', p1))
        with m2:
            if abs(delta) < 0.0005:
                st.metric('Scenario risk', f'{p1:.0%}', delta='no change', delta_color='off')
            else:
                st.metric('Scenario risk', f'{p1:.0%}', delta=f'{delta * 100:+.1f} pp', delta_color='inverse')
            lvl = f'{cp.pill(l0, T.RISK_COLORS[l0])} → {cp.pill(l1, T.RISK_COLORS[l1])}' if l0 != l1 else cp.pill(l1, T.RISK_COLORS[l1])
            sev0, sev1 = base_s['sev']['severity'], scen_s['sev']['severity']
            cp.show(f'<div style="margin:2px 0 8px 0">{lvl}</div><div class="cl-sub">Expected severity: <b>{sev1}</b>' + (f' (was {sev0})' if sev0 != sev1 else '') + '</div>')
        if mode == 'weather' and saturated:
            cp.show(cp.note('This hour\'s live risk is already at the top of the model\'s scale because delays have built up, and weather changes cannot raise it further. '
                            'The simulator therefore starts in <b>Weather only</b> view (risk from weather and time before delays accumulate), so the first bar differs from the risk card above.'))
        elif not overrides:
            cp.show(cp.note('Nothing changed yet. Pick a preset or move a slider to see how the risk responds.'))
        if overrides and p0 > 0.9 and abs(delta) < 0.005 and mode == 'full':
            cp.show(cp.note('This hour\'s risk is already at the top of the model\'s scale, driven by delays that have built up. Weather changes cannot raise it further. '
                            'Switch the model view to <b>Weather only</b>, or choose an hour with lower risk, to see the weather effect.', 'warn'))
        extremes = (vis < 1) + (RAIN_MM[rain] >= 7.5) + (wind >= 28)
        if extremes >= 2 or wind > 35:
            cp.show(cp.note('These conditions are more extreme than almost anything in the training data (about 70 flights ever had low visibility together with rain; '
                            'the highest wind was 35 kt). Read the result as an upper bound, not a forecast.', 'warn'))
        if overrides:
            cp.show('<hr class="cl-hr">')
            cp.card_title('What changed the risk', 'Change in each factor\'s push on the prediction, caused by the what-if')
            dshap = {k: scen_s['shap'][k] - base_s['shap'][k] for k in scen_s['shap']}
            plot(ch.shap_local(dshap, scen_s['used'], top=6, height=240), key='shap_delta')

    with st.container(key='card_curves'):
        cp.card_title('How risk responds to each input', 'Other inputs held as in the what-if. The shaded bands are the Low, Medium and High risk levels.')
        a, b2 = st.columns(2)
        vx = tuple(np.round(np.linspace(0.3, 16, 17), 2))
        wx = tuple(np.round(np.linspace(0, 45, 16), 1))
        with st.spinner('Running prediction...'):
            vy = dt.sensitivity(base, {k: v for k, v in overrides.items() if k != 'visibility_km'}, 'visibility_km', vx, mode)
            wy = dt.sensitivity(base, {k: v for k, v in overrides.items() if k != 'wind_speed_knots'}, 'wind_speed_knots', wx, mode)
        with a:
            plot(ch.sensitivity_curve(vx, vy, 'Visibility (km)', (vis, p1), (d0['vis'], p0), reverse_x=True), key='curve_vis')
        with b2:
            plot(ch.sensitivity_curve(wx, wy, 'Wind speed (knots)', (wind, p1), (d0['wind'], p0)), key='curve_wind')
    return {'active': bool(overrides), 'prob': p1, 'sev': scen_s['sev']['severity'], 'shap': scen_s['shap'], 'used': scen_s['used'],
            'overrides': overrides, 'base_prob': p0, 'mode': mode}


# ------------------------------------------------------------------------------------------------ tab 3
def resilience_tab(ctx):
    res = ctx.res
    with st.container(key='card_rank'):
        cp.card_title('Airport resilience ranking', 'Index from 0 (most vulnerable) to 100 (most resilient), with the four inputs it is built from')
        cp.show(cp.ranking_table(res))
    if res['resilience_index'].nunique() == 1:
        cp.show(cp.note('<b>Read this ranking with care.</b> With only two airports the index is rescaled between them, so each input can only be "worse" or "better". '
                        'Here the two airports each win two of the four inputs and both score exactly 50. The underlying numbers below are the real comparison; '
                        'the index needs four or more airports, or fixed reference ranges, before it can rank airports meaningfully.', 'warn'))
    l, r = st.columns(2)
    with l.container(key='card_rbar'):
        cp.card_title('Resilience index', 'Bars are coloured by tier')
        plot(ch.resilience_bar(res), key='rbar')
    with r.container(key='card_rmap'):
        cp.card_title('Where the airports are', 'Colour = resilience index (darker is more resilient), bubble size = disruption rate')
        plot(ch.resilience_map(res), key='rmap')
    with st.container(key='card_dims'):
        cp.card_title('The four inputs behind the index', 'Higher disruption rate, longer delays and higher sensitivities all lower the index')
        plot(ch.dimension_chart(res), key='dims')
        cp.show('<div class="cl-mut"><b>Method.</b> Each input is scaled 0-1 across airports, then combined as vulnerability = 35% disruption rate + 30% average delay '
                '+ 20% weather sensitivity + 15% congestion sensitivity. Resilience index = 100 − 100 × vulnerability. '
                'Tiers: Vulnerable ≤ 40, Moderate 41-70, Resilient above 70. The weights are a design choice (docs/model_report.md).</div>')


# ------------------------------------------------------------------------------------------------ tab 4
def recommendations_tab(ctx, scen: dict):
    opts = ['Selected hour', 'What-if scenario']
    src = st.radio('Base the recommendations on', opts, index=1 if scen['active'] else 0, horizontal=True,
                   help='The what-if option becomes available after you change something in the Scenario Simulator.')
    use_scen = src == opts[1] and scen['active']
    if src == opts[1] and not scen['active']:
        cp.show(cp.note('No what-if scenario is active yet, so the recommendations below describe the selected hour. Change a slider in the Scenario Simulator tab to see them react.'))
    if use_scen:
        prob, sev, shap, used = scen['prob'], scen['sev'], scen['shap'], scen['used']
        who = 'what-if scenario'
    else:
        prob, sev, shap, used = ctx.score['pred']['probability'], ctx.score['sev']['severity'], ctx.score['shap'], ctx.score['used']
        who = 'selected hour'
    top = dt.top_drivers(shap)
    recs = R.build_recommendations(prob, sev, top, dt.conditions_of(used))
    level = R.tier_for(prob)

    l, r = st.columns([1.7, 1])
    with l.container(key='card_recs'):
        cp.card_title(f'Suggested operational actions', f'Based on the {who}')
        cp.show(f'<div style="display:flex;gap:12px;align-items:center;margin:2px 0 4px 0">{cp.pill(level + " risk", T.RISK_COLORS[level])}'
                f'<span class="cl-sub">{prob:.0%} predicted probability · {sev} expected severity</span></div>')
        cp.show(cp.rec_groups(recs))
    with r.container(key='card_why'):
        cp.card_title('What is driving this', f'Factors behind the {who}')
        chips = ''.join(f'<span class="cl-chip">{dt.label(k)} <b>{dt.value_text(k, used[k])}</b></span>' for k in top) or '<span class="cl-sub">No factor pushes the risk up.</span>'
        cp.show(f'<div class="cl-eyebrow">Top factors raising the risk</div>{chips}')
        cond = dt.conditions_of(used)
        cp.show('<hr class="cl-hr">' + f'<div class="cl-eyebrow">Weather</div>' + cp.flag_chips(used) +
                f'<div class="cl-mut" style="margin-top:8px">Visibility {cond["visibility_km"]:.1f} km · wind {cond["wind_speed_knots"]:.0f} kt · rain {cond["precip_mm_hr"]:.1f} mm/h · {cond["temperature_c"]:.0f} °C</div>')
        low, high = C.RISK_BANDS
        cp.show('<hr class="cl-hr">'
                f'<div class="cl-mut"><b>Risk bands:</b> Low below {low:.0%}, Medium {low:.0%}-{high:.0%}, High above {high:.0%}. They are set in one place (config.RISK_BANDS) and the team has not '
                'finalised them (the project report uses 50% / 80%).</div>')
        brief = '\n'.join([f'# CLEARS operations brief — {ctx.airport}', '', f'- Hour analysed: {ctx.snap["timestamp"]:%Y-%m-%d %H:00} (local time)',
                           f'- Basis: {who}', f'- Disruption probability: {prob:.0%} ({level})', f'- Expected severity: {sev}', '', '## Suggested actions']
                          + [f'- [{r_["urgency"].upper()}] {r_["text"]} (why: {r_["trigger"]})' for r_ in recs]
                          + ['', '_Decision support only: the model separates risky hours from calm ones but is far from perfect (ROC-AUC 0.66 on unseen data)._'])
        st.download_button('Download operations brief', brief, file_name=f'clears_brief_{ctx.airport}_{ctx.snap["timestamp"]:%Y%m%d_%H}.md', mime='text/markdown',
                           width='stretch')
    with st.expander('How the recommendations are generated'):
        st.markdown(
            '1. **Risk tier** from the calibrated probability decides the main action set: proactive (High), monitoring (Medium) or nominal (Low).\n'
            '2. **Driver rules** add an action when its factor is among the top SHAP drivers **or** is plainly present in the weather (visibility under 3 km, wind of 25 kt or more, '
            'rain of 2.5 mm/h or more, storm flag, temperature extremes).\n'
            '3. **Severity rules** add passenger-facing actions for moderate or severe expected delays.\n\n'
            'The rules live in `src/recommend.py` and the output changes with the scenario, nothing is hard-coded per airport.')


# ------------------------------------------------------------------------------------------------ tab 5
def model_tab(ctx):
    m = dt.load_metrics()
    t, lr = m['test_tuned_threshold'], m['baselines']['logistic_regression']
    items = [('F1, disrupted class', f"{t['f1_1']:.2f}", f"logistic regression {lr['f1_1']:.2f}"), ('Precision', f"{t['precision_1']:.2f}", 'of flagged flights, share truly delayed'),
             ('Recall', f"{t['recall_1']:.2f}", 'of delayed flights, share flagged'), ('ROC-AUC', f"{t['roc_auc']:.2f}", f"logistic regression {lr['roc_auc']:.2f}"),
             ('PR-AUC', f"{t['pr_auc']:.2f}", f"base rate {m['baselines']['base_rate']:.2f}")]
    cp.show('<div class="cl-kpis">' + ''.join(cp.kpi(a, b, '', T.MUTED, c, cp.ACCENTS[i]) for i, (a, b, c) in enumerate(items)) + '</div>')
    cp.show(cp.note(f'<b>Honest summary.</b> The model ranks risky hours better than chance and better than a simple baseline, but it is a modest signal: the project goal of F1 ≥ 0.70 is not reachable with weather, '
                    f'time and short delay history (a more generous cross-validation reaches F1 {m["diagnostics"]["date_grouped_cv"]["f1_1_at_base_rate_threshold"]:.2f}). '
                    'Use it to prioritise attention, not to predict individual flights.', 'warn'))
    test_h = ctx.hourly[ctx.hourly['period'] == 'test']
    a, b = st.columns(2)
    with a.container(key='card_cal'):
        cp.card_title('Are the probabilities trustworthy?', 'Predicted risk against what actually happened, on test hours the model never saw')
        plot(ch.calibration_chart(test_h), key='cal')
        cp.show('<div class="cl-mut">A predicted 20% really meant about 20% of flights delayed (the points sit close to the diagonal). The highest-risk hours are somewhat over-confident.</div>')
    with b.container(key='card_groups'):
        cp.card_title('What the model relies on', 'Share of the model\'s total influence, by group of inputs')
        plot(ch.group_importance(dt.group_importance(dt.global_importance()), 300), key='groups')
        cp.show('<div class="cl-mut">Recent delays and time of day carry most of the signal. Weather is a real but smaller part, which is why the model separates risky hours only modestly.</div>')
    l, r = st.columns(2)
    with l.container(key='card_dist'):
        cp.card_title('How often each risk level occurs', 'Share of airport-hours in the unseen test period')
        share = test_h['risk_level'].value_counts(normalize=True)
        cp.show(cp.segmented([(k, float(share.get(k, 0)), T.RISK_COLORS[k]) for k in ['Low', 'Medium', 'High']]))
        cp.show('<div class="cl-mut" style="margin-top:12px">Most hours are Low risk. High risk is rare, but about 65% of flights in those hours were delayed.</div>')
    with r.container(key='card_data'):
        cp.card_title('Data and method')
        d = m['data']
        st.markdown(
            f'- **Flights:** BTS On-Time Performance. **Weather:** NOAA ISD hourly observations, matched to the latest observation before each departure.\n'
            f'- **Airports:** Chicago O\'Hare (ORD), Memphis (MEM). **Months:** Jan, Apr, Jul, Oct of 2019-2022.\n'
            f'- **Rows:** {d["train_rows"]:,} training, {d["val_rows"]:,} validation, {d["test_rows"]:,} test flights, split by date so the test set is the future.\n'
            f'- **Label:** departure delay above 15 minutes. Cancelled flights were removed.\n'
            f'- **Model:** gradient-boosted trees (XGBoost) with isotonic probability calibration, plus a separate severity model.')
    with st.container(key='card_limits'):
        cp.card_title('Known limitations')
        a1, a2 = st.columns(2)
        a1.markdown(
            '- Two airports and four months a year: results may not transfer to other airports or to winter operations (the test set has no winter month).\n'
            '- The resilience index is not meaningful with two airports.')
        a2.markdown(
            '- Weather explains less of the delay than recent delays and time of day; extreme combined weather is beyond the training data.\n'
            '- Risk bands and the cancelled-flight policy are open team decisions; there is no live weather feed.')
