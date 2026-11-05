"""HTML building blocks for the dashboard (all strings go through theme.html so markdown keeps them intact)."""
import streamlit as st

import data as dt
import theme as T
from theme import html, icon


def show(s: str) -> None:
    st.markdown(html(s), unsafe_allow_html=True)


def pill(text: str, color: str, bg: str = None) -> str:
    bg = bg or {T.GOOD: T.GOOD_BG, T.WARN: T.WARN_BG, T.CRIT: T.CRIT_BG}.get(color, T.INFO_BG)
    return f'<span class="cl-pill" style="background:{bg}"><i style="background:{color}"></i>{text}</span>'


def heading(title: str, sub: str = None) -> str:
    return f'<div class="cl-h">{title}</div>' + (f'<div class="cl-s">{sub}</div>' if sub else '<div class="cl-s" style="margin:0 0 8px 0"></div>')


def card_title(title: str, sub: str = None) -> None:
    show(heading(title, sub))


def note(text: str, kind: str = 'info') -> str:
    cls = {'info': '', 'warn': ' warn', 'bad': ' bad'}[kind]
    ico = {'info': 'info', 'warn': 'alert', 'bad': 'alert'}[kind]
    return f'<div class="cl-note{cls}">{icon(ico, 18)}<div>{text}</div></div>'


def header(code: str, name: str, city: str, ts, period: str, n_flights: int) -> None:
    badge = {'test': ('Unseen test data', T.GOOD), 'validation': ('Validation period', T.BLUE), 'train': ('Training period', T.WARN)}[period]
    show(f"""
    <div class="cl-header">
      <div class="cl-brand">
        <div class="cl-logo">{icon('plane', 22, '#fff')}</div>
        <div><div class="cl-title"><em>CLEARS</em> &nbsp;Airport Resilience Dashboard</div>
        <div class="cl-tag">Climate-Linked Evaluation and Analytics for Resilience in Airport Systems</div></div>
      </div>
      <div class="cl-headchips">
        <span class="cl-chip"><b>Airport</b> {code} · {city}</span>
        <span class="cl-chip"><b>Hour</b> {ts:%a %d %b %Y, %H:00}</span>
        <span class="cl-chip"><b>Flights</b> {n_flights}</span>
        {pill(badge[0], badge[1])}
      </div>
    </div>""")


def meter(value: float, color: str, ticks: list, unit: str = '%') -> str:
    """Thin meter: the fill carries the state colour, the track is a pale step, ticks mark the band edges."""
    v = max(0.0, min(100.0, value))
    t = ''.join(f'<i class="tick" style="left:{p}%"></i>' for p, _ in ticks)
    lab = ''.join(f'<span style="left:{p}%">{l}</span>' for p, l in ticks)
    return (f'<div class="cl-meter"><i class="fill" style="width:{v:.1f}%;background:{color}"></i>{t}'
            f'<i class="dot" style="left:{v:.1f}%"></i></div><div class="cl-scale">{lab}</div>')


def risk_meter(prob: float) -> str:
    low, high = (b * 100 for b in dt.C.RISK_BANDS)
    level = dt.risk_band_of(prob)
    return meter(prob * 100, T.RISK_COLORS[level], [(0, '0'), (low, f'{low:.0f}'), (high, f'{high:.0f}'), (100, '100')])


def risk_hero(prob: float, level: str, rel: float, flagged: bool, threshold: float, base: float) -> str:
    color = T.RISK_COLORS[level]
    rel_txt = (f'{rel:.1f}× the {base:.0%} airport average' if rel >= 1.15 else
               f'about the {base:.0%} airport average' if rel > 0.87 else f'{rel:.1f}× the {base:.0%} airport average, below normal')
    alert = (f'{icon("alert", 15, T.CRIT)} Above the {threshold:.0%} alert threshold' if flagged
             else f'{icon("check", 15, T.GOOD)} Below the {threshold:.0%} alert threshold')
    return f"""<div class="cl-wash" style="--w:{T.RISK_BG[level]};--c:{color}">
    <div class="cl-eyebrow">Disruption risk</div>
    <div class="cl-herorow"><span class="cl-hero">{prob * 100:.0f}<small>%</small></span>{pill(level + ' risk', color)}</div>
    <div class="cl-sub">Chance that a flight departing this hour leaves more than 15 minutes late</div>
    {risk_meter(prob)}
    <div class="cl-hr" style="margin:10px 0 8px 0"></div>
    <div class="cl-mut" style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap"><span>{rel_txt}</span>
      <span style="display:inline-flex;gap:5px;align-items:center">{alert}</span></div></div>"""


def severity_hero(sev: dict) -> str:
    name = sev['severity']
    order = ['none', 'minor', 'moderate', 'severe']
    segs = ''.join(f'<i style="width:{max(sev["probabilities"][k] * 100, 0.5):.1f}%;background:{T.SEVERITY_COLORS[k]}"></i>' for k in order)
    legend = ''.join(f'<span><i style="background:{T.SEVERITY_COLORS[k]}"></i>{k.capitalize()} <b style="color:{T.INK};font-weight:600">{sev["probabilities"][k]:.0%}</b></span>' for k in order)
    sc = T.SEVERITY_COLORS[name]
    return f"""<div class="cl-wash" style="--w:{sc}26;--c:{sc}">
    <div class="cl-eyebrow">Expected delay severity</div>
    <div class="cl-herorow"><span class="cl-hero" style="font-size:2.4rem;letter-spacing:-.03em">{name.capitalize()}</span>{pill('most likely', sc, '#FFFFFF')}</div>
    <div class="cl-sub">Most likely outcome for a flight departing in this hour</div>
    <div class="cl-seg" style="margin-top:22px">{segs}</div>
    <div class="cl-legend">{legend}</div></div>"""


def resilience_hero(row, rank_text: str) -> str:
    idx, tier = row['resilience_index'], row['tier']
    color = T.TIER_COLORS[tier]
    return f"""<div class="cl-wash" style="--w:{T.TIER_BG[tier]};--c:{color}">
    <div class="cl-eyebrow">Resilience index · {row['airport_code']}</div>
    <div class="cl-herorow"><span class="cl-hero">{idx:.0f}<small>/ 100</small></span>{pill(tier, color)}</div>
    <div class="cl-sub">{rank_text}</div>
    {meter(idx, color, [(0, '0'), (40, '40'), (70, '70'), (100, '100')])}
    <div class="cl-hr" style="margin:10px 0 8px 0"></div>
    <div class="cl-mut">Vulnerable up to 40 · Moderate 41-70 · Resilient above 70</div></div>"""


def meter_block(title: str, prob: float) -> str:
    level = dt.risk_band_of(prob)
    return f"""<div style="display:flex;justify-content:space-between;align-items:baseline"><span class="cl-sub">{title}</span>
      <span style="display:inline-flex;gap:10px;align-items:center">{pill(level, T.RISK_COLORS[level])}<b style="font-size:1.5rem;letter-spacing:-.02em">{prob:.0%}</b></span></div>
      {risk_meter(prob)}"""


ACCENTS = [T.BLUE, T.TEAL, T.VIOLET, T.INDIGO, '#0EA5E9']          # one decorative accent per KPI tile, in order


def kpi(label_text: str, value: str, delta: str = '', delta_color: str = T.MUTED, note_text: str = '', accent: str = T.BLUE) -> str:
    return (f'<div class="cl-kpi" style="--a:{accent}"><div class="l">{label_text}</div><div class="v">{value}</div>'
            f'<div class="d" style="color:{delta_color}">{delta}</div><div class="n">{note_text}</div></div>')


def kpi_strip(k: dict, weather_sens: float, airport: str, n_days: int) -> None:
    rate, prior = k['rate'], k['prior_rate']
    if prior == prior and rate == rate:
        d = rate - prior
        arrow = '▲' if d > 0.05 else ('▼' if d < -0.05 else '■')
        delta = f'{arrow} {abs(d):.1f} pp vs previous {n_days} days'
        dcolor = '#B42318' if d > 0.05 else ('#006300' if d < -0.05 else T.MUTED)       # text-safe red / green
    else:
        delta, dcolor = 'no earlier period in the data', T.MUTED
    prem = k['peak_premium']
    prem_note = 'peak hours are less disrupted than the rest of the day' if prem < 1 else 'peak hours are more disrupted than the rest of the day'
    show(f"""<div class="cl-kpis">
      {kpi('Disruption rate', f"{rate:.1f}%", delta, dcolor, f"{k['disrupted']:,} of {k['flights']:,} flights delayed over 15 min", ACCENTS[0])}
      {kpi('Average delay when disrupted', f"{k['avg_delay']:.0f} min", '', T.MUTED, 'capped at 300 min per flight', ACCENTS[1])}
      {kpi('Weather sensitivity', f"{weather_sens:.3f}", '', T.MUTED, f'correlation of storm flag with disruption at {airport}', ACCENTS[2])}
      {kpi('Peak-hour premium', f"{prem:.2f}×", '', T.MUTED, prem_note, ACCENTS[3])}
      {kpi('Flights analysed', f"{k['flights']:,}", '', T.MUTED, f"{k['hours']:,} airport-hours in range", ACCENTS[4])}
    </div>""")


def condition_list(u: dict) -> str:
    def row(label_text, value, unit, warn=False):
        mark = f'<span style="color:{T.CRIT};margin-right:6px;font-size:.7rem">●</span>' if warn else ''
        return f'<div><span>{label_text}</span><b>{mark}{value}<small>{unit}</small></b></div>'
    return f"""<div class="cl-list">
      {row('Wind', f"{u['wind_speed_knots']:.0f}", 'kt', u['wind_speed_knots'] >= 25)}
      {row('Visibility', f"{u['visibility_km']:.1f}", 'km', u['visibility_km'] < 3)}
      {row('Rainfall', f"{u['precip_mm_hr']:.1f}", 'mm/h', u['precip_mm_hr'] >= 2.5)}
      {row('Temperature', f"{u['temperature_c']:.0f}", '°C', u['temp_stress'] > 0)}
      {row('Humidity', f"{u['humidity_pct']:.0f}", '%')}
      {row('Traffic in last 2 h', f"{u['congestion_level']:.0f}", 'flights')}
    </div>"""


def flag_chips(u: dict) -> str:
    flags = []
    if u['storm_flag']:
        flags.append(pill('Storm flag', T.CRIT))
    if u['visibility_risk']:
        flags.append(pill('Low visibility', T.WARN))
    if u['temp_stress']:
        flags.append(pill('Temperature stress', T.WARN))
    if u['wind_category_enc'] >= 2:
        flags.append(pill('Strong wind', T.WARN))
    if u['rain_severity_enc'] >= 3:
        flags.append(pill('Heavy rain', T.CRIT))
    elif u['rain_severity_enc'] == 2:
        flags.append(pill('Moderate rain', T.WARN))
    return ' '.join(flags) if flags else pill('No weather flags', T.GOOD)


def history_bars(u: dict) -> str:
    rows = ''.join(f"""<div class="cl-row"><span class="k">{n}</span><span class="b"><i style="width:{max(u[c] * 100, 1.5):.0f}%;background:{T.BLUE}"></i></span>
        <span class="p">{u[c]:.0%}</span></div>""" for c, n in [('delay_last_1h', '2 hours ago'), ('delay_last_3h', 'Last 3 hours'), ('delay_last_24h', 'Last 24 hours')])
    return f'<div class="cl-eyebrow" style="margin:16px 0 4px 0">Recent flights that were delayed</div>{rows}'


def segmented(parts: list, tall: bool = True) -> str:
    """One whole split into parts: [(label, value 0-1, colour)]. The legend lists names and shares in block order."""
    segs = ''.join(f'<i style="width:{max(v * 100, 0.4):.2f}%;background:{c}" title="{l}: {v:.1%}"></i>' for l, v, c in parts)
    legend = ''.join(f'<span><i style="background:{c}"></i>{l} <b style="color:{T.INK};font-weight:600">{v:.1%}</b></span>' for l, v, c in parts)
    return f'<div class="cl-seg{" tall" if tall else ""}">{segs}</div><div class="cl-legend">{legend}</div>'


GROUPS = [('tier', 'Operational response'), ('driver', 'Weather and traffic drivers'), ('severity', 'Passengers')]
TAGS = {'critical': 'Act now', 'watch': 'Monitor', 'info': 'Routine'}
URGENCY_RANK = {'critical': 0, 'watch': 1, 'info': 2}


def rec_groups(recs: list) -> str:
    """Actions grouped by kind; each group header carries one urgency pill, so rows stay quiet (dot + text)."""
    out = ''
    for kind, title in GROUPS:
        items = [r for r in recs if r['kind'] == kind]
        if not items:
            continue
        top = min((r['urgency'] for r in items), key=URGENCY_RANK.get)
        c = T.URGENCY_COLORS[top]
        why = f' <span class="cl-mut" style="text-transform:none;letter-spacing:0;font-weight:400">· {items[0]["trigger"]}</span>' if kind == 'tier' else ''
        out += f'<div class="cl-recgroup" style="display:flex;align-items:center;gap:10px">{title}{why}<span style="margin-left:auto">{pill(TAGS[top], c, T.URGENCY_BG[top])}</span></div>'
        for r in items:
            rc = T.URGENCY_COLORS[r['urgency']]
            detail = '' if kind == 'tier' else f'<div class="w">Why: {r["trigger"]}</div>'
            out += f'<div class="cl-rec"><i class="cl-dot" style="background:{rc}"></i><div><div class="x">{r["text"]}</div>{detail}</div></div>'
    return out


def ranking_table(res) -> str:
    rows = ''
    for r in res.itertuples():
        c = T.TIER_COLORS[r.tier]
        rank = 1 + int((res['resilience_index'] > r.resilience_index).sum())
        tie = int((res['resilience_index'] == r.resilience_index).sum()) > 1
        rows += f"""<tr><td style="width:46px;color:{T.MUTED};font-weight:600">{'=' if tie else ''}{rank}</td>
          <td><b>{r.airport_code}</b><div class="cl-mut">{r.municipality} · {r.hub_size.replace('_', ' ')}</div></td>
          <td style="min-width:200px"><div style="display:flex;align-items:center;gap:12px"><b style="font-size:1.1rem;width:30px">{r.resilience_index:.0f}</b>
            <div class="cl-row" style="margin:0;flex:1"><span class="b"><i style="width:{r.resilience_index:.0f}%;background:{c}"></i></span></div></div></td>
          <td>{pill(r.tier, c)}</td><td>{r.disruption_rate:.1f}%</td><td>{r.avg_recovery_time:.0f} min</td>
          <td>{r.weather_sensitivity:.3f}</td><td>{r.congestion_sensitivity:.3f}</td></tr>"""
    return f"""<table class="cl-table"><thead><tr><th>Rank</th><th>Airport</th><th>Resilience index</th><th>Tier</th><th>Disruption rate</th>
      <th>Avg delay</th><th>Weather sens.</th><th>Congestion sens.</th></tr></thead><tbody>{rows}</tbody></table>"""
