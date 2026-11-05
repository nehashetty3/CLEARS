"""Plotly figures for the dashboard. Thin marks, hairline grid, one blue ramp for magnitude, blue/orange for two series,
red/blue for a push up/down; every axis is labelled (T-3.8)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data as dt
import theme as T

C = dt.C
BAR = dict(cornerradius=4, line_width=0)          # 4px rounded data end


def trend_chart(daily: pd.DataFrame, snap_date, show_pred: bool, mark_events: bool, height: int = 340) -> go.Figure:
    d = daily.copy()
    d['rolling'] = d['actual_rate'].rolling(7, min_periods=3, center=True).mean().where(d['actual_rate'].notna())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['date'], y=d['actual_rate'], name='Actual, daily', mode='lines', line=dict(color=T.BLUE_SOFT, width=1.2),
                             hovertemplate='%{x|%d %b %Y}<br>Actual: %{y:.1f}%<extra></extra>', connectgaps=False))
    fig.add_trace(go.Scatter(x=d['date'], y=d['rolling'], name='Actual, 7-day average', mode='lines', line=dict(color=T.BLUE, width=2.2),
                             hovertemplate='%{x|%d %b %Y}<br>7-day average: %{y:.1f}%<extra></extra>'))
    if show_pred:
        fig.add_trace(go.Scatter(x=d['date'], y=d['predicted_rate'], name='Model estimate', mode='lines', line=dict(color=T.ORANGE, width=2),
                                 hovertemplate='%{x|%d %b %Y}<br>Model: %{y:.1f}%<extra></extra>'))
    if mark_events:
        ev, lv = d[d['storm'] == 1], d[(d['low_vis'] == 1) & (d['storm'] != 1)]
        if len(ev):
            fig.add_trace(go.Scatter(x=ev['date'], y=[0] * len(ev), mode='markers', name='Storm-flag day',
                                     marker=dict(symbol='triangle-up', size=10, color=T.CRIT, line=dict(color='#fff', width=2)),
                                     hovertemplate='%{x|%d %b %Y}<br>Storm flag active<extra></extra>'))
        if len(lv):
            fig.add_trace(go.Scatter(x=lv['date'], y=[0] * len(lv), mode='markers', name='Low-visibility day',
                                     marker=dict(symbol='diamond', size=9, color=T.WARN, line=dict(color='#fff', width=2)),
                                     hovertemplate='%{x|%d %b %Y}<br>Visibility below 3 km<extra></extra>'))
    start, end = d['date'].min(), d['date'].max()
    if start < C.TEST_START <= end:
        fig.add_vrect(x0=C.TEST_START, x1=end, fillcolor=T.INFO_BG, opacity=0.6, line_width=0, layer='below',
                      annotation_text='unseen test period', annotation_position='top left', annotation_font=dict(size=10, color=T.MUTED))
    if snap_date is not None and start <= pd.Timestamp(snap_date) <= end:
        fig.add_vline(x=pd.Timestamp(snap_date).value / 1e6, line=dict(color=T.INK2, width=1),
                      annotation_text='selected hour', annotation_position='top', annotation_font=dict(size=10, color=T.INK2))
    T.style_figure(fig, height)
    fig.update_xaxes(title='', showgrid=False)
    fig.update_yaxes(title='Flights delayed over 15 min', rangemode='tozero', ticksuffix='%')
    fig.update_layout(hovermode='x unified', margin=dict(l=4, r=8, t=52, b=4), legend=dict(orientation='h', x=0, y=1.14))
    return fig


def shap_local(shap: dict, values: dict, top: int = 8, height: int = 320) -> go.Figure:
    s = pd.Series(shap)
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
    labels = [f'{dt.label(k)}  ·  {dt.value_text(k, values[k])}' for k in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=labels, orientation='h', marker=dict(color=[T.RAISES if v > 0 else T.LOWERS for v in s.values], **BAR),
                           text=[f'{v:+.2f}' for v in s.values], textposition='outside', textfont=dict(size=11, color=T.INK2),
                           hovertemplate='%{y}<br>push on risk: %{x:+.3f}<extra></extra>', cliponaxis=False))
    T.style_figure(fig, height, legend=False)
    lim = max(0.2, float(s.abs().max()) * 1.4)
    fig.update_xaxes(title='Lowers risk  ←   SHAP value   →  Raises risk', range=[-lim, lim], zeroline=True, zerolinecolor=T.MUTED)
    fig.update_yaxes(title='', gridcolor='rgba(0,0,0,0)', automargin=True)
    fig.update_layout(bargap=0.55)
    return fig


def shap_global(imp: pd.Series, top: int = 10, height: int = 320) -> go.Figure:
    s = imp.head(top)[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=[dt.label(k) for k in s.index], orientation='h', marker=dict(color=T.BLUE, **BAR),
                           text=[f'{v:.2f}' for v in s.values], textposition='outside', textfont=dict(size=11, color=T.INK2),
                           hovertemplate='%{y}<br>mean |SHAP|: %{x:.3f}<extra></extra>', cliponaxis=False))
    T.style_figure(fig, height, legend=False)
    fig.update_xaxes(title='Average influence on the model (mean |SHAP|)', range=[0, float(s.max()) * 1.2])
    fig.update_yaxes(title='', gridcolor='rgba(0,0,0,0)', automargin=True)
    fig.update_layout(bargap=0.55)
    return fig


def group_importance(groups: pd.Series, height: int = 260) -> go.Figure:
    share = (groups / groups.sum() * 100).sort_values()
    fig = go.Figure(go.Bar(x=share.values, y=share.index, orientation='h', marker=dict(color=T.BLUE, **BAR),
                           text=[f'{v:.0f}%' for v in share.values], textposition='outside', textfont=dict(color=T.INK2, size=11.5),
                           hovertemplate='%{y}: %{x:.1f}% of total influence<extra></extra>', cliponaxis=False))
    T.style_figure(fig, height, legend=False)
    fig.update_xaxes(title="Share of the model's total influence", range=[0, float(share.max()) * 1.2], ticksuffix='%')
    fig.update_yaxes(title='', gridcolor='rgba(0,0,0,0)', automargin=True)
    fig.update_layout(bargap=0.6)
    return fig


def hour_heatmap(rng: pd.DataFrame, height: int = 370) -> go.Figure:
    g = rng.groupby(['day_of_week', 'hour_of_day'])[['n_disrupted', 'n_flights']].sum()
    g = g[g['n_flights'] >= 15]                 # cells with only a handful of flights are noise
    z = (g['n_disrupted'] / g['n_flights'] * 100).unstack('hour_of_day').reindex(index=range(7), columns=range(24))
    zmax = max(30.0, float(np.nanpercentile(z.values, 97)))
    ramp = [[i / (len(T.BLUE_RAMP) - 1), c] for i, c in enumerate(T.BLUE_RAMP)]
    fig = go.Figure(go.Heatmap(z=z.values, x=[f'{h:02d}' for h in range(24)], y=dt.DAYS, xgap=2, ygap=2, zmin=0, zmax=zmax, colorscale=ramp,
                               colorbar=dict(title=dict(text='% delayed', side='right'), thickness=8, len=0.85, tickfont=dict(size=10), outlinewidth=0),
                               hovertemplate='%{y} %{x}:00<br>%{z:.1f}% of flights delayed over 15 min<extra></extra>'))
    T.style_figure(fig, height, legend=False)
    fig.update_xaxes(title='Scheduled departure hour (local time)', showgrid=False, tickmode='array', tickvals=[f'{h:02d}' for h in range(0, 24, 3)])
    fig.update_yaxes(title='', autorange='reversed', showgrid=False)
    return fig


def sensitivity_curve(xs, ys, xlabel: str, scenario_xy, base_xy, height: int = 250, reverse_x: bool = False) -> go.Figure:
    fig = go.Figure()
    low, high = (b * 100 for b in C.RISK_BANDS)
    for y0, y1, c in [(0, low, T.GOOD_BG), (low, high, T.WARN_BG), (high, 100, T.CRIT_BG)]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.55, line_width=0, layer='below')
    fig.add_trace(go.Scatter(x=list(xs), y=[y * 100 for y in ys], mode='lines', line=dict(color=T.BLUE, width=2.2), name='Risk',
                             hovertemplate=xlabel + ' %{x:.1f}<br>risk %{y:.1f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=[base_xy[0]], y=[base_xy[1] * 100], mode='markers', name='Selected hour',
                             marker=dict(size=10, color=T.INK, line=dict(color='#fff', width=2)), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[scenario_xy[0]], y=[scenario_xy[1] * 100], mode='markers', name='What-if',
                             marker=dict(size=12, color=T.ORANGE, symbol='diamond', line=dict(color='#fff', width=2)), hoverinfo='skip'))
    T.style_figure(fig, height)
    fig.update_xaxes(title=xlabel, autorange='reversed' if reverse_x else True, showgrid=False)
    fig.update_yaxes(title='Disruption risk', range=[0, 100], ticksuffix='%')
    return fig


def resilience_bar(df: pd.DataFrame, height: int = 250) -> go.Figure:
    d = df.sort_values('resilience_index')
    fig = go.Figure(go.Bar(x=d['resilience_index'], y=d['airport_code'], orientation='h', marker=dict(color=[T.TIER_COLORS[t] for t in d['tier']], **BAR),
                           text=[f'{v:.0f} · {t}' for v, t in zip(d['resilience_index'], d['tier'])], textposition='outside', textfont=dict(color=T.INK, size=12),
                           hovertemplate='%{y}<br>Resilience index: %{x:.1f}<extra></extra>', cliponaxis=False))
    for x in (40, 70):
        fig.add_vline(x=x, line=dict(color=T.BORDER, width=1))
    T.style_figure(fig, height, legend=False)
    fig.update_xaxes(title='Resilience index (0 = most vulnerable, 100 = most resilient)', range=[0, 100], tickvals=[0, 40, 70, 100])
    fig.update_yaxes(title='', gridcolor='rgba(0,0,0,0)')
    fig.update_layout(bargap=0.6)
    return fig


def resilience_map(df: pd.DataFrame, height: int = 300) -> go.Figure:
    ramp = [[i / (len(T.BLUE_RAMP) - 1), c] for i, c in enumerate(T.BLUE_RAMP)]
    fig = go.Figure(go.Scattergeo(
        lat=df['lat'], lon=df['lon'], mode='markers+text', text=df['airport_code'], textposition='middle right', textfont=dict(color=T.INK, size=12),
        marker=dict(size=df['disruption_rate'] * 1.5, color=df['resilience_index'], colorscale=ramp, cmin=0, cmax=100, line=dict(color='#fff', width=2), opacity=0.95,
                    colorbar=dict(title=dict(text='Resilience', side='right'), thickness=8, len=0.8, tickfont=dict(size=10), outlinewidth=0)),
        customdata=np.stack([df['airport_code'], df['resilience_index'], df['disruption_rate'], df['tier']], axis=-1),
        hovertemplate='%{customdata[0]}<br>Resilience %{customdata[1]:.0f} (%{customdata[3]})<br>Disruption rate %{customdata[2]:.1f}%<extra></extra>'))
    fig.update_geos(scope='usa', projection_type='albers usa', bgcolor='rgba(0,0,0,0)', landcolor='#EEF1F6', subunitcolor='#D5DBE5',
                    countrycolor='#D5DBE5', showlakes=False, showsubunits=True, framecolor='rgba(0,0,0,0)')
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=6, b=0), paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Inter, system-ui, sans-serif', color=T.INK2), hoverlabel=dict(bgcolor='#fff', bordercolor=T.BORDER, font=dict(color=T.INK)))
    return fig


def dimension_chart(df: pd.DataFrame, height: int = 270) -> go.Figure:
    dims = [('disruption_rate', 'Disruption rate (%)'), ('avg_recovery_time', 'Average delay when disrupted (min)'),
            ('weather_sensitivity', 'Weather sensitivity (r)'), ('congestion_sensitivity', 'Congestion sensitivity (r)')]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[d[1] for d in dims], horizontal_spacing=0.06)
    colors = dict(zip(sorted(df['airport_code']), [T.BLUE, T.ORANGE]))
    for i, (col, _) in enumerate(dims, 1):
        for _, r in df.sort_values('airport_code').iterrows():
            fig.add_trace(go.Bar(x=[r['airport_code']], y=[r[col]], name=r['airport_code'], marker=dict(color=colors[r['airport_code']], **BAR),
                                 text=[f'{r[col]:.2f}' if r[col] < 1 else f'{r[col]:.1f}'], textposition='outside', textfont=dict(color=T.INK, size=11.5),
                                 showlegend=(i == 1), hovertemplate='%{x}: %{y:.3f}<extra></extra>', cliponaxis=False), row=1, col=i)
        fig.update_yaxes(range=[0, float(df[col].max()) * 1.3], showticklabels=False, showgrid=False, row=1, col=i)
        fig.update_xaxes(showgrid=False, row=1, col=i)
    T.style_figure(fig, height)
    fig.update_annotations(font=dict(size=11.5, color=T.INK2))
    fig.update_layout(bargap=0.5, margin=dict(l=4, r=8, t=64, b=4), legend=dict(orientation='h', x=0, y=1.3))
    return fig


def calibration_chart(h: pd.DataFrame, height: int = 330) -> go.Figure:
    d = h.copy()
    d['bin'] = pd.cut(d['probability'], [0, .1, .15, .2, .25, .3, .4, .65, 1.0], include_lowest=True)
    g = d.groupby('bin', observed=True).apply(lambda x: pd.Series({
        'pred': np.average(x['probability'], weights=x['n_flights']) * 100,
        'actual': x['n_disrupted'].sum() / x['n_flights'].sum() * 100, 'flights': x['n_flights'].sum()}), include_groups=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 90], y=[0, 90], mode='lines', line=dict(color=T.MUTED, width=1), name='Perfect calibration', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=g['pred'], y=g['actual'], mode='lines+markers', name='Model, unseen test hours', line=dict(color=T.BLUE, width=2.2),
                             marker=dict(size=9, color=T.BLUE, line=dict(color='#fff', width=2)),
                             hovertemplate='Predicted %{x:.0f}%<br>Actually delayed %{y:.0f}%<extra></extra>'))
    T.style_figure(fig, height)
    fig.update_xaxes(title='Predicted disruption probability', range=[0, 90], ticksuffix='%', showgrid=False)
    fig.update_yaxes(title='Share of flights actually delayed', range=[0, 90], ticksuffix='%')
    return fig
