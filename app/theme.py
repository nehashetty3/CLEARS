"""Design system of the CLEARS dashboard: tokens, CSS and the shared Plotly look.

Coloured but calm. A navy-to-blue brand gradient (banner, sidebar, active tab), a softly tinted page, white cards with
tinted accents, and state-coloured hero cards. In the charts orange is the only second series colour (the pair is
colour-blind safe: CVD delta-E 24.7, see docs/dashboard_layout.md); status colours are reserved for risk and always
paired with a text label. Magnitude uses a single blue ramp; marks are thin, grids are hairlines.
"""
import plotly.graph_objects as go

# --- surfaces and ink
PAGE, CARD, BORDER, HAIR = '#EFF3FA', '#FFFFFF', '#DCE5F3', '#EBF0F8'
INK, INK2, MUTED = '#0F172A', '#475569', '#7A869A'
NAVY, NAVY2, SKY = '#0B2447', '#1B3F8B', '#7DD3FC'
TEAL, VIOLET, INDIGO = '#14B8A6', '#7C5CD6', '#4F5BD5'      # decorative accents for tiles only, never data colours
BRAND = f'linear-gradient(115deg, {NAVY} 0%, {NAVY2} 55%, #2A78D6 100%)'

# --- series colours (validated pair) and the blue ramp used for magnitude
BLUE, ORANGE = '#2A78D6', '#EB6834'
BLUE_RAMP = ['#EAF2FC', '#9EC5F4', '#5598E7', '#256ABF', '#0D366B']
BLUE_SOFT = '#9EC5F4'

# --- status colours (risk only) with their soft backgrounds
GOOD, WARN, CRIT = '#0CA30C', '#FAB219', '#D03B3B'
GOOD_BG, WARN_BG, CRIT_BG, INFO_BG = '#E7F5E7', '#FEF3D6', '#FBE6E6', '#EAF2FC'
RAISES, LOWERS = '#D9534F', BLUE                      # diverging pair for SHAP: red raises risk, blue lowers it

RISK_COLORS = {'Low': GOOD, 'Medium': WARN, 'High': CRIT}
RISK_BG = {'Low': GOOD_BG, 'Medium': WARN_BG, 'High': CRIT_BG}
TIER_COLORS = {'Resilient': GOOD, 'Moderate': WARN, 'Vulnerable': CRIT}
TIER_BG = {'Resilient': GOOD_BG, 'Moderate': WARN_BG, 'Vulnerable': CRIT_BG}
SEVERITY_COLORS = {'none': '#CBD5E1', 'minor': WARN, 'moderate': '#EC835A', 'severe': CRIT}
URGENCY_COLORS = {'critical': CRIT, 'watch': WARN, 'info': BLUE}
URGENCY_BG = {'critical': CRIT_BG, 'watch': WARN_BG, 'info': INFO_BG}

ICONS = {   # feather-style 24x24 stroke icons
    'alert': '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    'check': '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    'info': '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    'plane': '<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>',
    'eye': '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
}


def icon(name: str, size: int = 18, color: str = 'currentColor') -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" style="flex:none">{ICONS[name]}</svg>')


def html(s: str) -> str:
    """Collapse indented multi-line HTML to one line (markdown treats 4-space indents as code blocks)."""
    return ''.join(line.strip() for line in s.strip().splitlines())


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.stApp {{ background: radial-gradient(1100px 520px at 0% 0%, #D9E7FB 0%, transparent 62%), radial-gradient(900px 520px at 100% 0%, #E4DFF8 0%, transparent 58%),
          radial-gradient(900px 600px at 50% 100%, #DDF1F0 0%, transparent 60%), {PAGE}; color: {INK}; }}
#MainMenu, footer, .stAppDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; height: 2.4rem; }}
.block-container {{ padding: 1.6rem 2.6rem 3rem 2.6rem; max-width: 1400px; }}
h1, h2, h3, h4, p, label {{ color: {INK}; }}
[data-testid="stVerticalBlock"] {{ gap: 1rem; }}

/* ---------------------------------------------------------------- sidebar */
[data-testid="stSidebar"] {{ background: linear-gradient(185deg, {NAVY} 0%, #12306B 55%, {NAVY2} 100%); border-right: 0; }}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] {{ padding-top: 1.4rem; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .55rem; }}
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {{ color: #D3E0F5; font-size: .82rem; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: #BFD0EE !important; font-size: .78rem; line-height: 1.5; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{ color: #E8EFFB; }}
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg, [data-testid="stSidebar"] button svg {{ color: #AFC4E8; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="input"] > div {{ background: rgba(255,255,255,.96); border: 0; }}
.cl-side-brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
.cl-side-brand b {{ font-size: 1.05rem; letter-spacing: .01em; }}
.cl-side-title {{ font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; color: #8FB4F0; margin: 1.2rem 0 .1rem 0; font-weight: 600; }}

/* ---------------------------------------------------------------- cards (st.container(key="card_*")) */
[class*="st-key-card_"] {{ background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFF 100%); border: 1px solid {BORDER}; border-radius: 14px; padding: 20px 22px 18px 22px;
    box-shadow: 0 10px 24px -16px rgba(27,63,139,.35), 0 1px 2px rgba(27,63,139,.06); }}
[data-testid="stHorizontalBlock"] {{ align-items: stretch; gap: 1rem; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ height: 100%; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:only-child {{ height: 100%; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:only-child > [class*="st-key-card_"] {{ height: 100%; box-sizing: border-box; }}
.cl-h {{ font-size: .98rem; font-weight: 650; color: {INK}; letter-spacing: -.005em; display: flex; align-items: center; gap: 9px; }}
.cl-h::before {{ content: ""; width: 4px; height: 17px; border-radius: 3px; background: linear-gradient(180deg, #2A78D6, {TEAL}); flex: none; }}
.cl-s {{ font-size: .8rem; color: {MUTED}; margin: 2px 0 10px 0; line-height: 1.4; }}
.cl-eyebrow {{ font-size: .7rem; letter-spacing: .09em; text-transform: uppercase; color: {MUTED}; font-weight: 600; margin-bottom: 8px; }}
.cl-sub {{ color: {INK2}; font-size: .82rem; line-height: 1.5; }}
.cl-mut {{ color: {MUTED}; font-size: .78rem; line-height: 1.45; }}
.cl-hr {{ height: 1px; background: {HAIR}; border: 0; margin: 14px 0; }}
.cl-wash {{ margin: -20px -22px -18px -22px; padding: 20px 22px 18px 22px; border-radius: 14px; background: linear-gradient(180deg, var(--w) 0%, rgba(255,255,255,0) 62%); position: relative; }}
.cl-wash::before {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; border-radius: 14px 14px 0 0; background: var(--c); }}

/* ---------------------------------------------------------------- header */
.cl-header {{ position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
    margin: 0 0 18px 0; padding: 20px 26px; border-radius: 16px; background: {BRAND}; color: #fff; box-shadow: 0 16px 32px -18px rgba(11,36,71,.75); }}
.cl-header::before {{ content: ""; position: absolute; right: -60px; top: -90px; width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(125,211,252,.35) 0%, rgba(125,211,252,0) 68%); }}
.cl-header::after {{ content: ""; position: absolute; left: 38%; bottom: -120px; width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(20,184,166,.28) 0%, rgba(20,184,166,0) 70%); }}
.cl-header > * {{ position: relative; z-index: 1; }}
.cl-brand {{ display: flex; align-items: center; gap: 14px; }}
.cl-logo {{ width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center; background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.3); color: #fff; }}
.cl-title {{ font-size: 1.4rem; font-weight: 700; letter-spacing: -.015em; line-height: 1.2; color: #fff; }}
.cl-title em {{ font-style: normal; color: {SKY}; }}
.cl-tag {{ color: #BBD3F5; font-size: .8rem; margin-top: 3px; }}
.cl-headchips {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
.cl-chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 8px; font-size: .78rem; font-weight: 500;
    background: #F1F6FD; border: 1px solid {BORDER}; color: {INK}; margin: 0 6px 6px 0; }}
.cl-chip b {{ color: {MUTED}; font-weight: 500; }}
.cl-header .cl-chip {{ background: rgba(255,255,255,.12); border-color: rgba(255,255,255,.24); color: #fff; margin: 0; }}
.cl-header .cl-chip b {{ color: #BBD3F5; }}

/* ---------------------------------------------------------------- pills, notes */
.cl-pill {{ display: inline-flex; align-items: center; gap: 7px; padding: 3px 11px 3px 9px; border-radius: 999px; font-size: .76rem; font-weight: 600; color: {INK}; }}
.cl-pill i {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.cl-note {{ display: flex; gap: 12px; align-items: flex-start; padding: 12px 16px; border-radius: 10px; font-size: .84rem; line-height: 1.5; color: {INK};
    background: {INFO_BG}; border: 1px solid #CFE0F7; }}
.cl-note svg {{ margin-top: 2px; color: {BLUE}; }}
.cl-note.warn {{ background: {WARN_BG}; border-color: #F4DC9B; }} .cl-note.warn svg {{ color: #B27A00; }}
.cl-note.bad {{ background: {CRIT_BG}; border-color: #F3C4C4; }} .cl-note.bad svg {{ color: {CRIT}; }}

/* ---------------------------------------------------------------- hero cards */
.cl-hero {{ font-size: 3.1rem; font-weight: 700; letter-spacing: -.035em; line-height: 1; color: {INK}; }}
.cl-hero small {{ font-size: 1.1rem; font-weight: 500; color: {MUTED}; letter-spacing: 0; margin-left: 4px; }}
.cl-herorow {{ display: flex; align-items: center; gap: 12px; margin: 4px 0 10px 0; flex-wrap: wrap; }}
.cl-meter {{ position: relative; height: 8px; background: {HAIR}; border-radius: 99px; margin: 18px 0 6px 0; }}
.cl-meter > .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; }}
.cl-meter > .tick {{ position: absolute; top: -3px; bottom: -3px; width: 1px; background: #C5CDDA; }}
.cl-meter > .dot {{ position: absolute; top: 50%; width: 16px; height: 16px; border-radius: 50%; transform: translate(-50%, -50%);
    background: {INK}; border: 3px solid {CARD}; box-shadow: 0 0 0 1px {BORDER}; }}
.cl-scale {{ position: relative; height: 14px; font-size: .68rem; color: {MUTED}; }}
.cl-scale span {{ position: absolute; transform: translateX(-50%); }}
.cl-seg {{ display: flex; height: 8px; border-radius: 99px; overflow: hidden; gap: 2px; margin: 12px 0 10px 0; background: {CARD}; }}
.cl-seg > i {{ display: block; height: 100%; min-width: 3px; }}
.cl-seg.tall {{ height: 20px; border-radius: 6px; }} .cl-seg.tall > i {{ border-radius: 4px; }}
.cl-legend {{ display: flex; justify-content: space-between; gap: 6px 12px; flex-wrap: wrap; font-size: .74rem; color: {INK2}; }}
.cl-legend span {{ display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }}
.cl-legend i {{ width: 9px; height: 9px; border-radius: 3px; display: inline-block; }}

/* ---------------------------------------------------------------- KPI strip */
.cl-kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin: 0; }}
.cl-kpi {{ --a: {BLUE}; position: relative; overflow: hidden; background: linear-gradient(135deg, color-mix(in srgb, var(--a) 13%, #fff) 0%, #fff 62%);
    border: 1px solid {BORDER}; border-radius: 14px; padding: 15px 18px 13px 22px; box-shadow: 0 10px 22px -16px rgba(27,63,139,.35); }}
.cl-kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--a); }}
.cl-kpi .l {{ font-size: .78rem; color: {INK2}; font-weight: 500; }}
.cl-kpi .v {{ font-size: 1.65rem; font-weight: 650; letter-spacing: -.02em; margin-top: 4px; line-height: 1.15; }}
.cl-kpi .d {{ font-size: .76rem; margin-top: 3px; font-weight: 600; }}
.cl-kpi .n {{ font-size: .72rem; color: {MUTED}; margin-top: 3px; line-height: 1.4; }}
@media (max-width: 1100px) {{ .cl-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}

/* ---------------------------------------------------------------- stat list, bars */
.cl-list > div {{ display: flex; justify-content: space-between; align-items: baseline; padding: 9px 0; border-bottom: 1px solid {HAIR}; font-size: .86rem; }}
.cl-list > div:last-child {{ border-bottom: 0; }}
.cl-list span {{ color: {INK2}; }} .cl-list b {{ font-weight: 600; }} .cl-list small {{ color: {MUTED}; font-weight: 400; margin-left: 3px; }}
.cl-row {{ display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: .8rem; color: {INK2}; }}
.cl-row .k {{ width: 88px; flex: none; }} .cl-row .b {{ flex: 1; height: 6px; background: {HAIR}; border-radius: 99px; overflow: hidden; }}
.cl-row .b i {{ display: block; height: 100%; border-radius: 99px; }} .cl-row .p {{ width: 40px; text-align: right; color: {INK}; font-weight: 600; }}

/* ---------------------------------------------------------------- recommendations */
.cl-recgroup {{ font-size: .7rem; letter-spacing: .09em; text-transform: uppercase; color: {MUTED}; font-weight: 600; margin: 16px 0 2px 0; }}
.cl-recgroup:first-child {{ margin-top: 2px; }}
.cl-rec {{ display: flex; gap: 12px; align-items: flex-start; padding: 12px 0; border-bottom: 1px solid {HAIR}; }}
.cl-rec:last-child {{ border-bottom: 0; }}
.cl-rec .x {{ font-weight: 550; font-size: .9rem; line-height: 1.4; }}
.cl-rec .w {{ color: {MUTED}; font-size: .74rem; margin-top: 2px; }}
.cl-rec .tag {{ margin-left: auto; flex: none; }}
.cl-dot {{ width: 9px; height: 9px; border-radius: 50%; margin-top: 6px; flex: none; }}

/* ---------------------------------------------------------------- table */
.cl-table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
.cl-table th {{ text-align: left; font-size: .72rem; color: {NAVY2}; padding: 9px 12px; background: #EEF4FD; border-bottom: 1px solid {BORDER}; font-weight: 650; }}
.cl-table th:first-child {{ border-radius: 8px 0 0 8px; }} .cl-table th:last-child {{ border-radius: 0 8px 8px 0; }}
.cl-table td {{ padding: 14px 12px; border-bottom: 1px solid {HAIR}; }}
.cl-table tr:last-child td {{ border-bottom: 0; }}

/* ---------------------------------------------------------------- tabs and widgets */
.stTabs [role="tablist"] {{ gap: 6px; padding: 5px; background: rgba(255,255,255,.85); border: 1px solid {BORDER}; border-radius: 14px;
    box-shadow: 0 8px 20px -16px rgba(27,63,139,.45); width: fit-content; max-width: 100%; }}
.stTabs [role="tab"] {{ height: 40px; padding: 0 18px; color: {INK2}; font-weight: 600; font-size: .88rem; background: transparent; border-radius: 10px; }}
.stTabs [role="tab"] p, .stButton > button p, .stDownloadButton > button p {{ color: inherit !important; }}
.stTabs [role="tab"]:hover {{ color: {NAVY2}; background: #EAF1FC; }}
.stTabs [aria-selected="true"] {{ color: #fff !important; background: {BRAND} !important; box-shadow: 0 6px 14px -8px rgba(27,63,139,.9); }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; border: 1px solid #C9D9F2; background: linear-gradient(180deg, #FFFFFF, #EEF4FD); color: {NAVY2};
    font-weight: 600; box-shadow: 0 2px 6px -3px rgba(27,63,139,.35); transition: all .12s; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {BLUE}; color: #fff; background: {BRAND}; }}
.stButton > button p {{ white-space: normal; line-height: 1.2; }}
.stButton > button {{ min-height: 2.7rem; }}
[data-testid="stMetricValue"] {{ font-weight: 650; letter-spacing: -.02em; }}
[data-testid="stMetricLabel"] p {{ color: {INK2}; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ border-radius: 8px; }}
@media (max-width: 768px) {{ .block-container {{ padding: 1rem .9rem 2rem .9rem; }} .cl-title {{ font-size: 1.15rem; }} .cl-hero {{ font-size: 2.6rem; }} }}
"""


def inject(st) -> None:
    st.markdown(f'<style>{CSS}</style>', unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    """Shared Plotly look: transparent background, ink text, hairline recessive grid, white hover cards."""
    fig.update_layout(
        height=height, margin=dict(l=4, r=8, t=24, b=4), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, system-ui, Segoe UI, sans-serif', color=INK2, size=12),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=INK, size=12)),
        showlegend=legend, legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(size=11.5, color=INK2), bgcolor='rgba(0,0,0,0)'),
        title=dict(font=dict(size=13, color=INK)))
    fig.update_xaxes(gridcolor=HAIR, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(size=11, color=MUTED), title_font=dict(size=11.5, color=INK2))
    fig.update_yaxes(gridcolor=HAIR, zerolinecolor=BORDER, linecolor='rgba(0,0,0,0)', tickfont=dict(size=11, color=MUTED), title_font=dict(size=11.5, color=INK2))
    return fig
