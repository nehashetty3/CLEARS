"""Resilience inputs and index for CLEARS (Person 1 builds the inputs, Person 2 owns the index).

build_resilience_inputs(df)   -> one row per airport (T-1.7)
compute_resilience_index(inp) -> adds resilience_index and tier (T-2.6)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / 'data' / 'processed'
INPUTS_PATH = PROCESSED / 'airport_resilience_inputs.csv'     # produced by build_resilience_inputs (T-1.7)
SCORES_PATH = PROCESSED / 'resilience_scores.csv'             # T-2.6 output, read by the dashboard

WEIGHTS = {
    'disruption_rate': 0.35,
    'avg_recovery_time': 0.30,
    'weather_sensitivity': 0.20,
    'congestion_sensitivity': 0.15,
}


def build_resilience_inputs(df: pd.DataFrame, context: pd.DataFrame = None) -> pd.DataFrame:
    """Aggregate the feature matrix to one row per airport.

    Columns:
      disruption_rate            % of flights with disruption = 1 (dep_delay > 15 min).
      avg_recovery_time          mean delay_minutes of disrupted flights (capped at 300). This is the average
                                 delay of a disrupted flight, not a measured time-to-recover.
      weather_sensitivity        Pearson corr(storm_flag, disruption) over all rows of the airport.
      congestion_sensitivity     Pearson corr(congestion_level, disruption).
      peak_hour_disruption_rate  % disrupted among is_peak_hour == 1 rows.
      data_period_days           calendar span first -> last date (NOT the number of days with data).
      days_with_data             number of distinct dates present (492 in the study data).
    `context` (airport_context.csv) optionally adds hub size, runways and coordinates.
    """
    rows = []
    for airport, g in df.groupby('airport_code'):
        rows.append({
            'airport_code': airport,
            'disruption_rate': round(g['disruption'].mean() * 100, 4),
            'avg_recovery_time': round(g.loc[g['disruption'] == 1, 'delay_minutes'].mean(), 4),
            'weather_sensitivity': round(g['storm_flag'].corr(g['disruption']), 4),
            'congestion_sensitivity': round(g['congestion_level'].corr(g['disruption']), 4),
            'peak_hour_disruption_rate': round(g.loc[g['is_peak_hour'] == 1, 'disruption'].mean() * 100, 4),
            'data_period_days': (pd.to_datetime(g['flight_date']).max()
                                 - pd.to_datetime(g['flight_date']).min()).days,
            'days_with_data': g['flight_date'].nunique(),
        })
    out = pd.DataFrame(rows)
    out = out.assign(**{'_k': 0})  # keep column order stable after the optional merge
    if context is not None:
        keep = ['airport_code', 'hub_size', 'runway_count', 'max_runway_length_ft',
                'latitude_deg', 'longitude_deg', 'elevation_ft', 'municipality']
        out = out.merge(context[keep], on='airport_code', how='left')
    return out.drop(columns='_k')


def tier_for(score: float) -> str:
    """Continuous boundaries: <= 40 Vulnerable, (40, 70] Moderate, > 70 Resilient."""
    if score <= 40:
        return 'Vulnerable'
    if score <= 70:
        return 'Moderate'
    return 'Resilient'


def compute_resilience_index(inputs: pd.DataFrame = None, weights: dict = WEIGHTS) -> pd.DataFrame:
    """Min-max normalise each dimension across airports, weight them, index = 100 - 100 * vulnerability.

    WARNING: with only two airports min-max scaling maps every dimension to exactly 0 or 1, so the
    index depends only on which airport is higher on each dimension, not on by how much. Use >= 4
    airports (or fixed external reference ranges) before interpreting the scores.
    A dimension with no spread across airports contributes 0 vulnerability.
    With inputs=None the table is read from data/processed/airport_resilience_inputs.csv (task T-2.6).
    """
    if inputs is None:
        inputs = pd.read_csv(INPUTS_PATH)
    out = inputs.copy()
    dims = list(weights)
    span = (out[dims].max() - out[dims].min()).replace(0, np.nan)
    norm = ((out[dims] - out[dims].min()) / span).fillna(0.0)
    vulnerability = sum(norm[d] * w for d, w in weights.items())
    out['resilience_index'] = (100 - 100 * vulnerability).round(1)
    out['tier'] = out['resilience_index'].apply(tier_for)
    # put the score right after the aggregate inputs, before the airport context columns
    front = [c for c in out.columns if c not in ('resilience_index', 'tier')]
    pos = front.index('days_with_data') + 1 if 'days_with_data' in front else len(front)
    return out[front[:pos] + ['resilience_index', 'tier'] + front[pos:]]


def save_resilience_scores(scores: pd.DataFrame, path: Path = SCORES_PATH) -> pd.DataFrame:
    """Write data/processed/resilience_scores.csv.

    Required columns (task guide T-2.6): airport_code, resilience_index, tier. The extra columns are what the
    dashboard's resilience visual needs (T-3.7): disruption_rate (marker size), lat / lon (scatter map)."""
    out = scores[['airport_code', 'resilience_index', 'tier']].copy()
    for extra, source in [('disruption_rate', 'disruption_rate'), ('lat', 'latitude_deg'), ('lon', 'longitude_deg')]:
        if source in scores:
            out[extra] = scores[source].values
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def run(features_path: Path, context_path: Path, scores_path: Path = SCORES_PATH) -> pd.DataFrame:
    """features -> per-airport inputs -> index -> resilience_scores.csv (also keeps the inputs file in sync)."""
    df = pd.read_csv(features_path, low_memory=False)
    context = pd.read_csv(context_path)
    inputs = build_resilience_inputs(df, context)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    inputs.to_csv(INPUTS_PATH, index=False)
    scores = compute_resilience_index(inputs)
    save_resilience_scores(scores, scores_path)
    return scores


if __name__ == '__main__':
    import config
    result = run(config.DATA_PATH, config.CONTEXT_PATH)
    print(result[['airport_code', 'resilience_index', 'tier']].to_string(index=False))
    print('wrote', SCORES_PATH)
