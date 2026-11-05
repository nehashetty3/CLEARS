# CLEARS dashboard: demo script (task T-3.8)

**Length:** about 4 minutes plus 1 minute for questions. **Start:** `streamlit run app/streamlit_app.py` from the project root, then open http://localhost:8501. Every number below was checked against the saved models (`tests/test_app.py` covers the scenario step).

Practice the sequence twice. Use the **ORD** airport and the default **Test period (unseen data)**, which is April, July and October 2022: the last three months in the data, and data the model never saw.

## Step 1 · 30 s · Header and the three hero cards
1. Show the header: airport ORD, hour analysed **Fri 15 Jul 2022, 13:00**, the green "Unseen test data" badge.
2. Point at the **risk card: 95%, High risk, 5.3× the airport average**. "Out of every 100 flights in an hour like this, about 95 are expected to depart more than 15 minutes late."
3. Point at **Resilience index 50 (Moderate)**. "Both airports score 50. We come back to why."

## Step 2 · 60 s · Risk Overview
1. **Trend chart:** the line spikes in the second half of July. "Actual disruption (blue), the 7-day average (teal) and the model's own estimate (dotted) move together. The worst day, **23 July, had 72% of flights delayed and the model estimated 72%**."
2. **SHAP chart, "This hour":** "The biggest drivers are the delays of the last three hours and the 24 hours before. Delays cascade: once flights start leaving late, the next ones do too. Weather is a secondary factor, which is an honest finding, not a gap."
3. Optionally click **Overall (all flights)** to show the same ranking across thousands of flights, and the heatmap: evenings are the worst hours.

## Step 3 · 60 s · Scenario Simulator
1. In the sidebar choose **Pick a date and hour**, set **11 Jul 2022** and **14:00**. The risk card now shows **30%** (just under the Medium band).
2. Open **Scenario Simulator**. Click **Severe storm**. "Visibility drops to 0.5 km, wind to 35 knots, heavy rain, peak traffic." The risk moves from **30% to 75% (+45 points), Low to High**, and severity becomes severe.
3. The "What changed the risk" chart shows wind, visibility and rain doing the work. Mention the amber warning: this combination is beyond almost anything in the data, so 75% is an upper bound.
4. Optional: click **Clear day** (risk falls to 25%) or **Light rain + peak hour** (49%, Medium).
5. "This is the what-if analysis: operations can see how much worse the next hours could get before conditions change."

## Step 4 · 30 s · Recommendations
1. Open **Recommendations**. It is already on **What-if scenario**: High risk, so the actions are "Initiate proactive departure delay programme", "Alert all gate agents of imminent disruption" and, because visibility is low, "Coordinate low-visibility procedures with ATC".
2. Switch to **Selected hour**: the same airport-hour now reads "Conditions nominal - continue standard operations". "The actions come from the risk level, the weather and the model's drivers. Nothing is hard-coded."
3. Click **Download operations brief** to show the hand-over document.

## Step 5 · 30 s · Resilience Rankings
1. Open **Resilience Rankings**. "Both airports tie at 50 because with only two airports the score can only say who is better or worse on each input, and each wins two of four."
2. Point to the four inputs: **ORD** has the higher disruption rate (18.0% vs 15.4%) and congestion sensitivity; **MEM** has longer delays (70 vs 68 minutes) and higher weather sensitivity. "The index becomes meaningful with four or more airports. That is the next step."

## If there is time · Model & Data
Show the calibration chart (a predicted 20% really meant about 20% on unseen data) and the honest summary box: F1 0.37 against 0.35 for logistic regression; the project target of 0.70 is not reachable with these inputs.

## Likely questions
| Question | Answer |
|---|---|
| Why is F1 only 0.37? | Weather, time and recent delays explain part of delay, not all. Even a generous cross-validation reaches only 0.40. The model ranks risky hours well enough to prioritise attention, not to predict single flights. |
| Why 95% on a hour with no weather flags? | The model reads delays that have already built up. That is why the simulator starts in "Weather only" view for such hours. |
| Are the probabilities trustworthy? | Mostly: the calibration chart shows predicted and actual rates agree on unseen data; the very highest-risk hours are somewhat over-confident (about 80% predicted, 66% actual). |
| Why are the risk bands 30% / 65%? | They come from the KPI definitions. The project report uses 50% / 80%. With an 18% base rate the team has to choose; the dashboard reads one constant (`config.RISK_BANDS`). |
| Does it use live weather? | No. It works on the historical NOAA and BTS data (January, April, July, October 2019-2022). |
| Why only two airports? | The data scope chosen by the team. The pipeline is airport-agnostic. |

## Before you present
- [ ] `python -m pytest tests -q` passes (22 tests).
- [ ] The dashboard opens, shows the green "Unseen test data" badge and ORD.
- [ ] Browser zoom 100% to 110%, sidebar open.
- [ ] You have walked through steps 1 to 5 once without notes.
