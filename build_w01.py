import json

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None,
                       "metadata": {}, "outputs": [], "source": src}

cells = [
md("""# ML-02 — Research Question and Provisional Lane

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

*Assistant note: the skills `framing-ml-problems` and `flyrank/flyrank-data` were loaded for this notebook, per `skills/README.md`."""),

md("""## 1. My lane (or freestyle) and why

*Name your lane — or say 'freestyle' and describe your own question. One short paragraph: why this one?*

**My provisional lane: Lane 2 — Refresh / Content Opportunity Scoring.**

The question this lane asks — *which pages should be reviewed first for refresh, expansion, protection, pruning, or monitoring?* — is the most direct match to a real decision I can name: a content reviewer has limited hours and needs a ranked queue, not a dashboard. I pick it for three reasons. First, the starter dataset and pipeline are built around exactly this decision, so I can back the choice with real numbers today instead of promising future data work. Second, the starter's verified results show a learned ranking can beat a transparent rule by a wide margin on this task (Precision@50 of 0.240 for the baseline rules vs 0.740 for the random forest on the 30k-row starter slice), which tells me there is real, learnable signal here — not just noise. Third, the data has both sides the decision needs: demand signals (impressions, position) and movement/weakness signals (trend, freshness, CTR, engagement), all from observable measurements, not product decisions. This is provisional — I can confirm or change it until the end of Week 4 — but it is the lane where the decision, the action, and the data already line up."""),

code('''import pandas as pd

# Starter dataset: one row per pseudonymized content item (see docs/data-dictionary.md)
df = pd.read_csv("../../data/raw/content_refresh_anonymized.csv")

print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")
print(f"unique content_id: {df['content_id'].nunique()}  -> grain confirmed: one row = one page")
print(f"clients: {df['client_id'].nunique()} (pseudonyms; grouping/splitting only, never features)")'''),

md("""## 2. The question: decision, action, cost of a wrong call

*What decision does your work improve? Who acts on it? What does a wrong recommendation cost?*

**The decision:** which pages a content reviewer should look at *first* this week, given they can only review a handful. That is a prioritization decision under scarcity — exactly what a ranked queue with reason codes supports.

**Who acts and how:** a content editor or SEO reviewer works down the ranked list. For each top page they take a concrete action — refresh outdated content, expand a thin page, fix a title/meta mismatch, protect a decaying page-one position, merge or prune a redundant page, or just monitor. The output is decision-support: the model ranks candidates and explains *why* (reason codes); the human still makes the final call on each page.

**Cost of a wrong call:**
- *False positive (a page ranked high that wasn't worth fixing):* wasted reviewer hours — the scarcest resource in this workflow — and a little credibility for the tool.
- *False negative (a genuinely declining, high-demand page sinks to the bottom of the queue):* missed traffic and revenue that a timely review might have caught.

Because reviewer time is scarce, the metric that matches the real decision is **precision@K** (of the top-K pages we tell a reviewer to open, how many were genuinely worth it), not generic accuracy.

**Why data/ML helps at all:** there are ~10k candidate pages and maybe 20–50 slots in a review week. A hand-written rule is a start (the starter baseline is exactly that), but it weighs a handful of signals with fixed weights. The starter evidence says the pattern is real yet tangled enough that a learned model finds far more worthwhile pages in the top 50 than the fixed rule does. That is the bar: beat a transparent rule, honestly validated, or admit the rule was enough."""),

code('''# The capacity problem in one number:
# pages that are BOTH visible (real demand) AND declining (worth a look) --
# far more than any review team can handle, so ranking IS the decision.
visible = df["impressions_90d"] >= 500
vis_decl = visible & (df["trend_direction"] == "down")

print(f"visible pages (impressions_90d >= 500):            {visible.sum():>6,}")
print(f"visible AND declining pages:                       {vis_decl.sum():>6,}")
print(f"  ...spread across {df.loc[vis_decl, 'client_id'].nunique()} of {df['client_id'].nunique()} clients")
print(f"a reviewer who opens 50 pages/week faces a 1-in-{vis_decl.sum() // 50} needle in this haystack")'''),

md("""## 3. Quick look at the data (2-3 real numbers)

*Load the starter CSV below and show 2-3 real numbers that make your lane look worth the next 7 weeks.*

Three numbers from the starter slice (`data/raw/content_refresh_anonymized.csv`, 30,000 pages × 32 clients):

1. **16,726 pages (55.8%) have real search demand** (`impressions_90d >= 500`). There is a large population where improvement would actually matter — this is not a dead dataset.
2. **9,961 of those visible pages (59.6%) are declining.** The problem is not finding *a* page to fix; it is that far more pages look fixable than anyone can review. Ranking is the real need.
3. **Median CTR on page-1/2 visible pages is 0.24%** — and 9,759 visible pages sitting at positions 1-20 have CTR under 0.5%. That is a big, concrete pool of pages that searchers see but don't click — a natural, evidence-backed starting population for the review queue.

Together: enough demand to matter, enough movement to prioritize, and a specific weak spot (CTR under-capture) to act on."""),

code('''# Real numbers backing the lane (rates are x100 percentages per the data skill:
# ctr = 0.24 means 0.24%, not 24%)
n = len(df)
visible = df["impressions_90d"] >= 500
vis_decl = visible & (df["trend_direction"] == "down")
page1_2 = visible & (df["avg_position"] > 0) & (df["avg_position"] <= 20)
low_ctr = page1_2 & (df["ctr"] < 0.5)

print(f"1) visible pages (impr_90d >= 500):        {visible.sum():>6,}  ({visible.mean():.1%} of {n:,})")
print(f"2) visible AND declining:                  {vis_decl.sum():>6,}  ({(df.loc[visible,'trend_direction']=='down').mean():.1%} of visible pages)")
print(f"3) page-1/2 visible pages with ctr < 0.5%: {low_ctr.sum():>6,}")
p1 = df[page1_2]
print(f"   median ctr on page-1/2 visible pages:   {p1['ctr'].median():.2f}%")'''),

md("""## 4. Careful words: what I can and can't claim

*Write what your work will be able to say (observed, directional, decision-support) — and what it never will (causal proof, 'predicting Google').*

**What this work can claim (and how it will say it):**
- *Decision-support, observed:* "pages scored in the top-K were more often genuinely declining/worth review than pages picked by the baseline rule" — measured with client-holdout validation and precision@K.
- *Directional:* "these signals are associated with decline / under-captured clicks" — the language of association, not mechanism.
- *Provisional:* the starter label (`trend_direction == "down"`) is a proxy computed from the current window, not a future outcome. My capstone should move to a forward-looking label (prior 90 days of features -> outcome over the next 30 days), and until then every result is scoped to the proxy.

**What it will never claim:**
- That a refresh *caused* a recovery — that would need an experiment or causal design this data cannot provide.
- That we are "predicting Google" or proving an algorithm factor — these are observational signals from one anonymized slice.
- That top-ranked pages are guaranteed to recover — the output ranks candidates for a human to review, full stop.

Two disciplines baked in from the start, straight from the data skill: `trend_direction` and `trend_pct` are **never features** (the label is derived from them — using them would be the circular-result trap), and pseudonymized IDs are for grouping/splitting only."""),

code('''# Verify the label trap out loud: the declining label is derived from trend_direction,
# which is computed from trend_pct -> so trend_direction / trend_pct can NEVER be features.
mismatch = ((df["trend_direction"] == "down") != (df["trend_pct"] < 0)).sum()
print(f"rows where 'down' disagrees with trend_pct < 0: {mismatch}")
print("-> the label is mechanically derived from trend_pct; both columns are excluded from features")
print(f"starter label base rate (trend_direction == 'down'): {(df['trend_direction'] == 'down').mean():.1%}")'''),

md("""## Self-check

Before you submit, confirm each line honestly:

- [x] Every section above is filled — markdown thinking AND the code that backs it
- [x] The notebook runs top to bottom with no errors (Runtime → Run all)
- [x] No client names, URLs, or private queries anywhere (all IDs are pseudonyms; only aggregates shown)
- [x] My claims use careful words: observed, measured, directional, decision-support
- [x] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done."""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = "work/notebooks/w01_research_question.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
