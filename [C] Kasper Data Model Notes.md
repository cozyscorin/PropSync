---
type: research
date: 2026-06-22
project: PropSync
source: https://www.youtube.com/watch?v=KjHt2xSm93o (Kasper MLB Slate Breakdown, June 22nd, @KasperMLB)
---

## What Kasper's model is built on

From the transcript, Kasper ranks hitters daily using these inputs per matchup:

- **Barrel %** — Statcast metric (exit velo + launch angle combo). Public via Baseball Savant.
- **Hard-hit %** — Statcast metric (95+ mph exit velo rate). Public via Baseball Savant.
- **Fly ball %** — standard batted-ball profile stat. Public via FanGraphs/Statcast.
- **Swinging strike % / swing-and-miss rate** — public via FanGraphs/Statcast.
- **"Form"** — his term for recent performance trend (increasing/decreasing). Not a named public stat — looks like a custom rolling-window calculation he built on top of the above.
- **"Zone fit"** — his term for how well a hitter's contact zones match a pitcher's location tendencies. Not a public stat name — appears to be a custom derived metric (heatmap overlap between batter strengths and pitcher tendencies).
- **Handedness splits** — lefty/righty matchup logic (standard).
- **Park factors** — e.g. ballpark suppressing/inflating fly balls. Public via FanGraphs park factors.
- **Sample size awareness** — he flags small samples (e.g. "14 ball sample") and discounts them.

He outputs two ranked lists per game: "KHR" (his own ranking/model output — likely "Kasper's Hit Ranking" or similar) and a separate "matchup" ranking.

## Tooling note
He mentioned the current breakdown is delivered via a Streamlit app (he's building a proper front end separately). So today his pipeline is: pull stats → run his ranking model → display in Streamlit.

## Takeaway for PropSync
The raw ingredients (barrel %, hard-hit %, fly ball %, swinging-strike %, park factors, handedness splits) are all public and pullable from Baseball Savant (Statcast CSV/search endpoints, free) and FanGraphs. What's NOT public is his exact weighting/formula for "form," "zone fit," and the final KHR ranking — that's his proprietary model logic, not something this video reveals directly.

So: we can source the same underlying data for free, but we'd have to build our own ranking logic rather than copy his.

## Open question
Do we want to reverse-engineer a similar ranking approach (barrel + hard-hit + form + zone fit + park factor), or build a different scoring method for PropSync's MLB picks?
