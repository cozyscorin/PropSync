---
type: problems
date: 2026-06-22
project: PropSync
---

## Goal
Build a sports betting tool that gives people data and live picks for player props and parlays, starting with MLB and expanding to NBA and NFL.

## Why
To ultimately make a profit.

## Tangible Outcomes
- An app or website showing data relative to player props
- A daily feed of the best picks
- MLB coverage first, then NBA and NFL

## Open Problems
1. Define the scoring model for each prop type (HR, hits, total bases, RBIs, singles, doubles, pitcher strikeouts) — see `[C] Scoring Framework Notes.md` for the breakdown of inputs per prop type.
2. ~~Figure out how to normalize confidence across prop types~~ — Decided: each prop model outputs a real win probability, then ranked by edge vs. the sportsbook's implied probability. See `[C] Scoring Framework Notes.md`.
3. Source the raw data pipeline: Statcast metrics/park factors/weather from Baseball Savant/FanGraphs, live FanDuel + DraftKings player prop odds via The Odds API (`batter_home_runs`, `batter_hits`, `batter_total_bases`, `batter_rbis`, `batter_singles`, `batter_doubles`, `pitcher_strikeouts`). Both books used as cross-references in case a line is missing from one. See `[C] Scoring Framework Notes.md`.
4. ~~Decide on the front-end delivery~~ — Decided: Streamlit Community Cloud (free, same tool Kasper uses). See `[C] Scoring Framework Notes.md`. Not yet built.
