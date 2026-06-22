---
type: research
date: 2026-06-22
project: PropSync
---

## Decision: parlay structure

PropSync builds its top-N picks list with no exclusion logic at all — same-player and same-game legs are both allowed. This came in two steps:

1. Same-game exclusion (different players, same game) was dropped first: a game can have more than one real, mostly-independent good pick, and blocking the rest of a game over one selected leg threw away real edge.
2. Same-player exclusion (different prop types, same player — e.g. his HR leg and his hits leg) was dropped next, on the same reasoning: they're different bets, and a player having a great game can legitimately justify multiple legs.

Worth keeping on record: a home run technically also counts as a hit, so "1+ hits" and "HR" aren't fully independent outcomes even though they're different markets — that overlap (and same-game correlation more generally — weather, a blowout changing playing time, etc.) is a real tradeoff being accepted deliberately, not an oversight. If a correlation concern becomes a real problem in practice, exclusion logic should be re-added deliberately rather than assumed back in.

## Architecture: shared data layer + per-prop scoring lens

One shared pool of raw player/pitcher stats, then a separate scoring "lens" per prop type that pulls and weights a different subset of that pool. Don't rebuild the data layer for each prop — just change what each lens emphasizes.

## Prop type breakdowns

### Home runs
- Pulled-air rate (pulled fly ball %) — more HR-specific than generic fly ball %
- Statcast Expected Home Runs / xSLG — controls for park and luck better than barrel % alone
- Pitcher side: HR/9, barrel % allowed, fastball-in-zone rate
- Same-day weather (wind direction/speed, temp) — park factor is static, weather isn't
- Recency-weighted form + statistical shrinkage toward league average for low-PA players (formalizes Kasper's gut-feel "small sample" calls)
- Compare model output against sportsbook HR prop odds — biggest disagreements with the market are where the edge is

### Hits (1+ hits props)
- Strikeout % (lower = better)
- Overall hard-hit % (not just barrels — contact rate matters more than contact quality here)
- Expected batting average (xBA)
- Pitcher's hit rate allowed / WHIP
- Platoon splits (batter vs. pitcher handedness)

### Total bases
- ISO and xSLG (more relevant than barrel % alone)
- Lineup spot (more PAs = more total-base opportunities)
- Park factor split by hit type — a park can suppress HRs but still favor doubles/triples (big gaps, foul territory), so park factor isn't one number here

### RBIs
- Lineup spot (3-4-5 hitters get far more RBI chances)
- On-base ability of hitters batting ahead of this player
- Team implied run total for the game (Vegas number, not a player stat)
- Less about the batter's own skill, more about context — a great hitter in a weak RBI spot/environment is still a weak RBI play

### Singles
- Contact rate / strikeout % (low K% matters more here than anywhere else — a single requires putting the ball in play)
- Groundball % and line drive % — most singles come off balls that stay in the infield/short outfield, not fly balls
- Sprint speed — infield hits and beating out throws are a real share of singles
- xBA on non-power contact (separate from overall xBA, which power hitters can inflate via XBH)
- Pitcher's groundball rate allowed / hit rate allowed

### Doubles
- Gap power — hard-hit line drives/fly balls that reach the gaps but don't clear the fence (different from HR-track barrels)
- Sprint speed — legging out a double matters more than for singles
- Park factor specific to doubles — foul territory size and gap dimensions, separate from the HR park factor number
- xSLG / extra-base hit rate, oppo-field power for hitters who spray to the gaps rather than pull

### Pitcher strikeouts
- K/9 and swinging-strike % (whiff rate)
- CSW% (called strikes + whiffs) — better predictor than K/9 alone since it captures called third strikes too
- Opposing team's strikeout rate, especially against this pitcher's primary pitch types
- Platoon splits (pitcher's K rate vs. lefties vs. righties)
- Expected innings/pitch count — more innings pitched means more strikeout opportunities, tied to recent workload and bullpen usage patterns
- Recency-weighted form on K rate, same shrinkage-for-small-samples logic as the hitting props

## Decision: cross-prop normalization — probability + market edge

Each prop type's model outputs an actual win probability, not an arbitrary score. Example: "68% chance of 1+ hits," "14% chance of a HR," not "92/100." Because every prop type is expressed in the same units (a real probability), any leg can be compared to any other leg regardless of prop type — a hits pick and a total-bases pick can be stacked into one parlay on equal footing.

Layer two: convert the sportsbook's odds on that same leg into an implied probability (strip the vig), then compare to the model's probability. The gap between the two is the edge. Rank every candidate leg — across all prop types — by edge, not by raw probability. This is what actually finds the best picks, not just confident ones, but the ones where the model disagrees with the market the most.

This is the chosen approach for ranking/selecting legs across prop types. Not yet implemented — still in the design/notes phase.

## Decision: live odds source — The Odds API

Sportsbooks for edge calculation: **FanDuel and DraftKings, both**. Neither has a public API, so odds come from a third-party aggregator instead of scraping (which would violate their ToS).

Using both as cross-references, not picking one over the other: if a given prop line isn't posted yet at one book, the other can still be used. Same idea applies to comparing — if both books have a line, the edge-ranking layer can check both rather than being blind to whichever book PropSync didn't pick.

[The Odds API](https://the-odds-api.com/) covers both FanDuel and DraftKings and has MLB player prop markets that map directly to PropSync's prop types:
- `batter_home_runs`
- `batter_hits`
- `batter_total_bases`
- `batter_rbis`
- `batter_singles`
- `batter_doubles`
- `pitcher_strikeouts`
- (also available if useful later: `batter_triples`, `batter_walks`, `batter_strikeouts` (batter K's, not currently in scope), `batter_runs_scored`, `batter_stolen_bases`, other pitcher props)

Access pattern: player props require the per-event endpoint (`/events/{eventId}/odds`), one game at a time — not the bulk odds endpoint used for standard markets like moneyline/totals.

Pricing: free tier = 500 credits/month (enough to prototype the pipeline). Paid tiers start at $30/mo for 20,000 credits if usage grows. Player props likely cost more credits per call than standard markets since each request is scoped to one event — exact player-prop credit cost not yet confirmed, check API docs before scaling usage.

Not yet implemented — this is the chosen provider, no integration built.

## Decision: display platform — Streamlit Community Cloud

Free hosting for the dashboard/picks display. Chosen because it's purpose-built for Python data apps (no separate frontend to build), it's free, and it's the same tool Kasper uses for his own MLB breakdown.

Limits: 3 free apps per account, 1GB resource cap per app, requires a public GitHub repo (so the scoring model code is visible to anyone — confirmed this is fine for now).

If usage outgrows the free tier or privacy becomes a concern later, the fallback is Render's free tier (750 free hours/month, private repos allowed), but that means building the web layer manually instead of getting Streamlit's dashboard for free.

Not yet implemented — this is the chosen platform, no app built.
