# PropSync Scoring Model

Per-prop probability layer. Takes player/pitcher rate stats (from the
data pipeline, once it's pulling live data) plus a specific betting line,
and outputs a real win probability for that line — e.g. "68% chance of
1+ hits," not a 0-100 arbitrary score. This is what the next layer
(market edge, not built here) will compare directly against the
sportsbook's implied probability for the same line.

Scope: scoring only. No market-edge/ranking logic, no dashboard. That's
explicitly a separate task.

## Zero third-party dependencies

Everything here (Poisson CDF, log5, shrinkage) is implemented against
the Python stdlib (`math`) — see `probability_utils.py`. No scipy
required, no `pip install` needed to run this layer or its tests. This
was a practical choice, not just a style preference: the build sandbox
has no outbound network access (same constraint the data_pipeline layer
hit — see `../[C] data_pipeline/README.md`), so depending on a package
that can't be installed and verified here would mean shipping unverified
code. `requirements.txt` lists scipy as optional, for anyone who wants to
cross-check the hand-rolled Poisson math against `scipy.stats.poisson` in
a notebook — nothing in this package imports it.

## File layout

```
[C] scoring_model/
├── README.md                    this file
├── requirements.txt              no required deps; scipy optional/unused
├── probability_utils.py          Poisson aggregation, shrinkage, log5, form-blending — shared math
├── league_constants.py           league-average baseline rates + stabilization points
├── expected_opportunities.py     expected PA (batters) / expected batters-faced (pitchers)
├── home_runs.py                  P(1+ HR)
├── hits.py                       P(hits > line)
├── total_bases.py                P(total bases > line)
├── rbis.py                       P(RBIs > line)
├── singles.py                    P(singles > line)
├── doubles.py                    P(doubles > line)
├── pitcher_strikeouts.py         P(strikeouts > line)
└── tests/
    └── test_scoring_model.py     runnable unittest suite, synthetic fixtures
```

Each prop module exposes a `score_<prop>_prop(inputs, line)` function
(home runs uses `score_home_run_prop(inputs)` — no line parameter, since
it's a yes/no "1+ HR" market, not an over/under) plus an `*Inputs`
dataclass documenting every field it needs.

## Statistical approach (shared across all 7 props)

1. **Per-opportunity rate.** Every prop is modeled as a count of events
   (hits, total bases, RBIs, K's, etc.) accumulated across a number of
   independent opportunities (plate appearances for batters, batters
   faced for pitchers). The core unit is therefore a *rate per
   opportunity* — e.g. HR per PA, not HR per season.

2. **Shrinkage toward league average.** A player's raw observed rate gets
   pulled toward a league-average baseline using an empirical-Bayes-style
   formula (`probability_utils.shrink_rate`):

   ```
   shrunk_rate = (observed_rate * n + league_avg * k) / (n + k)
   ```

   `n` = the player's actual sample size (PAs/batters faced), `k` = a
   "stabilization point" — how many observations it takes before a
   player's own rate should meaningfully outweigh the league-average
   prior. Different stats stabilize at different sample sizes (well
   documented in sabermetric reliability research): swing-and-miss
   outcomes (strikeouts) stabilize fast because they depend mostly on the
   player's own decisions; HR rate and BABIP-driven outcomes (hits,
   singles, doubles) stabilize much more slowly because batted-ball luck
   plays a bigger role. See `league_constants.py` for the specific
   stabilization point used per stat, with a one-line rationale each.

   **Important fix made during testing:** the secondary contact-quality /
   power multipliers (barrel%, xSLG, ISO, CSW%, gap-power composite, etc.)
   are *also* sample-size-dependent — a 10-PA sample's barrel% is just as
   noisy as its raw HR rate. Early versions of this code shrunk the raw
   rate but applied those multipliers at full, unshrunk strength, which
   let a small-sample hot streak's inflated barrel%/xSLG numbers
   re-introduce overconfidence even after the raw rate was correctly
   shrunk — caught by the `test_hot_streak_rookie_does_not_spike` test
   (a 10-PA, 30%-observed-HR-rate rookie was outranking an established,
   550-PA slugger). Fixed by shrinking every secondary multiplier toward
   1.0 (neutral) using the *same* shrinkage weight computed for the raw
   rate, in every prop module. This is the single most important
   correctness detail in this codebase — if you add a new derived
   metric/multiplier to any prop module later, shrink it the same way.

3. **Recency-weighted form.** `probability_utils.blend_form()` combines a
   season-long rate and a recent-window (15/30-day, from the pipeline's
   `rolling_form.py`) rate into one "current form" rate, weighting each
   recent-window observation more heavily than each season observation
   (`recent_weight_multiplier`, default 2.0 — a judgment call, see below).
   The blended rate's *effective sample size* fed into shrinkage is the
   unweighted `season_n + recent_n` — the recency tilt affects the rate,
   not the confidence the model has earned in that rate, so a hot streak
   built on very few recent PAs still gets shrunk hard even after the
   recency weighting is applied.

4. **Batter-pitcher matchup blending: log5.** A hitter's raw season rate
   isn't enough — the opposing pitcher's susceptibility matters (per the
   build instructions). `probability_utils.log5()` implements Bill James'
   log5 method, anchored to the league-average rate for that stat:

   ```
   log5(A, B, lg) = (A*B/lg) / (A*B/lg + (1-A)*(1-B)/(1-lg))
   ```

   Chosen over a naive average specifically because log5 correctly
   handles extremes — a great hitter facing a great pitcher lands close
   to league average, not at the arithmetic midpoint of two extreme
   numbers, which is what a naive average would (incorrectly) produce.

5. **Poisson aggregation across expected opportunities.**
   `probability_utils.poisson_sf` / `prob_over_line` / `prob_at_least_one`
   take the final per-opportunity rate, multiply by the player's expected
   number of opportunities for the game (`expected_opportunities.py`), and
   convert that into a probability of clearing a specific line:
   - Over/under props (hits, TB, RBIs, singles, doubles, pitcher Ks):
     `P(count > line) = 1 - PoissonCDF(floor(line), lambda)`. MLB lines
     are always posted as X.5 so there's no push case to handle.
   - HR (1+ yes/no market): `P(count >= 1) = 1 - exp(-lambda)`, the
     standard Poisson "at least one event" form.

## Per-prop method summary

| Prop | Batter signal | Pitcher signal | Extra adjustments |
|---|---|---|---|
| **Home runs** | HR/PA (shrunk + form-blended), nudged by barrel%/xSLG (also shrunk) | HR/9 → per-PA, nudged by barrel% allowed | log5 blend, then **park factor (HR-specific)** × **weather multiplier** |
| **Hits** | Hit/PA, nudged by xBA/hard-hit% (contact quality, not power — per the notes) | WHIP → per-PA hit rate (or direct opponent BA-allowed if available) | log5 blend, optional platoon multiplier |
| **Total bases** | TB/PA (≈ ISO/xSLG-informed expected bases per PA) | Composite of hits/9 + HR/9, weighted for extra-base value | log5 blend, then **composite hit-type-specific park factor** (1B/2B/3B/HR weighted, NOT the blended HR number) |
| **RBIs** | RBI/PA (shrunk — the model treats some real signal even though RBIs are context-heavy) | Composite "damage allowed" (hits/9 + HR/9) | log5 blend, then **on-base-ahead multiplier** × **team-implied-run-total multiplier** (both dampened by a sensitivity factor, not full linear swing) |
| **Singles** | 1B/PA, nudged by groundball%/line-drive%/sprint speed/approximated non-power xBA | Hits/9 scaled by groundball% allowed, then singles-share-of-hits applied | log5 blend |
| **Doubles** | 2B/PA, nudged by a gap-power composite (hard-hit% + LD% + FB% + xSLG) and sprint speed (separate weight — different skill from gap power) | Hits/9 × league doubles-share-of-hits, nudged by barrel% allowed | log5 blend, then **doubles-specific park factor** (NOT the HR park factor) |
| **Pitcher strikeouts** | — | K/9 → per-batter-faced (shrunk + form-blended), nudged by CSW% (heavier weight than other props' secondary metrics — notes call CSW% a better predictor than K/9 alone) | log5 blend against opposing lineup's K%, aggregated across **expected batters faced** (tied to expected innings, not PAs) |

## Judgment calls made (and why)

These are places where the Scoring Framework Notes specify an input or a
general approach but not an exact formula, or where the data pipeline
doesn't yet expose a clean column for something the notes call for. Each
is also flagged inline in the relevant module's docstring.

1. **Power/contact-quality metrics aren't a fitted regression — they're a
   bounded multiplier on the raw observed rate.** Barrel%/xSLG (HR), xBA/
   hard-hit% (hits), ISO/xSLG (total bases), groundball%/LD%/sprint speed
   (singles), gap-power composite/sprint speed (doubles), and CSW%
   (pitcher Ks) are all blended in as a *secondary, weighted multiplier*
   on top of the player's own raw per-opportunity rate, rather than fit
   into the rate directly via regression. There's no historical
   outcomes dataset available yet to fit real regression weights against
   (the pipeline hasn't pulled live data) — so this uses fixed, documented
   weights (e.g. `POWER_METRIC_WEIGHT = 0.35` in `home_runs.py`) chosen to
   be a meaningful nudge without letting a secondary metric override the
   primary observed rate. **Revisit once real season data is available**:
   the right move then is to backtest these weights against actual
   outcomes and adjust, or replace the multiplier approach with a proper
   logistic regression per prop type.

2. **Recency weight multiplier (`recent_weight_multiplier = 2.0` in
   `blend_form`).** One recent-window PA counts as much as two season PAs
   when blending season + 15/30-day form. No single textbook-correct
   value exists for this; 2.0 is a moderate tilt — meaningful but not
   dominant. Worth tuning once there's a way to backtest against real
   results.

3. **RBI context multipliers are dampened linear scalars, not a full
   run-expectancy model.** A real run-expectancy-by-base/out-state model
   would need PA-level base/out data, which the pipeline doesn't expose
   and which is arguably overkill for a props model. Instead,
   `on_base_ahead_multiplier` and `team_run_total_multiplier` are linear
   scalars around 1.0, each dampened by a `*_SENSITIVITY = 0.6` factor so
   neither single context input can swing the estimate too aggressively.

4. **Total bases and "expected bases per PA" modeled as Poisson, even
   though a single PA's true outcome is a small discrete distribution
   over {0,1,2,3,4} bases, not literally Poisson-distributed.** Poisson's
   mean still equals the correct expected total across PAs, and for the
   over/under lines actually posted (1.5, 2.5, 3.5) this approximation is
   standard practice in public sports-analytics modeling. Flagged in
   `total_bases.py`'s docstring as an approximation, not treated as
   exact.

5. **"xBA on non-power contact" (singles props) is approximated, not
   pulled from a real split.** The notes call for this specific metric,
   but neither FanGraphs nor Savant's leaderboard exports a non-power-
   contact-specific xBA — it's all one blended number. `singles.py`
   approximates it by discounting overall xBA proportional to how far
   above league-average the batter's ISO is (more power → more of their
   hard contact becomes XBH rather than singles → their blended xBA
   overstates their singles-specific rate). Revisit if a raw-Statcast
   derivation (excluding batted balls that became HR/2B/3B) gets built
   into the pipeline later — `statcast/savant.py`'s pattern of deriving
   non-leaderboard metrics from raw pitch-by-pitch data is the template
   for doing that properly.

6. **Total-bases-allowed and doubles-allowed for pitchers are built from
   hits/9 + HR/9 composites, not a real extra-base-hit-allowed
   breakdown.** The pipeline doesn't currently expose pitcher-side 2B/3B
   allowed rates directly (only aggregate hits/9 and HR/9 from FanGraphs'
   pitching leaderboard). Approximated using league-average hit-type
   shares (e.g. "~19% of hits allowed are doubles") applied to the
   pitcher's general hits-allowed rate, nudged by barrel% allowed where
   available.

7. **Expected PA / expected batters faced are fixed lookup tables, not
   pulled from the pipeline.** See `expected_opportunities.py`:
   - Batters: `EXPECTED_PA_BY_LINEUP_SPOT` is a fixed table of
     league-average PA-per-game by batting order slot (1 → 4.6 PA, 9 →
     3.7 PA). The pipeline doesn't currently expose a per-game lineup feed
     — every prop module takes `batter_lineup_spot` as an input parameter
     rather than hardcoding this internally, so a real lineup feed can be
     wired in later with zero changes to the prop modules themselves.
   - Pitchers: `expected_batters_faced()` defaults to 5.5 expected innings
     (`DEFAULT_EXPECTED_INNINGS`) × ~4.3 batters/inning if no
     `expected_innings` override is passed. This is the single biggest
     real uncertainty in the pitcher strikeouts prop specifically — a
     pitcher pulled after 4 innings instead of 6 loses a third of his
     strikeout opportunities regardless of stuff quality. **Top priority
     pipeline gap to close** if/when workload/pitch-count projection data
     becomes available (the notes call this out explicitly: "expected
     innings/pitch count...tied to recent workload and bullpen usage
     patterns").

8. **League-average baseline rates (`league_constants.py`) are hardcoded
   constants, not pulled live.** Approximate modern-era (2021-2025) MLB
   averages. Once the pipeline is live, the better move is computing
   these from an actual league-wide aggregate each season
   (`get_batter_season_stats()` summed across all qualified batters) rather
   than a fixed constant — the swap-in point is exactly these module-level
   constants in `league_constants.py`, nothing else needs to change.

9. **Weather is a caller-supplied multiplier, not sourced.** Per the data
   pipeline README, same-day weather (wind/temp) was explicitly flagged as
   an open gap — no free API was identified. `home_runs.py` accepts
   `weather_hr_multiplier` (default 1.0/neutral) so the HR module is ready
   the moment a weather source exists; nothing in this layer needs to
   change, only the caller needs to start passing a real value.

## Exactly how this plugs into the data pipeline

The data pipeline (`../[C] data_pipeline/`) has not yet been run against
live data (no internet access in the build sandbox — see its README).
This scoring layer was built against its **documented** interface, not
live output, per the task instructions: this layer shouldn't care
whether the data is real or synthetic, only that it matches the expected
shape. Here's the exact mapping from pipeline output to scoring-model
input, and what to watch for once real data is confirmed:

### Batter inputs (home_runs / hits / total_bases / rbis / singles / doubles)

| Scoring model field | Pipeline source | Watch for |
|---|---|---|
| `batter_*_per_pa_season` | Derive from `statcast.savant.get_batter_season_stats()` — divide the relevant counting stat by PA. **The FanGraphs JSON API columns aren't guaranteed** (see pipeline README: column names depend on what FanGraphs' API actually returns). Confirmed-expected columns per the pipeline docs: `Barrel%`, `HardHit%`, `FB%`, `SwStr%`, `xSLG`, `xBA` (renamed from `xAVG`), `ISO`. **HR/Hit/1B/2B/RBI counting columns aren't explicitly listed as confirmed in the pipeline README** — print `.columns` on the real DataFrame first; likely candidates are `HR`, `H`, `1B`, `2B`, `RBI`, `PA` but unconfirmed until live. |
| `batter_*_per_pa_recent`, `batter_pa_recent` | `statcast.rolling_form.batter_rolling_batted_ball_profile(window_days)` — returns `barrel_pct`, `hard_hit_pct`, `fly_ball_pct`, `groundball_pct`, `line_drive_pct` per the pipeline code, but **does NOT currently return raw HR/hit/1B/2B counts or PA** — only the batted-ball profile. **Gap to close**: this function needs to also aggregate raw counting stats (HR, H, 1B, 2B, RBI) per batter within the window before the scoring model's recency-blending can actually use real rolling counts — right now `rolling_form.py` only derives batted-ball-profile metrics (barrel%, hard-hit%, FB/GB/LD%), not box-score counting stats. |
| `batter_barrel_pct`, `batter_xslg`, `batter_xba`, `batter_iso`, `batter_hard_hit_pct` | `get_batter_season_stats()` columns directly (`Barrel%`, `xSLG`, `xBA`, `ISO`, `HardHit%`) | Same column-name caveat as above — confirm against the live DataFrame. |
| `batter_groundball_pct`, `batter_line_drive_pct`, `batter_fly_ball_pct` | `get_batter_season_stats()` (season) or `rolling_form.batter_rolling_batted_ball_profile()` (rolling) — both expose these per the pipeline docs | — |
| `batter_sprint_speed` | `statcast.savant.get_sprint_speed(season)` | Pipeline README doesn't confirm the exact column name pybaseball's `statcast_sprint_speed()` returns — check `.columns` on first live run. |
| `batter_lineup_spot` | **Not sourced anywhere in the pipeline currently.** | This is a real gap — no lineup-feed puller exists yet. Until one is built, callers must supply this manually (e.g. from a daily lineups scrape/API) or omit it and accept the `DEFAULT_EXPECTED_PA` fallback in `expected_opportunities.py`. |
| `obp_of_hitters_ahead`, `team_implied_run_total` (RBI prop only) | **Not sourced anywhere in the pipeline currently.** | `obp_of_hitters_ahead` needs a lineup feed (same gap as above) plus a way to compute the OBP of the specific 1-3 hitters batting ahead. `team_implied_run_total` needs either The Odds API's team totals market (not currently in `PLAYER_PROP_MARKETS`) or a separate Vegas totals source — flagged as a real open gap, not just a column-name issue. |

### Pitcher inputs (all batting props' pitcher side + pitcher_strikeouts)

| Scoring model field | Pipeline source | Watch for |
|---|---|---|
| `pitcher_hr_per_9`, `pitcher_k_per_9`, `pitcher_barrel_pct_allowed` | `statcast.savant.get_pitcher_season_stats()` — confirmed-expected columns: `K/9`, `HR/9`, `Barrel%`, `SwStr%` | Same FanGraphs-API column-naming caveat as the batter side. |
| `pitcher_csw_pct` | `statcast.savant.compute_csw_rate(raw_statcast_df)` — needs a raw Statcast pull via `get_raw_statcast()` first, or the rolling version via `rolling_form.pitcher_rolling_profile()` | Requires raw pitch-by-pitch data, not a leaderboard pull — slower, and depends on `description` column values matching Statcast's documented vocabulary (`called_strike`, `swinging_strike`, `swinging_strike_blocked`). |
| `pitcher_whip`, `pitcher_hits_per_9` | `get_pitcher_season_stats()` — **WHIP and `H/9`/hits-allowed aren't in the pipeline README's confirmed-column list** (only K/9, HR/9, Barrel%, SwStr% are explicitly confirmed). Check `.columns` for `WHIP`, `H/9`, or similar on first live run. |
| `pitcher_groundball_pct_allowed` | `get_pitcher_season_stats()` — FanGraphs pitching leaderboards typically expose `GB%`, but **not explicitly confirmed in the pipeline docs**. | Verify column name live. |
| `opponent_team_k_pct` (pitcher Ks prop) | **Not directly sourced.** Would need a team-level aggregate of `get_batter_season_stats()` (sum/group by team, compute team K rate) — straightforward to build once batter season stats are confirmed live, but no dedicated team-aggregate function exists yet in the pipeline. | Build a small aggregation helper once batter data is live; not a column-mismatch issue, just an unbuilt convenience function. |
| `expected_innings` (pitcher Ks prop) | **Not sourced anywhere in the pipeline.** | Real gap, called out explicitly above (judgment call #7) and in the Scoring Framework Notes themselves ("tied to recent workload and bullpen usage patterns... not yet implemented"). |

### Park factors (home_runs, total_bases, doubles)

| Scoring model field | Pipeline source | Watch for |
|---|---|---|
| `park_hr_factor` | `park_factors.park_factors.get_hr_park_factor(season)` → `hr_park_factor` column | Pipeline README flags that the *exact* FanGraphs Guts table column names (`1B`, `2B`, `3B`, `HR`, etc.) are confirmed live via direct page fetch, but pybaseball isn't used here at all (it's HTML-scraped directly) — should be low-risk, but confirm `_find_column()`'s candidate list still matches if FanGraphs ever renames a column. |
| `park_factor_1b`, `park_factor_2b`, `park_factor_3b`, `park_factor_hr` (total_bases) | `park_factors.get_park_factors(season)` — full table, slice the four columns | Same as above. |
| `park_2b_factor` (doubles) | `park_factors.get_extra_base_park_factors(season)` → `doubles_park_factor` column | Same as above. |

### Weather (home_runs only)

`weather_hr_multiplier` has no pipeline source at all currently — see
judgment call #9. This is a genuinely open data gap, not a column-name
issue.

### Odds API (not used by this layer directly)

The Odds API client (`odds/odds_api_client.py`) outputs `line` and
`price` per player/market — that's exactly the input the **next layer**
(market edge, out of scope here) will need: take this scoring layer's
probability output for the same `(player, market, line)` tuple the Odds
API returns, convert the API's `price` into an implied probability, and
compare. Nothing in this scoring layer consumes the Odds API directly.

## Running the tests

```bash
cd "[C] scoring_model"
python3 tests/test_scoring_model.py
```

39 tests, stdlib `unittest`, no pip install needed. Covers, for every
prop type: probability bounds (`[0, 1]`), monotonicity in the expected
direction (better batter stat → higher probability; tougher matchup →
lower probability; park factor → higher probability when hitter-friendly;
higher line → lower probability), and the shrinkage sanity check
(`test_hot_streak_rookie_does_not_spike` — a 10-PA hot-streak rookie must
not outrank an established large-sample slugger).
