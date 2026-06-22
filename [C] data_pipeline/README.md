# PropSync Data Pipeline

Raw data layer for PropSync. Pulls Statcast/FanGraphs batter & pitcher
metrics, park factors split by hit type, and (once a key is added)
FanDuel + DraftKings player prop odds from The Odds API. This is sourcing
only — no scoring, no edge calculation, no dashboard. That's the next
layer, built on top of what's here.

## IMPORTANT — read this first: what's actually been tested

This pipeline was built in a sandboxed environment with **no outbound
internet access from Python/pip** (the shell's proxy blocked PyPI,
Baseball Savant, FanGraphs, and every other external host — confirmed by
testing `pip install`, `curl`, and `requests.get()` against multiple
hosts, all of which failed at the proxy layer with `403 Forbidden`).

So, honestly:

- **pybaseball could not be installed or run in the build environment.**
  No live Python test pull happened.
- A separate web-fetch tool (not Python, not subject to the same proxy)
  *was* used to confirm the underlying data sources are real, live, and
  shaped the way the code assumes:
  - Fetched `baseballsavant.mlb.com/statcast_search/csv` directly and got
    a real `200 OK` with `Content-Type: application/download` and actual
    CSV bytes back — the endpoint works and is reachable with no auth.
  - Fetched `fangraphs.com/tools/guts?type=pf` and saw the live, current
    (2025 season, updated June 2026) park factors table — confirmed it
    really is split by hit type (separate 1B/2B/3B/HR/SO/BB/GB/FB/LD
    columns per team), not one blended number, which is exactly what the
    Scoring Framework Notes called for.
- **None of this proves the `pybaseball` Python package itself installs
  cleanly or that its specific function signatures/column names match
  what this code assumes.** pybaseball's API has shifted across versions
  before. The code is written against pybaseball's documented/typical
  interface as of its public docs, but the very first thing to do on a
  real machine is run `python run_pipeline.py` and look at the printed
  output — if a column name doesn't match, it'll be an easy, obvious fix
  with the real DataFrame in front of you.

**Bottom line: the code is built correctly and the data sources are
confirmed real and reachable, but the actual `pip install pybaseball` +
live pull has not been executed end-to-end yet. Do that first, on your
own machine, before building anything on top of this.**

## What's implemented

### Statcast / FanGraphs (no auth needed) — `statcast/`, `park_factors/`

| File | What it does |
|---|---|
| `statcast/savant.py` | Core pybaseball wrappers: season batting/pitching stats (barrel%, hard-hit%, fly ball%, swinging-strike%, xSLG, xBA, ISO, K/9, HR/9, etc.), Savant expected stats, sprint speed. Also derives pulled-air rate, CSW%, and fastball-in-zone rate from raw pitch-by-pitch data since those have no pre-built leaderboard. |
| `statcast/savant_csv_fallback.py` | Direct GET against `baseballsavant.mlb.com/statcast_search/csv` — bypasses pybaseball entirely. Use only if pybaseball doesn't expose something needed (documented per-metric in `savant.py`'s docstring). |
| `statcast/rolling_form.py` | Trailing 15/30-day windows, re-aggregated from raw Statcast pulls since FanGraphs only offers season totals, not rolling windows. |
| `park_factors/park_factors.py` | FanGraphs Guts! park factors, split by hit type (1B/2B/3B/HR get separate numbers). Confirmed live and current via direct page fetch — see "what's tested" above. |

### The Odds API (FanDuel + DraftKings player props) — `odds/`

| File | What it does |
|---|---|
| `odds/odds_api_client.py` | **Fully built, cannot be tested without a key.** `get_todays_events()` lists today's MLB games and event IDs. `get_event_player_props(event_id)` hits the per-event endpoint (`/events/{eventId}/odds`) — required for player props, since the bulk `/odds` endpoint only covers standard markets like moneyline/totals. `get_all_player_props_today()` chains both into one clean DataFrame: event, market, player, line, price, bookmaker. Pulls both FanDuel and DraftKings by default — each row is tagged with its `bookmaker` so downstream logic can use whichever book has a given line posted, or compare both if both do. |

Market keys wired up (from `config.py`, matches the Scoring Framework
Notes exactly): `batter_home_runs`, `batter_hits`, `batter_total_bases`,
`batter_rbis`, `batter_singles`, `batter_doubles`, `pitcher_strikeouts`.

### Shared

| File | What it does |
|---|---|
| `config.py` | Loads `.env`, exposes `ODDS_API_KEY`, season detection, market list, target bookmakers (FanDuel + DraftKings), rolling window sizes (15/30 days). |
| `run_pipeline.py` | Manual entry point — pulls everything testable, prints sanity-check samples, saves CSVs to `data/raw/`. Skips the Odds API section cleanly (with a message, not a crash) if no key is set. |

## What's stubbed pending your API key

Everything in `odds/odds_api_client.py`. The code is complete and should
work the moment a real key is in place — nothing else needs to change.

**Your next action:**
1. Go to https://the-odds-api.com/ and sign up for a free key (500
   credits/month free tier).
2. In this folder, copy `.env.example` to `.env`:
   ```
   cp .env.example .env
   ```
3. Open `.env` and replace `your_api_key_here` with your real key.
4. Run `python run_pipeline.py` — section 6 will switch from "SKIPPED" to
   a live pull of today's FanDuel + DraftKings player prop odds.

`.env` is gitignored — your key will never get committed.

## Metrics from the notes that needed a fallback / derivation

- **Pulled-air rate / pulled fly ball %** — not exposed by any pybaseball
  leaderboard wrapper. Derived in `savant.compute_pulled_air_rate()` from
  raw Statcast hit-coordinate data (`hc_x`/`hc_y` + batter handedness). If
  pybaseball's raw schema ever drops those columns, `savant_csv_fallback.py`
  pulls the same fields directly from Savant's CSV endpoint.
- **CSW% (called strikes + whiffs)** — not a native FanGraphs/pybaseball
  column. Derived in `savant.compute_csw_rate()` from raw pitch
  `description` values.
- **Fastball-in-zone rate** — not a pre-built leaderboard stat. Derived in
  `savant.compute_fastball_zone_rate()` from raw pitch type + zone data.
- **Platoon splits** — `savant.get_platoon_splits()` is a stub that raises
  `NotImplementedError` with instructions, because pybaseball's splits
  API has varied across versions and the exact function signature needs
  to be confirmed against whatever version actually installs on your
  machine. The manual fallback (documented in the function) is to pull
  raw Statcast and group by `stand` (vs. LHP/RHP for batters) or
  `p_throws` (vs. LHB/RHB for pitchers) — straightforward once you can
  actually run pybaseball and see real column names.
- **Weather (wind/temp)** — mentioned in the Scoring Framework Notes for
  HR props but explicitly described there as "same-day" / not static like
  park factors. Not sourced in this pass — no free API was named in the
  notes, and it's arguably closer to a same-day scoring input than a raw
  data-layer concern. Flagging it as an open gap rather than guessing at
  a weather provider.

## How to run this for real

```bash
cd "PropSync/02 Projects/PropSync/[C] data_pipeline"
pip install -r requirements.txt
python run_pipeline.py
```

First run will likely surface small mismatches between what this code
assumes about pybaseball's column names and what your installed version
actually returns — that's expected and normal for a brand-new install,
not a sign the architecture is wrong. Fix forward: print `.columns` on
whatever DataFrame errors out, update the column-name references in that
one function, move on.

For the odds side, once your key is in `.env`, you can also call the
client directly:

```python
from odds.odds_api_client import get_all_player_props_today
df = get_all_player_props_today()
print(df.head(20))
```

## Folder structure

```
[C] data_pipeline/
├── README.md
├── requirements.txt
├── .env.example          (placeholder key, safe to commit)
├── .env                  (your real key — gitignored, never commit)
├── .gitignore
├── config.py              shared settings, env loading, market list
├── run_pipeline.py        manual entry point / smoke test
├── statcast/
│   ├── savant.py                  pybaseball wrappers + derived metrics
│   ├── savant_csv_fallback.py     direct Savant CSV endpoint fallback
│   └── rolling_form.py            15/30-day rolling windows
├── park_factors/
│   └── park_factors.py            FanGraphs park factors by hit type
├── odds/
│   └── odds_api_client.py         FanDuel player props (needs API key)
└── data/raw/              pulled CSVs land here when you run things
```
