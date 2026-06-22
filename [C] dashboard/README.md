# PropSync Dashboard

The Streamlit front end. Displays two things:

1. **Best Picks** — the ranked "best picks" table across all 7 prop
   types, built from `edge_ranking`'s `EdgeCandidate` output, plus a
   top-N parlay builder (`select_top_n()`).
2. **Raw Stats Showcase** — a Kasper-style browsable table of the
   underlying player stats: barrel %, hard-hit %, fly ball %,
   swinging-strike %, handedness splits, park factors split by hit type,
   pulled-air rate, sprint speed.

Scope: display only. Does not modify `data_pipeline/`, `scoring_model/`,
or `edge_ranking/` — only imports and calls into them (see `sample_data.py`
and `data_loader.py` for exactly how).

## Status: running on sample data, not live data

Same situation as the rest of the project (see
`../[C] data_pipeline/README.md`): no Odds API key has been entered yet,
and pybaseball has never been run live (no outbound internet access in any
build sandbox so far — confirmed again here: `pip install streamlit`
fails at the proxy layer with the same `403 Forbidden` the data_pipeline
README documents). So this dashboard runs against a **fabricated but
realistic** sample dataset by default, clearly labeled in the UI with a
banner: *"Showing sample data — live data pipeline not yet connected."*

The sample data isn't a fake lookalike of the picks table — it's real
current MLB player names with fabricated-but-plausible stat lines, run
through the **actual** `scoring_model` and `edge_ranking` code
(`score_<prop>_prop()`, `build_candidate_legs()`, `rank_edges()`,
`select_top_n()`, real de-vig math). Only the inputs are fake; the
pipeline producing the picks table is the real one.

## How to run this locally

You need a machine with normal internet access (this sandbox doesn't have
it — see above).

```bash
cd "PropSync/02 Projects/PropSync/[C] dashboard"
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser, normally at `http://localhost:8501`.

## Single-page-with-tabs, not multipage

This uses one `app.py` with `st.tabs()` for the two views, not
Streamlit's multipage `pages/` folder convention. Two views, both reading
off the same daily slate, neither big enough to need its own sidebar nav
entry — tabs keep this one file to read and reason about. If a third view
gets added later (e.g. a per-game breakdown, a backtest/results page),
that's the point to reconsider multipage.

## How to deploy free on Streamlit Community Cloud

Per the Scoring Framework Notes' platform decision (free, same tool
Kasper uses):

1. **Push this repo to a public GitHub repo.** Streamlit Community Cloud
   requires a public repo on the free tier — this means the scoring
   model/edge-ranking code becomes visible to anyone, which the notes
   already confirmed is fine for now. (Don't commit `[C] data_pipeline/.env`
   — it's gitignored already; double check it stays that way before
   pushing, since it would hold a real Odds API key once one exists.)
2. Go to [share.streamlit.io](https://share.streamlit.io/), sign in with
   GitHub, click "New app."
3. Point it at this repo, branch, and set the **main file path** to:
   ```
   02 Projects/PropSync/[C] dashboard/app.py
   ```
   (adjust if the repo root differs from this workspace's folder structure).
4. Streamlit Cloud installs from this folder's `requirements.txt`
   automatically — no extra config needed for the sample-data version,
   since it has zero network/secret dependencies.
5. **Once live data is wired in** (see "Switching to live data" below),
   add the Odds API key as a Streamlit Cloud "Secret" (Settings → Secrets)
   rather than committing `.env` — `data_pipeline/config.py` already reads
   from environment variables via `python-dotenv`, and Streamlit secrets
   are injected as environment variables the same way.

**Free-tier limits to know about** (per the Scoring Framework Notes):
3 free apps per account, 1GB resource cap per app. If PropSync outgrows
this (e.g. live odds pulls + a bigger slate push memory past 1GB), the
documented fallback is Render's free tier (750 hours/month, private repos
allowed) — but that means building the web-serving layer manually instead
of getting Streamlit's dashboard UI for free.

## Folder structure

```
[C] dashboard/
├── README.md                 this file
├── requirements.txt           streamlit + pandas only
├── app.py                      Streamlit UI — two tabs (Best Picks, Raw Stats)
├── data_loader.py               THE SEAM — swap sample data for live pipeline data here
├── sample_data.py                fabricated players/stats/odds, run through REAL scoring/edge code
├── formatting.py                 pure pandas helpers — table shaping, filters, no Streamlit imports
└── tests/
    └── test_dashboard_helpers.py   runnable unittest suite, 39 tests
```

## The seam: switching from sample data to live data

**`data_loader.py` is the only file that needs to change.** Its docstring
has the exact step-by-step (also summarized here):

1. Set `data_loader.IS_LIVE_DATA = True` — this flips the UI banner from
   the sample-data warning to a "live data" confirmation automatically.
2. Rewrite `load_raw_stats()` to call the real `data_pipeline` functions
   (`statcast.savant.get_batter_season_stats()`,
   `park_factors.park_factors.get_park_factors()`,
   `statcast.rolling_form.batter_rolling_batted_ball_profile()`, etc.)
   instead of `sample_data.build_sample_raw_stats_df()` — return a
   DataFrame with the **same column names** `sample_data.py` uses, so
   nothing in `formatting.py` or `app.py` needs to change.
3. Rewrite `load_ranked_edges()` to:
   - Pull real odds via `odds.odds_api_client.get_all_player_props_today()`
     instead of fabricating odds rows.
   - Build a real `(player, market) -> *Inputs` registry from live
     pipeline data. This is genuinely new integration work — the
     `edge_ranking` README is explicit that building this registry isn't
     its job, and nothing else in the project has built it yet either.
     `sample_data._build_inputs_registry()` is a worked example of the
     exact shape needed (which `*Inputs` field comes from which pipeline
     call) — port that function's mapping to pull from real DataFrames
     once they exist, using the `scoring_model` README's "Exactly how this
     plugs into the data pipeline" table as the authoritative column-name
     reference (pipeline column names aren't 100% confirmed until a live
     pull happens — see that README's repeated "print `.columns` and
     check" guidance).
   - Call the real `build_candidate_legs()` + `rank_edges()` exactly like
     `sample_data.build_sample_edge_candidates()` already does — that part
     doesn't change.
4. Nothing in `app.py` or `formatting.py` needs to change either way,
   since both loader functions return the same shapes (a DataFrame, and a
   `list[EdgeCandidate]`) regardless of where the data came from.

Caching: `app.py` wraps both loader calls in `st.cache_data(ttl=600)`.
Keep that in place once real network calls are involved — each Odds API
per-event call costs real credits against the 500/month free tier, and
you don't want a 10-minute cache window turning into "every slider drag
fires a live API call."

## Testing — what's actually verified here

No live `streamlit run` browser session was available in the build
sandbox (no internet access — confirmed again by `pip install streamlit`
failing at the proxy with `403 Forbidden`, same as `data_pipeline`'s and
`scoring_model`'s build environments). What WAS verified:

1. **Syntax validity** — every `.py` file in this folder parses cleanly:
   ```bash
   python3 -c "import ast; ast.parse(open('app.py').read())"
   ```
   (and the same for `data_loader.py`, `sample_data.py`, `formatting.py`,
   `tests/test_dashboard_helpers.py`) — all pass.

2. **All non-Streamlit logic, with a real `unittest` suite (39 tests)** —
   `sample_data.py`'s fabricated data (bounds-checked, internally
   consistent, deterministic across reruns), `data_loader.py`'s seam
   functions, and every `formatting.py` transformation (the picks
   DataFrame shape, filters, the top-N parlay view, the raw-stats showcase
   tables, the park-factor-by-hit-type table). Run with:
   ```bash
   cd "[C] dashboard"
   python3 tests/test_dashboard_helpers.py
   ```

3. **The actual scoring_model + edge_ranking code, exercised end-to-end**
   — `sample_data.build_sample_edge_candidates()` doesn't fake the picks
   table; it builds real `*Inputs` dataclasses, fabricates an odds
   DataFrame shaped exactly like `get_all_player_props_today()`'s output,
   and runs both through the real `build_candidate_legs()` / `rank_edges()`
   from `edge_ranking/ranking.py`. The test suite confirms this produces
   sane output: all 7 prop types appear, edges are bounded and sorted
   correctly, same-player/same-game overlap is allowed (per the Scoring
   Framework Notes' deliberate no-exclusion decision), and both
   FanDuel/DraftKings prices get compared per-leg.

4. **`pip install streamlit` was attempted** (per the task instructions)
   and failed immediately at the proxy layer — same `403 Forbidden`
   pattern as every other `pip install` attempt across this project's
   build history. Reported here plainly rather than assumed to work.

**What's NOT verified**: the actual rendered Streamlit UI (widget
layout, whether `st.dataframe`'s `column_config` renders as intended,
whether the sidebar filters interact smoothly) — none of that can be
checked without a real `streamlit run` session in a browser. That's
exactly what cozy needs to do on his own machine (see "How to run this
locally" above) before trusting the UI is polished, not just
syntactically valid.
