"""
PropSync dashboard — Streamlit single-page app with two tabs:
  1. Picks / Edge view — the ranked "best picks" table across all 7 prop
     types, plus a top-N parlay builder.
  2. Raw Stats Showcase — the Kasper-style raw-stats browser (barrel%,
     hard-hit%, fly ball%, swinging-strike%, handedness splits, park
     factors split by hit type, pulled-air rate, sprint speed).

Single-page-with-tabs over Streamlit's multipage `pages/` convention: two
views, both reading from the same daily slate, neither complex enough to
need its own URL/sidebar nav entry — tabs keep it one mental model and one
file to scan. If this grows a 3rd or 4th view later (e.g. a per-game
breakdown), multipage is worth revisiting.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Does NOT modify data_pipeline/scoring_model/edge_ranking — only imports
and calls into them (via data_loader.py and formatting.py). See
data_loader.py for the one seam to swap sample data for live data later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Same sys.path bootstrap convention as the rest of the project
# (edge_ranking/market_map.py, dashboard/sample_data.py) so bare imports
# from scoring_model/edge_ranking resolve regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("[C] scoring_model", "[C] edge_ranking", "[C] data_pipeline"):
    _p = _PROJECT_ROOT / _folder
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import data_loader
import formatting

st.set_page_config(page_title="PropSync", page_icon="⚾", layout="wide")


# ---------------------------------------------------------------------------
# Cached data loading — wraps data_loader.py, the actual seam. Caching here
# (not in data_loader.py itself) keeps data_loader.py framework-agnostic
# and trivially testable with plain unittest (see tests/test_dashboard_helpers.py).
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _cached_raw_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    return data_loader.load_raw_stats()


@st.cache_data(ttl=600)
def _cached_ranked_edges() -> list:
    return data_loader.load_ranked_edges()


def _sample_data_banner() -> None:
    if not data_loader.IS_LIVE_DATA:
        st.warning(
            "**Showing sample data — live data pipeline not yet connected.** "
            "Player names are real (for flavor); stats, odds, and lines below "
            "are fabricated, not pulled from Baseball Savant/FanGraphs/The Odds API. "
            "See `data_loader.py` for the swap-to-live-data seam.",
            icon="⚠️",
        )
    else:
        st.success("Showing live pipeline data.", icon="✅")


# ---------------------------------------------------------------------------
# Picks / Edge view
# ---------------------------------------------------------------------------

def render_picks_view() -> None:
    candidates = _cached_ranked_edges()
    full_df = formatting.edge_candidates_to_dataframe(candidates)

    st.subheader("Best Picks — Ranked by Market Edge")
    st.caption(
        "Every leg, across all 7 prop types, ranked by edge = PropSync's model "
        "probability minus the de-vigged sportsbook market probability. "
        "No same-player or same-game exclusion — a player's HR leg and hits leg "
        "can both appear, and so can two players from the same game (deliberate "
        "design choice, see Scoring Framework Notes)."
    )

    with st.sidebar:
        st.header("Filters")
        prop_options = sorted(full_df["market"].unique()) if not full_df.empty else []
        prop_labels = {m: formatting.prop_display_name(m) for m in prop_options}
        selected_props = st.multiselect(
            "Prop type",
            options=prop_options,
            format_func=lambda m: prop_labels.get(m, m),
            default=[],
            help="Leave empty to show all prop types.",
        )

        min_edge = st.slider(
            "Minimum edge",
            min_value=-0.20, max_value=0.20, value=-0.20, step=0.01,
            help="Filter out legs below this edge. Default (-20%) shows everything.",
        )

        book_options = sorted(full_df["sportsbook"].unique()) if not full_df.empty else []
        selected_books = st.multiselect(
            "Sportsbook used",
            options=book_options,
            default=[],
            help="Leave empty to show both FanDuel and DraftKings.",
        )

        player_options = sorted(full_df["player"].unique()) if not full_df.empty else []
        selected_players = st.multiselect(
            "Player",
            options=player_options,
            default=[],
            help="Leave empty to show all players.",
        )

    filtered = formatting.filter_picks(
        full_df,
        prop_types=selected_props or None,
        min_edge=min_edge if min_edge > -0.20 else None,
        players=selected_players or None,
        sportsbooks=selected_books or None,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Legs shown", len(filtered))
    col2.metric("Avg edge", f"{filtered['edge'].mean():+.1%}" if len(filtered) else "—")
    col3.metric("Best edge", f"{filtered['edge'].max():+.1%}" if len(filtered) else "—")

    display_cols = {
        "player": "Player",
        "opponent": "Opponent",
        "prop_type": "Prop Type",
        "line": "Line",
        "side": "Side",
        "sportsbook": "Book Used",
        "chosen_book_price": "Price",
        "model_prob_pct": "Model Prob.",
        "market_prob_pct": "Market Prob. (de-vigged)",
        "edge_pct": "Edge",
        "other_book": "Other Book",
    }
    show_df = filtered[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(
        show_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Edge": st.column_config.TextColumn(help="Model probability minus de-vigged market probability"),
        },
    )

    st.divider()
    st.subheader("Top-N Parlay Builder")
    st.caption(
        "Plain top-N slice of the edge-ranked list (`select_top_n()` from "
        "edge_ranking/parlay_selector.py) — no exclusion logic. The same "
        "player or the same game can appear more than once on purpose."
    )
    n = st.slider("Number of legs", min_value=1, max_value=min(20, len(candidates) or 1), value=min(5, len(candidates) or 1))
    top_df = formatting.top_n_parlay_view(candidates, n)
    top_show = top_df[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(top_show, use_container_width=True, hide_index=True)

    if not top_df.empty:
        combined_prob = top_df["model_prob"].prod()
        st.caption(
            f"If treated as independent events (a simplification — see Scoring "
            f"Framework Notes on correlation), PropSync's model implies roughly "
            f"a **{combined_prob:.1%}** chance all {n} legs hit."
        )


# ---------------------------------------------------------------------------
# Raw Stats Showcase view
# ---------------------------------------------------------------------------

def render_raw_stats_view() -> None:
    batter_df, pitcher_df = _cached_raw_stats()

    st.subheader("Raw Stats Showcase")
    st.caption(
        "Underlying player-level data, browsable the way Kasper's MLB breakdown "
        "shows it: barrel %, hard-hit %, fly ball %, swinging-strike %, "
        "handedness splits, and park factors split by hit type (not blended)."
    )

    tab_batters, tab_pitchers, tab_parks = st.tabs(["Batters", "Pitchers", "Park Factors"])

    with tab_batters:
        team_options = sorted(batter_df["team"].unique())
        selected_teams = st.multiselect("Filter by team", options=team_options, default=[], key="batter_team_filter")
        view_df = batter_df if not selected_teams else batter_df[batter_df["team"].isin(selected_teams)]

        sort_options = {
            "Barrel %": "barrel_pct", "Hard-Hit %": "hard_hit_pct", "Fly Ball %": "fly_ball_pct",
            "SwStr %": "swinging_strike_pct", "Pulled-Air Rate": "pulled_air_rate",
            "xSLG": "xslg", "Sprint Speed": "sprint_speed_ft_s",
        }
        sort_label = st.selectbox("Sort by", options=list(sort_options.keys()), index=0, key="batter_sort")
        view_df = view_df.sort_values(sort_options[sort_label], ascending=False)

        showcase = formatting.batter_showcase_table(view_df)
        st.dataframe(showcase, use_container_width=True, hide_index=True)

    with tab_pitchers:
        p_team_options = sorted(pitcher_df["team"].unique())
        selected_p_teams = st.multiselect("Filter by team", options=p_team_options, default=[], key="pitcher_team_filter")
        p_view_df = pitcher_df if not selected_p_teams else pitcher_df[pitcher_df["team"].isin(selected_p_teams)]
        p_view_df = p_view_df.sort_values("k_per_9", ascending=False)

        p_showcase = formatting.pitcher_showcase_table(p_view_df)
        st.dataframe(p_showcase, use_container_width=True, hide_index=True)

    with tab_parks:
        st.caption(
            "Per the Scoring Framework Notes: park factor is split by hit type "
            "(1B/2B/3B/HR each get a separate number), not one blended figure — "
            "a park can suppress home runs but still inflate doubles."
        )
        pf_table = formatting.park_factor_table(batter_df)
        st.dataframe(pf_table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("⚾ PropSync")
    st.caption("MLB player prop research — ranked picks and the raw stats behind them.")
    _sample_data_banner()

    tab_picks, tab_stats = st.tabs(["📊 Best Picks", "🔎 Raw Stats Showcase"])
    with tab_picks:
        render_picks_view()
    with tab_stats:
        render_raw_stats_view()


if __name__ == "__main__":
    main()
