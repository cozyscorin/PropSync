"""
PropSync dashboard — Streamlit app with two tabs:
  1. Best Picks — ranked edge table + parlay builder
  2. Raw Stats — batter/pitcher stats showcase + park factors
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("[C] scoring_model", "[C] edge_ranking", "[C] data_pipeline"):
    _p = _PROJECT_ROOT / _folder
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import data_loader
import formatting

st.set_page_config(page_title="PropSync", page_icon="⚾", layout="wide")

# ---------------------------------------------------------------------------
# CSS — white/black/grey, clean, no decoration
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Base */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #ffffff !important;
    color: #111111 !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f5f5f5 !important;
    border-right: 1px solid #e0e0e0;
}
[data-testid="stSidebar"] * { color: #111111 !important; }

/* Title */
h1 {
    font-size: 2rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.03em !important;
    color: #000000 !important;
    border-bottom: 3px solid #000000;
    padding-bottom: 0.4rem;
    margin-bottom: 0.2rem !important;
    text-transform: uppercase;
}

/* Subheadings */
h2, h3 {
    font-weight: 700 !important;
    color: #000000 !important;
    letter-spacing: -0.01em !important;
    text-transform: uppercase;
    font-size: 0.9rem !important;
}

/* Caption / small text */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: #888888 !important;
    font-size: 0.75rem !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #000000;
    gap: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #888888 !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.5rem 1.2rem;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #000000 !important;
    border-bottom: 3px solid #000000 !important;
    margin-bottom: -2px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 0;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #888888 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 900 !important;
    color: #000000 !important;
    letter-spacing: -0.02em !important;
}

/* Dataframe / table */
[data-testid="stDataFrame"] {
    border: 1px solid #e0e0e0 !important;
}
iframe {
    border: none !important;
}

/* Buttons */
[data-testid="stButton"] button {
    background: #000000 !important;
    color: #ffffff !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 0.5rem 1.5rem;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #000000 !important;
}

/* Multiselect tags */
[data-baseweb="tag"] {
    background-color: #000000 !important;
    border-radius: 0 !important;
}

/* Selectbox */
[data-baseweb="select"] {
    border-radius: 0 !important;
}

/* Divider */
hr { border-color: #e0e0e0 !important; margin: 1.5rem 0 !important; }

/* Warning/success banners */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    border-left: 3px solid #000000 !important;
    background: #f5f5f5 !important;
    color: #111111 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading — cached 10 min
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _load_stats():
    return data_loader.load_raw_stats()

@st.cache_data(ttl=600)
def _load_picks():
    return data_loader.load_ranked_edges()


def _status_banner():
    if not data_loader.IS_LIVE_DATA:
        st.warning("Sample data — live pipeline not connected.", icon="⚠️")
    else:
        st.success("Live data", icon="✅")


# ---------------------------------------------------------------------------
# Picks tab
# ---------------------------------------------------------------------------

def render_picks():
    candidates = _load_picks()
    full_df = formatting.edge_candidates_to_dataframe(candidates)

    st.subheader("Best Picks — Ranked by Edge")

    with st.sidebar:
        st.markdown("### Filters")
        prop_options = sorted(full_df["market"].unique()) if not full_df.empty else []
        selected_props = st.multiselect(
            "Prop type",
            options=prop_options,
            format_func=formatting.prop_display_name,
            default=[],
        )
        min_edge = st.slider("Min edge", -0.20, 0.30, -0.20, 0.01)
        player_options = sorted(full_df["player"].unique()) if not full_df.empty else []
        selected_players = st.multiselect("Player", options=player_options, default=[])

    filtered = formatting.filter_picks(
        full_df,
        prop_types=selected_props or None,
        min_edge=min_edge if min_edge > -0.20 else None,
        players=selected_players or None,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Legs", len(filtered))
    c2.metric("Avg Edge", f"{filtered['edge'].mean():+.1%}" if len(filtered) else "—")
    c3.metric("Best Edge", f"{filtered['edge'].max():+.1%}" if len(filtered) else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    display_cols = {
        "player": "Player",
        "prop_type": "Prop",
        "line": "Line",
        "side": "Side",
        "chosen_book_price": "Price",
        "model_prob_pct": "Model",
        "market_prob_pct": "Market",
        "edge_pct": "Edge",
        "sportsbook": "Book",
    }
    show_df = filtered[[c for c in display_cols if c in filtered.columns]].rename(columns=display_cols)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Parlay Builder")
    n = st.slider("Legs", 1, min(10, len(candidates) or 1), min(5, len(candidates) or 1))
    top_df = formatting.top_n_parlay_view(candidates, n)
    top_show = top_df[[c for c in display_cols if c in top_df.columns]].rename(columns=display_cols)
    st.dataframe(top_show, use_container_width=True, hide_index=True)

    if not top_df.empty:
        combined = top_df["model_prob"].prod()
        st.caption(f"Combined model probability (independent assumption): {combined:.1%}")


# ---------------------------------------------------------------------------
# Raw stats tab
# ---------------------------------------------------------------------------

def render_stats():
    batter_df, pitcher_df = _load_stats()

    tab_b, tab_p, tab_pf = st.tabs(["Batters", "Pitchers", "Park Factors"])

    with tab_b:
        teams = sorted(batter_df["team"].unique())
        sel_teams = st.multiselect("Team", teams, default=[], key="bt")
        view = batter_df if not sel_teams else batter_df[batter_df["team"].isin(sel_teams)]
        sort_map = {
            "Barrel %": "barrel_pct", "Hard-Hit %": "hard_hit_pct",
            "xSLG": "xslg", "SwStr %": "swinging_strike_pct",
        }
        sort_label = st.selectbox("Sort by", list(sort_map), key="bs")
        view = view.sort_values(sort_map[sort_label], ascending=False)
        st.dataframe(formatting.batter_showcase_table(view), use_container_width=True, hide_index=True)

    with tab_p:
        p_teams = sorted(pitcher_df["team"].unique())
        sel_p = st.multiselect("Team", p_teams, default=[], key="pt")
        p_view = pitcher_df if not sel_p else pitcher_df[pitcher_df["team"].isin(sel_p)]
        p_view = p_view.sort_values("k_per_9", ascending=False)
        st.dataframe(formatting.pitcher_showcase_table(p_view), use_container_width=True, hide_index=True)

    with tab_pf:
        st.caption("Park factors split by hit type — 1B/2B/3B/HR each get a separate number.")
        st.dataframe(formatting.park_factor_table(batter_df), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("PropSync")
    st.caption("MLB player prop picks — ranked by edge against the sportsbook.")
    _status_banner()

    tab_picks, tab_stats = st.tabs(["Best Picks", "Raw Stats"])
    with tab_picks:
        render_picks()
    with tab_stats:
        render_stats()


if __name__ == "__main__":
    main()
