"""
Park factors, split by hit type — not one blended number.

Per the Scoring Framework Notes: "a park can suppress HRs but still favor
doubles/triples (big gaps, foul territory), so park factor isn't one
number here." FanGraphs' Guts! tool already publishes exactly this shape
of data: separate park factor columns for 1B, 2B, 3B, HR, SO, BB, GB, FB,
LD, IFFB, and FIP, per team, per season — confirmed live at
https://www.fangraphs.com/tools/guts?type=pf while building this module
(checked the page directly; as of today it shows 2025 entries like
Rockies: 1B=108, 2B=111, 3B=135, HR=107 — visibly different numbers per
hit type for the same park, exactly the split this project needs).

pybaseball.park_factors() wraps this exact FanGraphs Guts table.

NOT yet verified: pybaseball's specific column naming/shape for this
function, since the build sandbox has no outbound network access to
actually run `import pybaseball; pybaseball.park_factors()` end-to-end
(see README "Known limitations"). The web fetch above confirms the
underlying FanGraphs data is real, current, and split by hit type — but
the exact DataFrame column names pybaseball returns should be printed and
sanity-checked the first time this runs in a real environment (cozy's
machine), since pybaseball's column naming has shifted across versions in
the past for some leaderboard wrappers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, SEASON  # noqa: E402


def get_park_factors(season: int = SEASON) -> pd.DataFrame:
    """
    Pull FanGraphs' Guts! park factors table for the given season, split
    by hit type (1B, 2B, 3B, HR, SO, BB, GB, FB, LD, IFFB, FIP).

    pybaseball has no park_factors() function, so this scrapes the Guts
    page directly. FanGraphs publishes the most recent completed season
    (or in-season data mid-year) — if `season` isn't found, the table
    will contain whatever season FanGraphs currently shows.
    """
    import requests
    from bs4 import BeautifulSoup

    url = f"https://www.fangraphs.com/guts.aspx?type=pf&startseason={season}&endseason={season}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")

    pf_table = None
    for t in tables:
        rows = t.find_all("tr")
        if rows:
            first_row_text = rows[0].get_text()
            if "Season" in first_row_text and "Team" in first_row_text and "1B" in first_row_text:
                pf_table = t
                break

    if pf_table is None:
        raise ValueError(
            "Could not find park factors table on FanGraphs Guts page. "
            "The page structure may have changed."
        )

    rows = pf_table.find_all("tr")
    headers = [cell.get_text(strip=True) for cell in rows[0].find_all(["th", "td"])]
    data = []
    for row in rows[1:]:
        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
        if cells:
            data.append(cells)

    df = pd.DataFrame(data, columns=headers)
    numeric_cols = [c for c in df.columns if c not in ("Season", "Team")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_hr_park_factor(season: int = SEASON) -> pd.DataFrame:
    """
    Convenience slice: just team + HR park factor, for the HR prop lens.
    """
    df = get_park_factors(season)
    hr_col = _find_column(df, ["HR"])
    team_col = _find_column(df, ["Team", "team_name", "team"])
    return df[[team_col, hr_col]].rename(columns={team_col: "team", hr_col: "hr_park_factor"})


def get_extra_base_park_factors(season: int = SEASON) -> pd.DataFrame:
    """
    Convenience slice for the doubles/total-bases prop lenses: 2B and 3B
    park factors, kept separate from HR per the Scoring Framework Notes
    (doubles props care about gap/foul-territory dimensions, not the HR
    number).
    """
    df = get_park_factors(season)
    team_col = _find_column(df, ["Team", "team_name", "team"])
    col_2b = _find_column(df, ["2B"])
    col_3b = _find_column(df, ["3B"])
    return df[[team_col, col_2b, col_3b]].rename(
        columns={team_col: "team", col_2b: "doubles_park_factor", col_3b: "triples_park_factor"}
    )


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of {candidates} found in park factors columns: {list(df.columns)}. "
        "pybaseball's park_factors() column naming may have changed — inspect "
        "df.columns directly and update the candidates list above."
    )


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
