"""Greedy lokasjonsoptimalisering for lagerlayout.

Denne modulen driver Streamlit-siden "🧭 Lokasjonsoptimalisering" og
lager et enkelt forslag til hvor artikler bør plasseres basert på
historisk etterspørsel og nærhet til inngangen.
"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def _sorted_locations(df_loc: pd.DataFrame, entry_x: float, entry_y: float) -> pd.DataFrame:
    """Returner lokasjoner sortert etter avstand til inngangspunktet."""

    locations = df_loc[["lokasjon", "x", "y"]].drop_duplicates().copy()
    locations["distance"] = (
        (locations["x"] - float(entry_x)) ** 2 + (locations["y"] - float(entry_y)) ** 2
    ).apply(math.sqrt)
    return locations.sort_values("distance").reset_index(drop=True)


def _target_slot_counts(demand: pd.Series, n_slots: int) -> pd.Series:
    """Fordel hvor mange lokasjoner hver artikkel bør få."""

    if demand.empty or n_slots <= 0:
        return pd.Series(dtype=int)

    demand = demand[demand > 0].sort_values(ascending=False)
    if demand.empty:
        return pd.Series(dtype=int)

    # Start med proporsjonal fordeling
    proportional = (demand / demand.sum() * n_slots).round().astype(int).clip(lower=1)

    # Juster ned hvis vi har over-allokert
    while proportional.sum() > n_slots:
        idx = proportional[proportional > 1].idxmin()
        proportional.loc[idx] -= 1

    # Juster opp hvis vi mangler slots
    while proportional.sum() < n_slots:
        idx = proportional.idxmax()
        proportional.loc[idx] += 1

    return proportional


def greedy_assignment(
    df_loc: pd.DataFrame, demand: pd.Series, entry_x: float = 0.0, entry_y: float = 0.0
) -> pd.DataFrame:
    """Returner DataFrame med tildeling av artikler til lokasjoner."""

    locations = _sorted_locations(df_loc, entry_x, entry_y)
    n_slots = len(locations)

    slot_counts = _target_slot_counts(demand, n_slots)
    if slot_counts.empty:
        # Fallback: behold eksisterende artikkelverdier dersom etterspørsel ikke er tilgjengelig
        loc_map = df_loc.set_index("lokasjon")
        locations["artikkel"] = locations["lokasjon"].map(loc_map["artikkel"])
        locations["antall"] = loc_map["antall"] if "antall" in loc_map else 1
        return locations[["lokasjon", "x", "y", "artikkel", "antall"]]

    # Lag liste med artikler i prioriteringsrekkefølge
    article_order: list[str] = []
    for art, count in slot_counts.items():
        article_order.extend([str(art)] * int(count))

    # Klipp eller fyll for å matche antall lokasjoner
    article_order = article_order[:n_slots]
    if len(article_order) < n_slots:
        article_order.extend([slot_counts.idxmax()] * (n_slots - len(article_order)))

    locations["artikkel"] = article_order
    locations["antall"] = 1

    return locations[["lokasjon", "x", "y", "artikkel", "antall"]]


def assignment_to_layout(assign_df: pd.DataFrame) -> pd.DataFrame:
    """Konverter tildelingstabell til layout-format brukt av simuleringen."""

    layout = assign_df.copy()
    if "antall" not in layout.columns:
        layout["antall"] = 1

    cols: Iterable[str] = ["lokasjon", "x", "y", "artikkel", "antall"]
    return layout.loc[:, cols]
