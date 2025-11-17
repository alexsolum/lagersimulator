# sim/simulation.py
import simpy
import pandas as pd

from .picker import picker_process
from .orders import order_manager_process
from .utils import format_time


# ---------------------------------------------------------
# BUILD STRUCTURES
# ---------------------------------------------------------
def build_environment(df_loc, df_orders, picker_profiles, section_length=None):
    """
    Tar layout + ordre + plukkerprofiler og bygger:
      - coord_map
      - article_locations
      - location_stock
      - location_resources (en SimPy Resource per lokasjon)
      - orders (liste av liste med artikler)
    """

    env = simpy.Environment()

    coord_map = {}
    article_locations = {}
    location_stock = {}
    location_resources = {}

    # --- Layout ---
    for _, r in df_loc.iterrows():
        loc = int(r["lokasjon"])
        x = float(r["x"])
        y = float(r["y"])

        coord_map[loc] = (x, y)
        article_locations.setdefault(r["artikkel"], []).append(loc)

        # antall = antall plukkbare enheter i denne lokasjonen
        count = int(r["antall"]) if "antall" in df_loc.columns else 1
        location_stock[loc] = location_stock.get(loc, 0) + count

        # en plukker om gangen ved hylla
        location_resources[loc] = simpy.Resource(env, capacity=1)

    # sorterer lokasjoner per artikkel (viktig)
    for lst in article_locations.values():
        lst.sort()

    # --- Ordre ---
    orders = []
    for oid, group in df_orders.groupby("ordre"):
        arts = [a for a in group["artikkel"] if a in article_locations]
        if arts:
            orders.append(arts)

    return {
        "env": env,
        "coord_map": coord_map,
        "article_locations": article_locations,
        "location_stock": location_stock,
        "location_resources": location_resources,
        "orders": orders,
        "picker_profiles": picker_profiles,
    }


# ---------------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------------
def run_simulation(df_loc, df_orders, picker_profiles):
    """
    Kjør hele simuleringen og returnér strukturer som brukes i visualisering.
    """

    data = build_environment(df_loc, df_orders, picker_profiles)

    env = data["env"]
    coord_map = data["coord_map"]
    article_locations = data["article_locations"]
    location_stock = data["location_stock"]
    location_resources = data["location_resources"]
    orders = data["orders"]

    # simpy store hvor pickere henter ordre
    store = simpy.Store(env)

    # logger
    log = []
    mv_log = []
    heatmap = {}
    stats = {}

    # initialiser stats for alle pickere
    pickers = []
    for profile in picker_profiles:
        pid = str(profile["picker"])
        stats[pid] = {"distance_m": 0.0, "wait_minutes": 0.0}
        pickers.append(profile.copy())

        env.process(
            picker_process(
                env,
                pid,
                store,
                coord_map,
                profile["pick_time_min"],
                log,
                mv_log,
                heatmap,
                stats,
                article_locations,
                location_resources,
                location_stock,
            )
        )

    # Start order manager
    env.process(order_manager_process(env, store, orders, len(picker_profiles)))

    # Kjør simulering
    env.run()

    # Konverter til DataFrames
    mv_df = pd.DataFrame(mv_log, columns=["time", "picker", "x", "y", "event"])
    mv_df = mv_df.sort_values("time")

    picker_df = pd.DataFrame(pickers)
    picker_df["distance_m"] = picker_df["picker"].apply(lambda p: stats[str(p)]["distance_m"])
    picker_df["queue_time_min"] = picker_df["picker"].apply(lambda p: stats[str(p)]["wait_minutes"])

    return {
        "total_minutes": env.now,
        "total_time_str": format_time(env.now * 60),
        "movement_df": mv_df,
        "pickers": picker_df,
        "coord_map": coord_map,
        "heatmap": heatmap,
        "total_distance_m": sum(v["distance_m"] for v in stats.values()),
        "total_wait_minutes": sum(v["wait_minutes"] for v in stats.values()),
        "layout_df": df_loc.copy(),
        "log": log,
    }
