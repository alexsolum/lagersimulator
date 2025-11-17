# simulation.py
import simpy
import pandas as pd
from utils import travel_time

def picker(env, pid, store, coord_map, pick_time, log, mv_log, heatmap,
           stats, article_locations, location_resources, location_stock):

    current = None
    stats[pid] = {"distance_m": 0, "wait_min": 0}

    while True:
        request_time = env.now
        item = yield store.get()
        wait = env.now - request_time
        stats[pid]["wait_min"] += wait

        if item["order_list"] is None:
            break

        for artic in item["order_list"]:
            candidates = article_locations.get(artic, [])
            candidates = [c for c in candidates if location_stock[c] > 0]
            if not candidates:
                continue

            chosen = None
            req = None

            # prøv "ledig og ingen kø"
            for loc in candidates:
                res = location_resources[loc]
                if res.count < res.capacity and len(res.queue) == 0:
                    chosen = loc
                    req = res.request()
                    break

            # ellers ta første
            if chosen is None:
                chosen = candidates[0]
                req = location_resources[chosen].request()

            yield req

            # reisetid
            if current is None:
                current = chosen

            t_travel = travel_time(coord_map, current, chosen)
            (x1, y1) = coord_map[current]
            (x2, y2) = coord_map[chosen]

            stats[pid]["distance_m"] += ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            mv_log.append((env.now, pid, x1, y1, "move"))

            yield env.timeout(t_travel)
            mv_log.append((env.now, pid, x2, y2, "move"))

            # plukk
            yield env.timeout(pick_time)
            mv_log.append((env.now, pid, x2, y2, "pick"))

            heatmap[chosen] = heatmap.get(chosen, 0) + 1
            location_stock[chosen] -= 1
            location_resources[chosen].release(req)

            current = chosen


def order_manager(env, store, orders, num_pickers):
    for oid, items in enumerate(orders):
        store.put({"order_id": oid, "order_list": items})
    for _ in range(num_pickers):
        store.put({"order_id": None, "order_list": None})


def run_simulation(df_loc, df_orders, picker_profiles):
    env = simpy.Environment()

    coord_map = {}
    article_locations = {}
    location_stock = {}
    location_resources = {}

    for _, r in df_loc.iterrows():
        loc = int(r["lokasjon"])
        coord_map[loc] = (float(r["x"]), float(r["y"]))
        article_locations.setdefault(r["artikkel"], []).append(loc)
        location_stock[loc] = int(r.get("antall", 1))
        location_resources[loc] = simpy.Resource(env, capacity=1)

    for alist in article_locations.values():
        alist.sort()

    orders = []
    for oid, g in df_orders.groupby("ordre"):
        orders.append(list(g["artikkel"]))

    store = simpy.Store(env)
    log = []
    mv_log = []
    heatmap = {}
    stats = {}

    for p in picker_profiles:
        pid = p["picker"]
        env.process(
            picker(
                env, pid, store, coord_map,
                p["pick_time_min"], log, mv_log, heatmap,
                stats, article_locations, location_resources, location_stock
            )
        )

    env.process(order_manager(env, store, orders, len(picker_profiles)))
    env.run()

    mv_df = pd.DataFrame(mv_log, columns=["time", "picker", "x", "y", "event"])
    mv_df.sort_values("time", inplace=True)

    return {
        "movement_df": mv_df,
        "heatmap": heatmap,
        "coord_map": coord_map,
        "stats": stats,
        "total_minutes": env.now,
        "layout_df": df_loc.copy()
    }
