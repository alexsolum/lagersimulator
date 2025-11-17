# sim/picker.py
import simpy
import math

from .utils import travel_time


# ---------------------------------------------------------
# PICKER PROCESS
# ---------------------------------------------------------
def picker_process(
    env,
    pid,
    store,
    coord_map,
    pick_time,
    log,
    mv_log,
    heatmap,
    stats,
    article_locations,
    location_resources,
    location_stock
):
    """
    En enkel men realistisk modell:
    - Plukkere henter ordre fra en Store
    - Venter i kø hvis flere vil inn i samme lokasjon
    - Registrerer travel, pick, queue, movement-flight
    """
    current = None
    log.append(f"P{pid} starter {env.now:.2f}")

    while True:
        request_time = env.now
        item = yield store.get()
        wait = env.now - request_time

        stats[pid]["wait_minutes"] += wait

        # LOG QUEUE EVENT
        cur_x, cur_y = coord_map.get(current, (0.0, 0.0))
        mv_log.append((request_time, pid, cur_x, cur_y, "queue"))
        if wait > 0:
            mv_log.append((env.now, pid, cur_x, cur_y, "queue"))

        if item is None or item["order_list"] is None:
            log.append(f"P{pid} STOP {env.now:.2f}")
            break

        order_id = item["order_id"]
        articles = item["order_list"]

        log.append(f"P{pid} starter ordre {order_id} ved {env.now:.2f}")

        for art in articles:

            # 1) Finn lokasjon(er) som har denne artikkelen
            candidates = [
                loc for loc in article_locations.get(art, [])
                if location_stock.get(loc, 0) > 0
            ]
            if not candidates:
                log.append(f"P{pid}: ingen beholdning for {art}")
                continue

            # 2) Velg ledig lokasjon først
            chosen = None
            req = None
            for loc in candidates:
                res = location_resources[loc]
                if res.count < res.capacity and len(res.queue) == 0:
                    chosen = loc
                    req = res.request()
                    break

            # Hvis ingen var ledig: ta første og kø
            if chosen is None:
                chosen = candidates[0]
                req = location_resources[chosen].request()

            queue_start = env.now
            yield req

            # log waiting at shelf
            if env.now > queue_start:
                mv_log.append((queue_start, pid, cur_x, cur_y, "queue"))
                mv_log.append((env.now, pid, cur_x, cur_y, "queue"))

            # 3) Travel time
            if current is None:
                current = chosen
            tx = travel_time(coord_map, current, chosen)

            (x1, y1) = coord_map[current]
            (x2, y2) = coord_map[chosen]

            # interpolated movement (5 points per minute)
            steps = max(1, int(tx * 5))
            for i in range(steps):
                frac = i / steps
                mv_log.append((
                    env.now + tx * frac,
                    pid,
                    x1 + (x2 - x1) * frac,
                    y1 + (y2 - y1) * frac,
                    "move"
                ))

            yield env.timeout(tx)
            mv_log.append((env.now, pid, x2, y2, "move"))

            stats[pid]["distance_m"] += math.dist((x1, y1), (x2, y2))
            heatmap[chosen] = heatmap.get(chosen, 0) + 1

            # 4) Pick time
            yield env.timeout(pick_time)
            mv_log.append((env.now, pid, x2, y2, "pick"))

            location_stock[chosen] -= 1
            location_resources[chosen].release(req)

            log.append(f"P{pid} plukket {art} fra {chosen} ved {env.now:.2f}")

            current = chosen

        log.append(f"P{pid} ferdig ordre {order_id} ved {env.now:.2f}")
