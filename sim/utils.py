# sim/utils.py
import math
import numpy as np
import pandas as pd

WALK_SPEED = 70               # meter per minutt
BASE_PICK_SECONDS = 10
BASE_PICK_MIN = BASE_PICK_SECONDS / 60
MEAN_PICK_EFF = 30
STD_PICK_EFF = 5


# ---------------------------------------------------------
# FORMATTERING
# ---------------------------------------------------------
def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------
# PICKER PROFILES
# ---------------------------------------------------------
def generate_picker_profiles(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(n):
        eff = max(1, rng.normal(MEAN_PICK_EFF, STD_PICK_EFF))
        pick_time = BASE_PICK_MIN * (MEAN_PICK_EFF / eff)
        profiles.append({
            "picker": i + 1,
            "efficiency": eff,
            "pick_time_min": pick_time
        })
    return profiles


# ---------------------------------------------------------
# DISTANCE & TRAVEL TIME
# ---------------------------------------------------------
def travel_time(coord_map, a, b):
    """Returnerer reisetid i MINUTTER."""
    if a not in coord_map or b not in coord_map:
        return 0
    (x1, y1) = coord_map[a]
    (x2, y2) = coord_map[b]
    dist = math.dist((x1, y1), (x2, y2))
    return dist / WALK_SPEED


# ---------------------------------------------------------
# DEMO-LAYOUTS
# ---------------------------------------------------------
def generate_demo_layout(seed=0):
    """Standard test-layout med 4 ganger × 8 posisjoner."""
    rng = np.random.default_rng(seed)
    rows = []
    loc_id = 1
    for aisle in range(4):
        x = aisle * 3.0
        for slot in range(8):
            y = slot * 2.4
            rows.append({
                "lokasjon": loc_id,
                "x": x,
                "y": y,
                "artikkel": f"A{rng.integers(1, 10)}",
                "antall": rng.integers(1, 3)
            })
            loc_id += 1
    return pd.DataFrame(rows)


def generate_demo_orders(seed, n_orders):
    rng = np.random.default_rng(seed)
    rows = []
    for oid in range(1, n_orders + 1):
        size = rng.integers(2, 5)
        for _ in range(size):
            rows.append({
                "ordre": f"ORD-{oid:03d}",
                "artikkel": f"A{rng.integers(1, 10)}"
            })
    return pd.DataFrame(rows)
