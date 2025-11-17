# utils.py
import numpy as np
import math

WALK_SPEED = 70   # meter/min
BASE_PICK_SECONDS = 10
BASE_PICK_MIN = BASE_PICK_SECONDS / 60

MEAN_PICK_EFF = 30
STD_PICK_EFF = 5


def generate_picker_profiles(num_pickers, seed=42):
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(num_pickers):
        eff = max(1, rng.normal(MEAN_PICK_EFF, STD_PICK_EFF))
        pick_time = BASE_PICK_MIN * (MEAN_PICK_EFF / eff)
        profiles.append({
            "picker": i + 1,
            "efficiency": eff,
            "pick_time_min": pick_time
        })
    return profiles


def travel_time(coord_map, a, b):
    if a not in coord_map or b not in coord_map:
        return 0
    (x1, y1) = coord_map[a]
    (x2, y2) = coord_map[b]
    dist = math.dist((x1, y1), (x2, y2))
    return dist / WALK_SPEED


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
