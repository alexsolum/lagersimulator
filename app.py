import streamlit as st
import simpy
import pandas as pd
import numpy as np
import math
import random
from io import StringIO

# ------------------------------------------------------------
# --- Simulation Parameters ---
# ------------------------------------------------------------
DEFAULT_NUM_PICKERS = 2
MEAN_PICK_EFF = 30
STD_DEV_PICK_EFF = 5
PICK_TIME = 2
PICKER_SPEED = 1.0  # meters per time unit

# ------------------------------------------------------------
# --- Travel Time Logic ---
# ------------------------------------------------------------
def travel_time_between(coord_map, loc_a, loc_b, picker_speed):
    if loc_a not in coord_map or loc_b not in coord_map:
        return 0
    (x1, y1) = coord_map[loc_a]
    (x2, y2) = coord_map[loc_b]
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist / picker_speed

# ------------------------------------------------------------
# --- Picker Process ---
# ------------------------------------------------------------
def picker_process(env, picker_id, orders_store, coord_map, picker_speed, picker_pick_time, log):
    current_location = None
    total_picking_time = 0
    log.append(f"Picker {picker_id} started at {env.now:.2f}")

    while True:
        item = yield orders_store.get()

        if item is None or item["order_list"] is None:
            log.append(f"Picker {picker_id} STOP at {env.now:.2f}")
            break

        order_id = item["order_id"]
        order_locs = item["order_list"]
        log.append(f"Picker {picker_id} starts order {order_id} at {env.now:.2f}")

        if current_location is None:
            current_location = order_locs[0]

        for next_loc in order_locs:
            t = travel_time_between(coord_map, current_location, next_loc, picker_speed)
            yield env.timeout(t)
            total_picking_time += t

            yield env.timeout(picker_pick_time)
            total_picking_time += picker_pick_time

            current_location = next_loc

        log.append(f"Picker {picker_id} completed order {order_id} at {env.now:.2f}")

    log.append(f"Picker {picker_id} FINISHED at {env.now:.2f}. Total time: {total_picking_time:.2f}")

# ------------------------------------------------------------
# --- Order Manager ---
# ------------------------------------------------------------
def order_manager(env, store, orders, num_pickers, log):
    for i, order_list in enumerate(orders):
        oid = f"O{i+1:03d}"
        store.put({"order_id": oid, "order_list": order_list})
    yield env.timeout(0)
    log.append("All orders loaded.")

    for _ in range(num_pickers):
        store.put({"order_id": "STOP", "order_list": None})

# ------------------------------------------------------------
# --- Page Layout ---
# ------------------------------------------------------------
st.title("📦 Lager-simulering med SimPy")
st.write("Last opp Excel med arkene **lokasjoner** og **ordrer** for å kjøre en plukk-simulering.")

uploaded_file = st.file_uploader("Last opp Excel-fil", type=["xlsx"])

num_pickers = st.number_input("Antall plukkere", value=DEFAULT_NUM_PICKERS, min_value=1)

run_button = st.button("🚀 Kjør simulering")

if uploaded_file and run_button:
    df_loc = pd.read_excel(uploaded_file, sheet_name="lokasjoner")
    df_orders = pd.read_excel(uploaded_file, sheet_name="ordrer")

    st.subheader("📍 Lokasjoner")
    st.dataframe(df_loc)

    st.subheader("📝 Ordrer")
    st.dataframe(df_orders)

    # Build maps
    location_map = {}
    coord_map = {}
    for _, row in df_loc.iterrows():
        art = row["artikkel"]
        loc = int(row["lokasjon"])
        x = float(row["x"])
        y = float(row["y"])
        sec = row["seksjon"]

        location_map[art] = {"loc": loc, "x": x, "y": y, "section": sec}
        coord_map[loc] = (x, y)

    # Build picking orders
    picking_orders = []
    for ordre, group in df_orders.groupby("ordre"):
        locs = []
        for art in group["artikkel"]:
            if art in location_map:
                locs.append(location_map[art]["loc"])
        if locs:
            picking_orders.append(locs)

    # -------------------------
    # Run SimPy simulation
    # -------------------------
    log = []
    env = simpy.Environment()
    store = simpy.Store(env)

    # Generate pickers
    pickers = []
    for i in range(num_pickers):
        eff = np.random.normal(MEAN_PICK_EFF, STD_DEV_PICK_EFF)
        eff = max(eff, 1.0)
        pit = (MEAN_PICK_EFF / eff) * PICK_TIME
        pickers.append({"id": str(i+1), "eff": eff, "pick_time": pit})
        env.process(picker_process(env, str(i+1), store, coord_map, PICKER_SPEED, pit, log))

    env.process(order_manager(env, store, picking_orders, num_pickers, log))

    env.run()

    # -------------------------
    # Output
    # -------------------------
    st.subheader("🧮 Resultater")
    st.write(f"**Total simuleringstid:** {env.now:.2f}")

    st.subheader("👷 Picker-konfigurasjon")
    st.table(pd.DataFrame(pickers))

    st.subheader("📜 Simuleringslogg")
    st.text("\n".join(log))
