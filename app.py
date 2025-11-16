import streamlit as st
import simpy
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Simulation Settings
# ------------------------------------------------------------
MEAN_PICK_EFF = 30
STD_PICK_EFF = 5
PICK_TIME = 2
PICKER_SPEED = 1.0


# ------------------------------------------------------------
# Helper: travel time using XY coordinates
# ------------------------------------------------------------
def travel_time_between(coord_map, loc_a, loc_b, picker_speed):
    if loc_a not in coord_map or loc_b not in coord_map:
        return 0
    (x1, y1) = coord_map[loc_a]
    (x2, y2) = coord_map[loc_b]
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist / picker_speed


# ------------------------------------------------------------
# Picker Process (logs movement + heatmap)
# ------------------------------------------------------------
def picker_process(env, picker_id, orders_store, coord_map, picker_speed,
                   picker_pick_time, log, movement_log, heatmap_counter):
    current_location = None
    total_time = 0

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
            (x1, y1) = coord_map[current_location]
            (x2, y2) = coord_map[next_loc]

            # Movement tracking over path
            travel_time = travel_time_between(coord_map, current_location, next_loc, picker_speed)
            steps = max(1, int(travel_time * 5))

            for i in range(steps):
                t = env.now + (travel_time * (i / steps))
                x = x1 + (x2 - x1) * (i / steps)
                y = y1 + (y2 - y1) * (i / steps)
                movement_log.append((t, picker_id, x, y))

            yield env.timeout(travel_time)
            total_time += travel_time

            # At picking moment
            (px, py) = coord_map[next_loc]
            movement_log.append((env.now, picker_id, px, py))

            heatmap_counter[next_loc] = heatmap_counter.get(next_loc, 0) + 1

            yield env.timeout(picker_pick_time)
            total_time += picker_pick_time

            current_location = next_loc

        log.append(f"Picker {picker_id} completed order {order_id} at {env.now:.2f}")

    log.append(f"Picker {picker_id} FINISHED at {env.now:.2f}. Total: {total_time:.2f}")


# ------------------------------------------------------------
# Order Manager
# ------------------------------------------------------------
def order_manager(env, store, orders, num_pickers, log):
    # Load all orders
    for i, order_list in enumerate(orders):
        order_id = f"O{i+1:03d}"
        store.put({"order_id": order_id, "order_list": order_list})

    yield env.timeout(0)
    log.append("All orders loaded into queue.")

    # Add STOP signals
    for _ in range(num_pickers):
        store.put({"order_id": "STOP", "order_list": None})


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("📦 Lager-simulering med SimPy")
st.write("Laster Excel med arkene **lokasjoner** og **ordrer**.")

uploaded_file = st.file_uploader("Last opp Excel-fil", type=["xlsx"])
num_pickers = st.number_input("Antall plukkere", min_value=1, max_value=20, value=2)
run_button = st.button("🚀 Kjør simulering")


if uploaded_file and run_button:

    # -------------------------
    # Load layout + orders
    # -------------------------
    df_loc = pd.read_excel(uploaded_file, sheet_name="lokasjoner")
    df_orders = pd.read_excel(uploaded_file, sheet_name="ordrer")

    st.subheader("📍 Lokasjoner")
    st.dataframe(df_loc)

    st.subheader("📝 Ordrer")
    st.dataframe(df_orders)

    # Maps
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

    # Create picking orders
    picking_orders = []
    for ordre, group in df_orders.groupby("ordre"):
        locs = []
        for art in group["artikkel"]:
            if art in location_map:
                locs.append(location_map[art]["loc"])
        if locs:
            picking_orders.append(locs)

    # -------------------------
    # Start Simulation
    # -------------------------
    log = []
    movement_log = []
    heatmap_counter = {}

    env = simpy.Environment()
    store = simpy.Store(env)

    # Start picker processes
    pickers = []
    for i in range(num_pickers):
        eff = max(1, np.random.normal(MEAN_PICK_EFF, STD_PICK_EFF))
        pick_time = (MEAN_PICK_EFF / eff) * PICK_TIME

        pickers.append({"picker": i+1, "efficiency": eff, "pick_time": pick_time})

        env.process(
            picker_process(
                env, str(i+1), store,
                coord_map,
                PICKER_SPEED,
                pick_time,
                log,
                movement_log,
                heatmap_counter
            )
        )

    env.process(order_manager(env, store, picking_orders, num_pickers, log))

    st.write("⏳ Kjører simulering...")
    env.run()
    st.success("Simulering fullført!")

    # ------------------------------------------------------------
    # Output Results
    # ------------------------------------------------------------
    st.subheader("📘 Plukker-konfigurasjon")
    st.table(pd.DataFrame(pickers))

    st.subheader("🧮 Logg")
    st.text("\n".join(log))

    # ------------------------------------------------------------
    # Visualisering av bevegelser
    # ------------------------------------------------------------
    st.subheader("📍 Visualisering av plukkernes bevegelse")

    movement_df = pd.DataFrame(movement_log, columns=["time", "picker", "x", "y"])
    movement_df = movement_df.sort_values(by="time")

    fig, ax = plt.subplots(figsize=(9, 4))

    # Lokasjoner
    loc_df = pd.DataFrame(
        [{"loc": k, "x": coord_map[k][0], "y": coord_map[k][1]} for k in coord_map]
    )
    ax.scatter(loc_df["x"], loc_df["y"], c="gray", s=40, label="Lokasjoner")

    # Paths
    for picker_id in movement_df["picker"].unique():
        p = movement_df[movement_df["picker"] == picker_id]
        ax.plot(p["x"], p["y"], label=f"Picker {picker_id}")

    ax.set_title("Plukkernes bevegelse")
    ax.legend()
    st.pyplot(fig)

    # ------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------
    st.subheader("🔥 Heatmap – mest besøkte lokasjoner")

    heat_df = pd.DataFrame([
        {"loc": loc,
         "visits": heatmap_counter.get(loc, 0),
         "x": coord_map[loc][0],
         "y": coord_map[loc][1]}
        for loc in coord_map
    ])

    fig2, ax2 = plt.subplots(figsize=(9, 4))

    scatter = ax2.scatter(
        heat_df["x"], heat_df["y"],
        c=heat_df["visits"],
        cmap="hot",
        s=200
    )

    plt.colorbar(scatter, ax=ax2, label="Antall besøk")
    ax2.set_title("Heatmap for bevegelse")
    st.pyplot(fig2)
