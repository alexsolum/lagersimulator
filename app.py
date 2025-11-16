import streamlit as st
import simpy
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# TIME FORMATTER (SECONDS → HH:MM:SS)
# ------------------------------------------------------------
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
# Picker Process with animation + heatmap logging
# ------------------------------------------------------------
def picker_process(env, picker_id, orders_store, coord_map, picker_speed,
                   picker_pick_time, log, movement_log, heatmap_counter):

    current_location = None
    total_time = 0
    log.append(f"Picker {picker_id} started at time {env.now:.2f}")

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
            # Current and next coordinates
            (x1, y1) = coord_map[current_location]
            (x2, y2) = coord_map[next_loc]

            travel_time = travel_time_between(coord_map, current_location, next_loc, picker_speed)

            # Animation steps
            steps = max(1, int(travel_time * 5))
            for i in range(steps):
                t = env.now + (travel_time * (i / steps))
                x = x1 + (x2 - x1) * (i / steps)
                y = y1 + (y2 - y1) * (i / steps)
                movement_log.append((t, picker_id, x, y))

            yield env.timeout(travel_time)
            total_time += travel_time

            # Pick location
            (px, py) = coord_map[next_loc]
            movement_log.append((env.now, picker_id, px, py))

            heatmap_counter[next_loc] = heatmap_counter.get(next_loc, 0) + 1

            yield env.timeout(picker_pick_time)
            total_time += picker_pick_time

            current_location = next_loc

        log.append(f"Picker {picker_id} completed {order_id} at {env.now:.2f}")

    log.append(f"Picker {picker_id} finished. Total time: {total_time:.2f}")


# ------------------------------------------------------------
# Order Manager
# ------------------------------------------------------------
def order_manager(env, store, orders, num_pickers, log):
    for i, order_list in enumerate(orders):
        order_id = f"O{i+1:03d}"
        store.put({"order_id": order_id, "order_list": order_list})

    yield env.timeout(0)
    log.append("All orders have been added to the queue.")

    for _ in range(num_pickers):
        store.put({"order_id": "STOP", "order_list": None})


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
st.title("📦 Lager-simulering med SimPy, Heatmap og Animert Avspilling")

uploaded_file = st.file_uploader("Last opp Excel-fil (lokasjoner + ordrer)", type=["xlsx"])
num_pickers = st.number_input("Antall plukkere", min_value=1, max_value=20, value=2)
run_button = st.button("🚀 Kjør simulering")


if uploaded_file and run_button:

    # Load Excel sheets
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

    # Build orders
    picking_orders = []
    for ordre, group in df_orders.groupby("ordre"):
        locs = []
        for art in group["artikkel"]:
            if art in location_map:
                locs.append(location_map[art]["loc"])
        if locs:
            picking_orders.append(locs)

    # Simulation
    env = simpy.Environment()
    store = simpy.Store(env)
    log = []
    movement_log = []
    heatmap_counter = {}

    # Start pickers
    pickers = []
    for i in range(num_pickers):
        eff = max(1, np.random.normal(30, 5))
        pick_time = (30 / eff) * 2

        pickers.append({"picker": i+1, "efficiency": eff, "pick_time": pick_time})

        env.process(
            picker_process(
                env,
                str(i+1),
                store,
                coord_map,
                1.0,
                pick_time,
                log,
                movement_log,
                heatmap_counter
            )
        )

    env.process(order_manager(env, store, picking_orders, num_pickers, log))

    st.write("⏳ Kjører simulering …")
    env.run()
    st.success("Simulering fullført!")

    # Show total simulation time
    total_sim_minutes = env.now
    st.subheader(f"⏱ Total simuleringstid: {format_time(total_sim_minutes * 60)}")

    # Picker configuration
    st.subheader("👷 Picker-konfigurasjon")
    st.table(pd.DataFrame(pickers))

    # Log
    st.subheader("📘 Simuleringslogg")
    st.text("\n".join(log))

    # Build movement dataframe
    movement_df = pd.DataFrame(movement_log, columns=["time", "picker", "x", "y"])
    movement_df = movement_df.sort_values(by="time")

    # Locations
    loc_df = pd.DataFrame(
        [{"loc": k, "x": coord_map[k][0], "y": coord_map[k][1]} for k in coord_map]
    )

    # ------------------------------------------------------------
    # 📍 STATIC VISUALIZATION OF PATHS
    # ------------------------------------------------------------
    st.subheader("📍 Plukkernes bevegelse (statiske spor)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(loc_df["x"], loc_df["y"], c="gray", s=40)

    for picker_id in movement_df["picker"].unique():
        p = movement_df[movement_df["picker"] == picker_id]
        ax.plot(p["x"], p["y"], label=f"Picker {picker_id}")

    ax.set_title("Plukkernes bevegelsesspor")
    ax.legend()
    st.pyplot(fig)

    # ------------------------------------------------------------
    # 🔥 HEATMAP
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
    ax2.set_title("Heatmap")
    st.pyplot(fig2)

    # ------------------------------------------------------------
    # 🎥 ANIMATED PLAYBACK (using slider)
    # ------------------------------------------------------------
    st.subheader("🎥 Animert avspilling av plukkere")

    max_time = movement_df["time"].max()

    selected_time = st.slider(
        "Velg tidspunkt i simuleringen",
        0.0,
        float(max_time),
        0.0,
        step=max_time / 200,
        format="%.2f"
    )

    st.write(f"⏱ Tid: **{format_time(selected_time * 60)}**")

    frame = movement_df[movement_df["time"] <= selected_time]

    fig_anim, ax_anim = plt.subplots(figsize=(9, 4))
    ax_anim.scatter(loc_df["x"], loc_df["y"], c="gray", s=40)

    for picker_id in frame["picker"].unique():
        p = frame[frame["picker"] == picker_id]
        ax_anim.plot(p["x"], p["y"], label=f"Picker {picker_id}")
        ax_anim.scatter(p["x"].iloc[-1], p["y"].iloc[-1], s=100)

    ax_anim.set_title(f"Tid: {format_time(selected_time * 60)}")
    ax_anim.legend()
    st.pyplot(fig_anim)

