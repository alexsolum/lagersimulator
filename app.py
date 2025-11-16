import streamlit as st
import simpy
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

###############################################################
#                 TIDSMODELL (VELDIG VIKTIG!)
#
#  • 1 simuleringstid-enhet = 1 MINUTT
#
#  Dette betyr:
#   - travel_time returnerer MINUTTER
#   - plukktid må også være MINUTTER
#   - format_time() konverterer minutter → sekunder → hh:mm:ss
#
#  GÅFART:
#      WALK_SPEED = 70 meter per minutt  (≈ 4,2 km/t)
#
#  PLUKKTID:
#      BASE_PICK_SECONDS = 10 sekunder per vare
#      BASE_PICK_MIN = 10 / 60 = 0.166 min
#
###############################################################

WALK_SPEED = 70                 # meter per minutt
BASE_PICK_SECONDS = 10          # sekunder per plukk
BASE_PICK_MIN = BASE_PICK_SECONDS / 60  # ≈ 0.166 minutter

MEAN_PICK_EFF = 30              # gjennomsnittlig effektivitet (varer/time)
STD_PICK_EFF = 5                # variasjon i effektivitet


###############################################################
# TIDSFORMAT: SECONDS → HH:MM:SS
###############################################################
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


###############################################################
# REISELOGIKK (MINUTTER)
###############################################################
def travel_time_between(coord_map, loc_a, loc_b):
    if loc_a not in coord_map or loc_b not in coord_map:
        return 0

    (x1, y1) = coord_map[loc_a]
    (x2, y2) = coord_map[loc_b]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # KONVERTERER METER → MINUTTER
    return distance / WALK_SPEED


###############################################################
# PICKER-PROSESS (MED ANIMASJON OG HEATMAP)
###############################################################
def picker_process(env, picker_id, orders_store, coord_map,
                   picker_pick_time_min, log,
                   movement_log, heatmap_counter):

    current_location = None
    total_time = 0

    log.append(f"Picker {picker_id} started at {env.now:.2f} min")

    while True:
        item = yield orders_store.get()

        if item is None or item["order_list"] is None:
            log.append(f"Picker {picker_id} STOP at {env.now:.2f}")
            break

        order_id = item["order_id"]
        locs = item["order_list"]
        log.append(f"Picker {picker_id} starts {order_id} at {env.now:.2f}")

        if current_location is None:
            current_location = locs[0]

        for next_loc in locs:

            # REISETID I MINUTTER
            t_travel = travel_time_between(coord_map, current_location, next_loc)

            # ANIMASJON (smooth movement)
            (x1, y1) = coord_map[current_location]
            (x2, y2) = coord_map[next_loc]
            steps = max(1, int(t_travel * 5))

            for i in range(steps):
                t = env.now + t_travel * (i / steps)
                x = x1 + (x2 - x1) * (i / steps)
                y = y1 + (y2 - y1) * (i / steps)
                movement_log.append((t, picker_id, x, y))

            yield env.timeout(t_travel)
            total_time += t_travel

            # PLOTT POSISJON FOR PICK
            px, py = coord_map[next_loc]
            movement_log.append((env.now, picker_id, px, py))

            # OPPDATER HEATMAP
            heatmap_counter[next_loc] = heatmap_counter.get(next_loc, 0) + 1

            # PLUKKTID (I MINUTTER)
            yield env.timeout(picker_pick_time_min)
            total_time += picker_pick_time_min

            current_location = next_loc

        log.append(f"Picker {picker_id} done with {order_id} at {env.now:.2f}")

    log.append(f"Picker {picker_id} finished after {total_time:.2f} min")


###############################################################
# ORDREMANAGER
###############################################################
def order_manager(env, store, orders, num_pickers, log):
    for i, order_list in enumerate(orders):
        store.put({"order_id": f"O{i+1:03d}", "order_list": order_list})

    yield env.timeout(0)
    log.append("All orders added.")

    for _ in range(num_pickers):
        store.put({"order_id": "STOP", "order_list": None})


###############################################################
# STREAMLIT APP
###############################################################
st.title("📦 Lager-simulering med korrekt tidsmodell + animasjon + heatmap")

file = st.file_uploader("Last opp Excel-fil", type=["xlsx"])
num_pickers = st.number_input("Antall plukkere", 1, 20, 3)
run = st.button("🚀 Kjør simulering")

if file and run:

    df_loc = pd.read_excel(file, sheet_name="lokasjoner")
    df_orders = pd.read_excel(file, sheet_name="ordrer")

    st.subheader("📍 Lokasjoner")
    st.dataframe(df_loc)

    st.subheader("🧾 Ordrer")
    st.dataframe(df_orders)

    # ARTIKKEL → LOKASJON
    coord_map = {}
    art_to_loc = {}
    for _, r in df_loc.iterrows():
        loc = int(r["lokasjon"])
        coord_map[loc] = (float(r["x"]), float(r["y"]))
        art_to_loc[r["artikkel"]] = loc

    # ORDRE → LOKASJONER
    picking_orders = []
    for oid, group in df_orders.groupby("ordre"):
        locs = [art_to_loc[a] for a in group["artikkel"] if a in art_to_loc]
        picking_orders.append(locs)

    # SIMULERING
    env = simpy.Environment()
    store = simpy.Store(env)
    log = []
    movement_log = []
    heatmap_counter = {}

    pickers = []
    for i in range(num_pickers):
        eff = max(1, np.random.normal(MEAN_PICK_EFF, STD_PICK_EFF))

        pick_time_min = BASE_PICK_MIN * (MEAN_PICK_EFF / eff)

        pickers.append({
            "picker": i+1,
            "efficiency": eff,
            "pick_time_min": pick_time_min
        })

        env.process(
            picker_process(
                env, str(i+1), store, coord_map,
                pick_time_min, log,
                movement_log, heatmap_counter
            )
        )

    env.process(order_manager(env, store, picking_orders, num_pickers, log))

    st.write("⏳ Kjører simulering...")
    env.run()
    st.success("Simulering fullført!")

    # TOTALTID I MINUTTER
    total_minutes = env.now
    st.subheader(f"⏱ Total tid: {format_time(total_minutes * 60)}")

    st.subheader("👷 Plukkere")
    st.table(pd.DataFrame(pickers))

    st.subheader("📘 Logg")
    st.text("\n".join(log))

    # BEVEGELSESDATA
    movement_df = pd.DataFrame(movement_log, columns=["time", "picker", "x", "y"])
    movement_df = movement_df.sort_values(by="time")

    loc_df = pd.DataFrame(
        [{"loc": k, "x": coord_map[k][0], "y": coord_map[k][1]} for k in coord_map]
    )

    ###############################################################
    # VISUALISERING – SPOR
    ###############################################################
    st.subheader("📍 Plukkernes spor")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(loc_df["x"], loc_df["y"], c="gray", s=50)

    for pid in movement_df["picker"].unique():
        p = movement_df[movement_df["picker"] == pid]
        ax.plot(p["x"], p["y"], label=f"Picker {pid}")

    ax.legend()
    st.pyplot(fig)

    ###############################################################
    # HEATMAP
    ###############################################################
    st.subheader("🔥 Heatmap for besøk")

    heat_df = pd.DataFrame([
        {"loc": loc, "visits": heatmap_counter.get(loc, 0),
         "x": coord_map[loc][0], "y": coord_map[loc][1]}
        for loc in coord_map
    ])

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sc = ax2.scatter(heat_df["x"], heat_df["y"],
                     c=heat_df["visits"], cmap="hot", s=200)
    plt.colorbar(sc, ax=ax2)
    st.pyplot(fig2)

    ###############################################################
    # ANIMASJON
    ###############################################################
    st.subheader("🎥 Animasjon")

    max_t = movement_df["time"].max()
    t_sel = st.slider("Tidspunkt (minutter)", 0.0, float(max_t),
                      0.0, step=max_t/200)

    st.write(f"⏱ Tid: **{format_time(t_sel * 60)}**")

    frame = movement_df[movement_df["time"] <= t_sel]

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.scatter(loc_df["x"], loc_df["y"], c="gray", s=50)

    for pid in frame["picker"].unique():
        p = frame[frame["picker"] == pid]
        ax3.plot(p["x"], p["y"], label=f"Picker {pid}")
        ax3.scatter(p["x"].iloc[-1], p["y"].iloc[-1], s=120)

    ax3.legend()
    st.pyplot(fig3)
