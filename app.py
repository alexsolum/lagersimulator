import streamlit as st
import simpy
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px

###############################################################
#               TIDSMODELL – VIKTIG!
#
#  • 1 simuleringstid-enhet = 1 MINUTT
#
#  Dette betyr:
#     - travel_time returnerer minutter
#     - plukktid angis i minutter
#     - format_time() konverterer til hh:mm:ss
#
#  GÅFART:
#     WALK_SPEED = 70 meter per minutt
#
#  PLUKKTID:
#     BASE_PICK_SECONDS = 10 sekunder
#     BASE_PICK_MIN = 0.166 min
###############################################################

WALK_SPEED = 70
BASE_PICK_SECONDS = 10
BASE_PICK_MIN = BASE_PICK_SECONDS / 60

MEAN_PICK_EFF = 30
STD_PICK_EFF = 5


###############################################################
# PICKERPROFILER
###############################################################
def generate_picker_profiles(num_pickers, seed=42):
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(num_pickers):
        eff = max(1, rng.normal(MEAN_PICK_EFF, STD_PICK_EFF))
        pick_time_min = BASE_PICK_MIN * (MEAN_PICK_EFF / eff)

        profiles.append({
            "picker": i + 1,
            "efficiency": eff,
            "pick_time_min": pick_time_min
        })

    return profiles


###############################################################
# TIDSFORMAT: SECONDS → HH:MM:SS
###############################################################
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


###############################################################
# DISTANSE OG REISETID
###############################################################
def travel_time(coord_map, a, b):
    if a not in coord_map or b not in coord_map:
        return 0
    (x1, y1) = coord_map[a]
    (x2, y2) = coord_map[b]
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist / WALK_SPEED   # minutter


###############################################################
# PICKER-PROSESS
###############################################################
def picker(env, pid, store, coord_map, pick_time, log, mv_log, heatmap,
           stats, article_locations, location_resources, location_stock):
    current = None
    total = 0
    log.append(f"Picker {pid} startet {env.now:.2f} min")

    while True:
        request_time = env.now
        item = yield store.get()
        wait_time = env.now - request_time
        stats[pid]["wait_minutes"] += wait_time

        if item is None or item["order_list"] is None:
            log.append(f"Picker {pid} STOP {env.now:.2f}")
            break

        order_id = item["order_id"]
        order_articles = item["order_list"]

        log.append(f"Picker {pid} starter {order_id} ved {env.now:.2f}")

        for article in order_articles:
            # Finn første ledige lokasjon med beholdning
            candidates = [loc for loc in article_locations.get(article, [])
                          if location_stock.get(loc, 0) > 0]
            if not candidates:
                log.append(f"Picker {pid} fant ingen lagerbeholdning for {article}")
                continue

            chosen = None
            req = None

            for loc in candidates:
                res = location_resources[loc]
                if res.count < res.capacity and len(res.queue) == 0:
                    chosen = loc
                    req = res.request()
                    break

            if chosen is None:
                chosen = candidates[0]
                req = location_resources[chosen].request()

            yield req

            if location_stock.get(chosen, 0) <= 0:
                location_resources[chosen].release(req)
                log.append(f"Picker {pid} ventet på {article}, men lokasjon {chosen} var tom")
                continue

            if current is None:
                current = chosen

            t_travel = travel_time(coord_map, current, chosen)
            (x1, y1) = coord_map[current]
            (x2, y2) = coord_map[chosen]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            stats[pid]["distance_m"] += dist

            steps = max(1, int(t_travel * 5))
            for i in range(steps):
                t = env.now + t_travel * (i / steps)
                x = x1 + (x2 - x1) * (i / steps)
                y = y1 + (y2 - y1) * (i / steps)
                mv_log.append((t, pid, x, y, "move"))

            yield env.timeout(t_travel)
            total += t_travel

            px, py = coord_map[chosen]
            mv_log.append((env.now, pid, px, py, "move"))

            heatmap[chosen] = heatmap.get(chosen, 0) + 1

            yield env.timeout(pick_time)
            total += pick_time

            mv_log.append((env.now, pid, px, py, "pick"))

            location_stock[chosen] -= 1
            location_resources[chosen].release(req)

            log.append(
                f"Picker {pid} plukket {article} fra {chosen} ved {env.now:.2f}"
            )

            current = chosen

        log.append(f"Picker {pid} ferdig {order_id} ved {env.now:.2f}")

    log.append(f"Picker {pid} totaltid {total:.2f} min")


###############################################################
# ORDREMANAGER
###############################################################
def order_manager(env, store, orders, num_pickers):
    for i, lst in enumerate(orders):
        store.put({"order_id": f"O{i+1:03d}", "order_list": lst})
    yield env.timeout(0)
    for _ in range(num_pickers):
        store.put({"order_id": "STOP", "order_list": None})


###############################################################
# KJØR EN HEL SIMULERING
###############################################################
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
        location_stock[loc] = location_stock.get(loc, 0) + (
            int(r["antall"]) if "antall" in df_loc.columns else 1
        )
        location_resources[loc] = simpy.Resource(env, capacity=1)

    for locations in article_locations.values():
        locations.sort()

    orders = []
    for oid, g in df_orders.groupby("ordre"):
        lst = [a for a in g["artikkel"] if a in article_locations]
        if lst:
            orders.append(lst)

    store = simpy.Store(env)

    log = []
    mv_log = []
    heatmap = {}
    stats = {}

    pickers = []
    for profile in picker_profiles:
        pid = str(profile["picker"])
        stats[pid] = {"distance_m": 0.0, "wait_minutes": 0.0}

        pickers.append(profile.copy())

        env.process(
            picker(env, pid, store, coord_map,
                   profile["pick_time_min"], log, mv_log, heatmap, stats,
                   article_locations, location_resources, location_stock)
        )

    env.process(order_manager(env, store, orders, len(picker_profiles)))
    env.run()

    total_minutes = env.now
    total_distance_m = sum(s["distance_m"] for s in stats.values())
    total_wait_minutes = sum(s["wait_minutes"] for s in stats.values())

    mv_df = pd.DataFrame(mv_log, columns=["time", "picker", "x", "y", "event"])
    mv_df = mv_df.sort_values("time")

    picker_df = pd.DataFrame(pickers)
    picker_df["distance_m"] = picker_df["picker"].apply(lambda pid: stats[str(pid)]["distance_m"])
    picker_df["queue_time_min"] = picker_df["picker"].apply(lambda pid: stats[str(pid)]["wait_minutes"])

    return {
        "total_minutes": total_minutes,
        "total_time_str": format_time(total_minutes * 60),
        "movement_df": mv_df,
        "heatmap": heatmap,
        "pickers": picker_df,
        "coord_map": coord_map,
        "total_distance_m": total_distance_m,
        "total_wait_minutes": total_wait_minutes,
        "layout_df": df_loc.copy()
    }


###############################################################
# STREAMLIT UI – MULTI-LAYOUT
###############################################################
###############################################################
# LOKASJONSOPTIMERING – GREEDY ASSIGNMENT
###############################################################
def greedy_assignment(df_loc, demand_series, entry_x=0.0, entry_y=0.0):
    locations = []
    for _, row in df_loc.iterrows():
        capacity = int(row.get("antall", 1)) if "antall" in df_loc.columns else 1
        locations.append({
            "lokasjon": int(row["lokasjon"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "remaining": max(1, capacity)
        })

    assignments = []

    for article, demand in demand_series.sort_values(ascending=False).items():
        # Finn nærmeste ledige lokasjon (greedy)
        best_idx = None
        best_cost = None
        for idx, loc in enumerate(locations):
            if loc["remaining"] <= 0:
                continue
            dist = math.sqrt((loc["x"] - entry_x) ** 2 + (loc["y"] - entry_y) ** 2)
            weighted = dist * demand
            if best_cost is None or weighted < best_cost:
                best_cost = weighted
                best_idx = idx

        if best_idx is not None:
            chosen = locations[best_idx]
            chosen["remaining"] -= 1
            assignments.append({
                "artikkel": article,
                "lokasjon": chosen["lokasjon"],
                "x": chosen["x"],
                "y": chosen["y"],
                "vektet_distanse": best_cost
            })

    return pd.DataFrame(assignments)


###############################################################
# UI – SIDEBAR NAVIGASJON
###############################################################
page = st.sidebar.radio(
    "Navigasjon",
    ["📊 Simulering", "🧭 Lokasjonsoptimalisering"],
    key="page_selector"
)

if page == "📊 Simulering":
    st.title("📦 Multi-layout Lager-Simulering")

    num_pickers = st.number_input(
        "Hvor mange plukkere skal simuleringen bruke?", 1, 50, 5
    )
    num_layouts = st.number_input("Hvor mange layouter vil du sammenligne?", 1, 5, 2)
    uploaded = {}

    if "scenarios" not in st.session_state:
        st.session_state["scenarios"] = None

    for i in range(num_layouts):
        uploaded[i] = st.file_uploader(f"Last opp Layout {i+1}", type=["xlsx"], key=f"layout{i}")

    run = st.button("🚀 Kjør simulering for alle layouts")

    if run:
        base_picker_profiles = generate_picker_profiles(num_pickers)
        scenarios = {}
        for i in range(num_layouts):
            if uploaded[i] is None:
                st.error(f"Mangler layout {i+1}")
                st.stop()

            df_loc = pd.read_excel(uploaded[i], sheet_name="lokasjoner")
            df_orders = pd.read_excel(uploaded[i], sheet_name="ordrer")

            st.write(f"⏳ Kjører layout {i+1}…")
            scenarios[f"Layout {i+1}"] = run_simulation(
                df_loc, df_orders, base_picker_profiles
            )

        st.session_state["scenarios"] = scenarios

        st.success("Alle layout-simuleringer fullført! Scroll ned for resultater.")

    scenarios = st.session_state.get("scenarios")

    if scenarios:
        ###############################################################
        # TABS FOR VISUALISERING
        ###############################################################
        layout_tabs = st.tabs(list(scenarios.keys()) + ["📊 Sammenligning"])

        ###############################################################
        # VISUALISERING PER LAYOUT
        ###############################################################
        for tab, (name, result) in zip(layout_tabs, scenarios.items()):
            with tab:
                st.header(name)

                st.subheader("⏱ Total tid")
                st.write(result["total_time_str"])

                st.subheader("📏 Total distanse")
                st.write(f"{result['total_distance_m']:.1f} meter")

                st.subheader("⏳ Tid i kø")
                st.write(format_time(result["total_wait_minutes"] * 60))

                st.subheader("👷 Plukkere")
                st.table(result["pickers"])

                mv = result["movement_df"]
                coord_map = result["coord_map"]

                loc_df = pd.DataFrame(
                    [{"loc": k, "x": coord_map[k][0], "y": coord_map[k][1]} for k in coord_map]
                )

                x_padding = 1
                y_padding = 1
                x_range = [loc_df["x"].min() - x_padding, loc_df["x"].max() + x_padding]
                y_range = [loc_df["y"].min() - y_padding, loc_df["y"].max() + y_padding]

                # LAYOUTTEGNING
                st.subheader("🏗️ Lagerlayout")
                fig_layout, ax_layout = plt.subplots(figsize=(10, 4))
                ax_layout.scatter(result["layout_df"]["x"], result["layout_df"]["y"], c="lightblue", s=200)
                for _, row in result["layout_df"].iterrows():
                    ax_layout.text(row["x"], row["y"], f"{int(row['lokasjon'])}", ha="center", va="center", fontsize=9, fontweight="bold")
                ax_layout.set_xlabel("X (m)")
                ax_layout.set_ylabel("Y (m)")
                ax_layout.set_title("Lagerposisjoner")
                st.pyplot(fig_layout)

                # SPOR
                st.subheader("📍 Plukkernes spor")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.scatter(loc_df["x"], loc_df["y"], c="gray", s=50)
                for pid in mv["picker"].unique():
                    p = mv[mv["picker"] == pid]
                    ax.plot(p["x"], p["y"], label=f"Picker {pid}")
                ax.legend()
                st.pyplot(fig)

                # HEATMAP
                st.subheader("🔥 Heatmap")
                heat_df = pd.DataFrame([
                    {"loc": loc, "visits": result["heatmap"].get(loc, 0),
                     "x": coord_map[loc][0], "y": coord_map[loc][1]}
                    for loc in coord_map
                ])
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                sc = ax2.scatter(heat_df["x"], heat_df["y"],
                                 c=heat_df["visits"], cmap="hot", s=200)
                plt.colorbar(sc, ax=ax2)
                st.pyplot(fig2)

                # PLOTLY-ANIMASJON
                st.subheader("🎬 Interaktiv animasjon (Plotly)")
                mv_plot = mv.copy()
                mv_plot["tid (min)"] = mv_plot["time"].round(2)
                mv_plot["marker_size"] = np.where(mv_plot["event"] == "pick", 16, 10)
                mv_plot.sort_values("time", inplace=True)
                fig_plotly = px.scatter(
                    mv_plot,
                    x="x",
                    y="y",
                    color="picker",
                    symbol="event",
                    symbol_map={"move": "circle", "pick": "star"},
                    animation_frame="tid (min)",
                    animation_group="picker",
                    size="marker_size",
                    size_max=18,
                    range_x=x_range,
                    range_y=y_range,
                    labels={"x": "X (m)", "y": "Y (m)", "picker": "Plukker"},
                    title="Plukkerbevegelser over tid"
                )
                st.plotly_chart(fig_plotly, use_container_width=True)

                # ANIMASJON (MATPLOTLIB-SLIDER)
                st.subheader("🎥 Animasjon med tids-slider")
                max_t = mv["time"].max()
                t_sel = st.slider(f"Tidspunkt – {name}", 0.0, float(max_t),
                                  0.0, step=max(0.01, max_t/200), key=f"slider_{name}")
                st.write(f"⏱ {format_time(t_sel * 60)}")

                frame = mv[mv["time"] <= t_sel]
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.scatter(loc_df["x"], loc_df["y"], c="gray", s=50)
                for pid in frame["picker"].unique():
                    p = frame[frame["picker"] == pid]
                    ax3.plot(p["x"], p["y"], label=f"Picker {pid}")
                    ax3.scatter(p["x"].iloc[-1], p["y"].iloc[-1], s=120)
                ax3.legend()
                st.pyplot(fig3)

        ###############################################################
        # SAMMENLIGNING
        ###############################################################
        with layout_tabs[-1]:
            st.header("📊 Sammenligning av layouts")

            # TOTALTID
            st.subheader("⏱ Total tid per layout")
            df_time = pd.DataFrame([
                {"Layout": name, "Total minutter": res["total_minutes"]}
                for name, res in scenarios.items()
            ])
            st.bar_chart(df_time.set_index("Layout"))

            # Distanse sammenligning
            st.subheader("📏 Total distanse (beregner av alle movement-punkter)")
            dist_data = []
            for name, res in scenarios.items():
                dist_data.append({"Layout": name, "Distanse (m)": round(res["total_distance_m"], 1)})

            st.table(pd.DataFrame(dist_data))

            # KØTID
            st.subheader("⏳ Tid i kø per layout")
            queue_df = pd.DataFrame([
                {"Layout": name, "Køtid (min)": res["total_wait_minutes"]}
                for name, res in scenarios.items()
            ])
            st.bar_chart(queue_df.set_index("Layout"))
    else:
        st.info("Last opp Excel-filer og trykk på \"Kjør simulering\" for å starte.")

elif page == "🧭 Lokasjonsoptimalisering":
    st.title("🧭 Assignment-basert lokasjonsoptimalisering")
    st.markdown(
        """
        Last opp et oppsett med arkene `lokasjoner` og `ordrer`, så beregner vi en
        greedy assignment der artikler med høyest etterspørsel får de nærmeste
        ledige lokasjonene til valgt inngangspunkt. Resultatet kan lastes ned og
        brukes som nytt grunnlag for simuleringene.
        """
    )

    uploaded_opt = st.file_uploader("Last opp layoutfil", type=["xlsx"], key="opt_file")
    col1, col2 = st.columns(2)
    with col1:
        entry_x = st.number_input("Inngang X-posisjon", value=0.0)
    with col2:
        entry_y = st.number_input("Inngang Y-posisjon", value=0.0)

    if st.button("🧮 Beregn forslag"):
        if uploaded_opt is None:
            st.error("Last opp en Excel-fil før du kjører optimeringen.")
            st.stop()

        df_loc = pd.read_excel(uploaded_opt, sheet_name="lokasjoner")
        df_orders = pd.read_excel(uploaded_opt, sheet_name="ordrer")

        if df_orders.empty or df_loc.empty:
            st.error("Filen mangler data i arkene 'lokasjoner' og/eller 'ordrer'.")
            st.stop()

        demand = df_orders.groupby("artikkel").size()
        assignment_df = greedy_assignment(df_loc, demand, entry_x, entry_y)

        if assignment_df.empty:
            st.warning("Ingen forslag generert – sjekk at lokasjoner og artikler er tilgjengelige.")
        else:
            st.success("Ferdig! Tabellen under viser anbefalte plasseringer basert på etterspørsel.")
            st.dataframe(assignment_df)

            csv = assignment_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Last ned forslag (CSV)",
                data=csv,
                mime="text/csv",
                file_name="lokasjonsforslag.csv"
            )
