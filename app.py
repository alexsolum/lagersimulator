import os

import streamlit as st
import simpy
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import plotly.express as px
import plotly.graph_objects as go

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


def assignment_to_layout(assignment_df):
    if assignment_df.empty:
        return pd.DataFrame(columns=["lokasjon", "x", "y", "artikkel", "antall"])

    layout = assignment_df.copy()
    layout["antall"] = 1
    return layout[["lokasjon", "x", "y", "artikkel", "antall"]]


###############################################################
# DEMO-DATA OG VISUALISERINGER
###############################################################
def generate_demo_layout(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    loc_id = 1
    aisle_spacing = 3.0
    slot_spacing = 2.4
    for aisle in range(4):
        x = aisle * aisle_spacing
        for slot in range(8):
            y = slot * slot_spacing
            article = f"A{rng.integers(1, 9)}"
            rows.append({
                "lokasjon": loc_id,
                "x": x,
                "y": y,
                "artikkel": article,
                "antall": rng.integers(1, 3)
            })
            loc_id += 1
    return pd.DataFrame(rows)


def generate_demo_orders(seed, n_orders):
    rng = np.random.default_rng(seed)
    orders = []
    for oid in range(1, n_orders + 1):
        order_size = rng.integers(2, 5)
        for _ in range(order_size):
            orders.append({
                "ordre": f"ORD-{oid:03d}",
                "artikkel": f"A{rng.integers(1, 9)}"
            })
    return pd.DataFrame(orders)


def draw_frame(result, t_sel):
    mv = result["movement_df"]
    layout_df = result["layout_df"]
    coord_map = result["coord_map"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plukkplasser som firkanter
    for _, row in layout_df.iterrows():
        rect = plt.Rectangle(
            (row["x"] - 0.6, row["y"] - 0.6), 1.2, 1.2,
            facecolor="#d6e9ff", edgecolor="#2a6fdb", linewidth=1.5, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(row["x"], row["y"], f"{int(row['lokasjon'])}\n{row['artikkel']}",
                ha="center", va="center", fontsize=8, color="#0a2f73")

    # Plukkerposisjoner over tid
    positions = {}
    trails = {}
    for pid in mv["picker"].unique():
        subset = mv[mv["picker"] == pid]
        current = subset[subset["time"] <= t_sel]
        if not current.empty:
            positions[pid] = current.iloc[-1][["x", "y"]].values
            trails[pid] = current
        else:
            positions[pid] = np.array([0.0, 0.0])
            trails[pid] = subset.head(1)

    colors = px.colors.qualitative.Safe
    for idx, (pid, pos) in enumerate(positions.items()):
        raw_color = colors[idx % len(colors)]
        if raw_color.startswith("rgb("):
            rgb_parts = [int(c.strip()) / 255 for c in raw_color[4:-1].split(",")]
            col = mcolors.to_hex(rgb_parts)
        else:
            col = mcolors.to_hex(raw_color)
        trail = trails.get(pid)
        if trail is not None and len(trail) > 1:
            ax.plot(trail["x"], trail["y"], color=col, linewidth=1.5, alpha=0.6)
        ax.scatter(pos[0], pos[1], s=140, color=col, edgecolor="black", zorder=3)
        ax.text(pos[0], pos[1] + 0.4, f"P{pid}", ha="center", fontsize=9,
                fontweight="bold", color=col)

    xs = [c[0] for c in coord_map.values()]
    ys = [c[1] for c in coord_map.values()]
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("SimPy-basert plukkflyt")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def _derive_zones(layout_df):
    """Return a Series with a simple zone label per radiale rad i layouten."""

    if "zone" in layout_df.columns:
        return layout_df["zone"]

    unique_rows = sorted(layout_df["y"].unique())
    zone_names = {y: f"Sone {idx + 1}" for idx, y in enumerate(unique_rows)}
    return layout_df["y"].map(zone_names)


def _color_with_opacity(color, opacity):
    if isinstance(color, str) and color.startswith("rgb("):
        rgb_parts = [float(c.strip()) / 255 for c in color[4:-1].split(",")]
        r, g, b = rgb_parts
    else:
        r, g, b = mcolors.to_rgb(color)

    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{opacity})"


def _trail_segments(mv_plot, frame_time, trail_length, color_map):
    traces = []
    for pid, color in color_map.items():
        history = mv_plot[(mv_plot["picker"] == pid) & (mv_plot["time"] <= frame_time)]
        tail = history.tail(trail_length + 1)
        if len(tail) < 2:
            continue

        xs = tail["x"].tolist()
        ys = tail["y"].tolist()
        segment_count = len(xs) - 1
        opacities = np.linspace(0.25, 0.9, segment_count)

        for idx in range(segment_count):
            traces.append(
                go.Scatter(
                    x=[xs[idx], xs[idx + 1]],
                    y=[ys[idx], ys[idx + 1]],
                    mode="lines",
                    line=dict(
                        color=_color_with_opacity(color, opacities[idx]),
                        width=3,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=f"picker_{pid}",
                    name=f"spor_P{pid}",
                )
            )

    return traces


def build_plotly_animation(mv_plot, x_range, y_range, layout_df=None, trail_length=12):
    mv_plot = mv_plot.copy()
    mv_plot["tid (min)"] = mv_plot["time"].round(2)
    mv_plot["marker_size"] = np.where(mv_plot["event"] == "pick", 16, 10)
    mv_plot["ikon"] = mv_plot["picker"].apply(lambda p: f"🧍 P{p}")
    mv_plot.sort_values("time", inplace=True)

    palette = px.colors.qualitative.Set3
    picker_colors = {
        pid: palette[idx % len(palette)]
        for idx, pid in enumerate(sorted(mv_plot["picker"].unique()))
    }

    fig = px.scatter(
        mv_plot,
        x="x",
        y="y",
        color="picker",
        symbol="event",
        symbol_map={"move": "circle", "pick": "star"},
        animation_frame="tid (min)",
        animation_group="picker",
        size="marker_size",
        size_max=20,
        range_x=x_range,
        range_y=y_range,
        labels={"x": "X (m)", "y": "Y (m)", "picker": "Plukker"},
        title="Plukkerbevegelser over tid",
        hover_name="ikon",
    )

    fig.update_traces(
        text=mv_plot["ikon"],
        textposition="top center",
        marker=dict(line=dict(color="black", width=1.2), symbol="circle"),
        opacity=0.9,
    )

    if layout_df is not None and not layout_df.empty:
        layout_with_zones = layout_df.copy()
        layout_with_zones["zone"] = _derive_zones(layout_with_zones)

        zone_palette = {
            zone: palette[i % len(palette)]
            for i, zone in enumerate(sorted(layout_with_zones["zone"].unique()))
        }

        shapes = []
        for _, row in layout_with_zones.iterrows():
            color = zone_palette.get(row["zone"], "#d6e9ff")
            shapes.append(
                dict(
                    type="rect",
                    x0=row["x"] - 0.6,
                    x1=row["x"] + 0.6,
                    y0=row["y"] - 0.6,
                    y1=row["y"] + 0.6,
                    line=dict(color="rgba(20, 20, 20, 0.6)", width=1.6),
                    fillcolor=color,
                    opacity=0.35,
                    layer="below",
                )
            )

        fig.update_layout(
            shapes=shapes,
            legend_title_text="Plukker (ikon)",
            plot_bgcolor="#f9f9f9",
            yaxis_scaleanchor="x",
        )

        fig.add_trace(
            go.Scatter(
                x=layout_with_zones["x"],
                y=layout_with_zones["y"],
                mode="text",
                text=[f"{int(row['lokasjon'])}<br>{row['zone']}" for _, row in layout_with_zones.iterrows()],
                textfont=dict(color="#0a2f73", size=10),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if fig.frames:
        def parse_frame_time(frame_name):
            try:
                return float(frame_name)
            except (TypeError, ValueError):
                return float(mv_plot["time"].min())

        for frame in fig.frames:
            frame_time = parse_frame_time(frame.name)
            trail_traces = _trail_segments(mv_plot, frame_time, trail_length, picker_colors)
            frame.data = frame.data + tuple(trail_traces)

        initial_time = parse_frame_time(fig.frames[0].name)
        initial_trails = _trail_segments(mv_plot, initial_time, trail_length, picker_colors)
        fig.add_traces(initial_trails)

    return fig


###############################################################
# UI – SIDEBAR NAVIGASJON
###############################################################
def main():
    page = st.sidebar.radio(
        "Navigasjon",
        ["📊 Simulering", "🧭 Lokasjonsoptimalisering", "🎨 Visuell demo"],
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
                    trail_length = st.slider(
                        "Hvor mange bevegelser skal vises i sporet per plukker?",
                        min_value=1,
                        max_value=200,
                        value=20,
                        step=1,
                        key=f"trail_length_{name}",
                    )
                    fig_plotly = build_plotly_animation(
                        mv, x_range, y_range, result["layout_df"], trail_length=trail_length
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
    
        sim_pickers = st.number_input(
            "Antall plukkere for simulering av baseline vs optimal", 1, 50, 3
        )
    
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
    
            st.session_state["baseline_layout"] = df_loc
            st.session_state["baseline_orders"] = df_orders
            st.session_state["assignment_df"] = assignment_df
    
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
    
        assignment_df = st.session_state.get("assignment_df")
        base_layout = st.session_state.get("baseline_layout")
        base_orders = st.session_state.get("baseline_orders")
    
        st.markdown("---")
        st.subheader("🚀 Simuler og sammenlign med baseline")
        st.markdown(
            "Kjør en rask simulering av nåværende layout mot det foreslåtte oppsettet "
            "for å se potensielle gevinster. Samme plukkerprofiler brukes for begge "
            "kjøringer."
        )
    
        if st.button("🎯 Simuler baseline og optimal layout"):
            if assignment_df is None or base_layout is None or base_orders is None:
                st.error("Kjør først optimeringen og generer et forslag før simulering.")
                st.stop()
    
            picker_profiles = generate_picker_profiles(sim_pickers)
    
            optimized_layout = assignment_to_layout(assignment_df)
    
            baseline_result = run_simulation(base_layout, base_orders, picker_profiles)
            optimal_result = run_simulation(optimized_layout, base_orders, picker_profiles)
    
            st.session_state["opt_compare"] = {
                "baseline": baseline_result,
                "optimal": optimal_result,
            }
    
        compare = st.session_state.get("opt_compare")
    
        if compare:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 🏁 Baseline")
                st.metric(
                    "Total tid",
                    compare["baseline"]["total_time_str"],
                )
                st.metric(
                    "Total distanse (m)", f"{compare['baseline']['total_distance_m']:.1f}"
                )
                st.metric(
                    "Tid i kø",
                    format_time(compare["baseline"]["total_wait_minutes"] * 60)
                )
    
            with col_b:
                st.markdown("### 🆕 Optimalisert")
                st.metric(
                    "Total tid",
                    compare["optimal"]["total_time_str"],
                    delta=f"{compare['baseline']['total_minutes'] - compare['optimal']['total_minutes']:.2f} min"
                )
                st.metric(
                    "Total distanse (m)",
                    f"{compare['optimal']['total_distance_m']:.1f}",
                    delta=f"{compare['baseline']['total_distance_m'] - compare['optimal']['total_distance_m']:.1f} m"
                )
                st.metric(
                    "Tid i kø",
                    format_time(compare["optimal"]["total_wait_minutes"] * 60),
                    delta=f"{compare['baseline']['total_wait_minutes'] - compare['optimal']['total_wait_minutes']:.2f} min"
                )
    
            st.markdown("### 📈 Visualisering av optimal layout")
            optimal_layout_df = compare["optimal"]["layout_df"]
            fig_layout_opt, ax_layout_opt = plt.subplots(figsize=(10, 4))
            ax_layout_opt.scatter(optimal_layout_df["x"], optimal_layout_df["y"], c="lightgreen", s=200)
            for _, row in optimal_layout_df.iterrows():
                ax_layout_opt.text(row["x"], row["y"], f"{int(row['lokasjon'])}", ha="center", va="center", fontsize=9, fontweight="bold")
            ax_layout_opt.set_xlabel("X (m)")
            ax_layout_opt.set_ylabel("Y (m)")
            ax_layout_opt.set_title("Optimal layout fra assignment")
            st.pyplot(fig_layout_opt)
    
    elif page == "🎨 Visuell demo":
        st.title("🎨 Visuell SimPy-simulering")
        st.markdown(
            """
            Denne siden kjører en forhåndsdefinert SimPy-simulering og viser en mer
            realistisk visualisering av lageret. Plukkplasser tegnes som firkanter,
            og plukkere vises som små runde figurer som beveger seg mellom lokasjonene.
            """
        )
    
        col_demo_a, col_demo_b = st.columns(2)
        with col_demo_a:
            demo_pickers = st.slider("Antall plukkere", 1, 10, 4)
            demo_orders = st.slider("Antall ordrer", 1, 30, 10)
        with col_demo_b:
            demo_seed = st.number_input("Tilfeldig seed", value=42, step=1)
            base_pick = st.number_input("Plukktid per vare (sek)", 5, 120, 12)
    
    
        if st.button("🚀 Kjør demovisualisering"):
            global BASE_PICK_SECONDS, BASE_PICK_MIN
            BASE_PICK_SECONDS = base_pick
            BASE_PICK_MIN = BASE_PICK_SECONDS / 60
    
            layout_df = generate_demo_layout(seed=demo_seed)
            orders_df = generate_demo_orders(seed=demo_seed + 1, n_orders=demo_orders)
            picker_profiles = generate_picker_profiles(demo_pickers, seed=demo_seed + 2)
    
            st.write("Simulerer ...")
            demo_result = run_simulation(layout_df, orders_df, picker_profiles)
            st.session_state["demo_result"] = demo_result
            st.success("Demovisualisering klar! Bruk slideren under for å se bevegelser.")
    
        demo_result = st.session_state.get("demo_result")
    
        if demo_result:
            st.markdown("---")
            st.metric("Total simuleringstid", demo_result["total_time_str"])
            st.metric("Total distanse (m)", f"{demo_result['total_distance_m']:.1f}")
    
            max_t = float(demo_result["movement_df"]["time"].max())
            t_sel = st.slider("Tidspunkt i simuleringen (min)", 0.0, max_t, 0.0,
                              step=max(0.05, max_t / 200), key="demo_time_slider")
            fig_demo = draw_frame(demo_result, t_sel)
            st.pyplot(fig_demo)
    
            st.caption(
                "Plukkplasser (firkantene) viser lokasjons-ID og artikkel. Plukkerne "
                "vises som fargede sirkler som beveger seg i SimPy-simuleringen."
            )

if os.environ.get("STREAMLIT_SUPPRESS_UI") != "1":
    main()
