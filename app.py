import os

import streamlit as st
import simpy
import pandas as pd
import numpy as np
import math
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

        current_pos = coord_map.get(current, (0.0, 0.0)) if current is not None else (0.0, 0.0)
        mv_log.append((request_time, pid, current_pos[0], current_pos[1], "queue"))
        if wait_time > 0:
            mv_log.append((env.now, pid, current_pos[0], current_pos[1], "queue"))

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

            queue_start = env.now
            yield req

            if env.now > queue_start:
                mv_log.append((queue_start, pid, current_pos[0], current_pos[1], "queue"))
                mv_log.append((env.now, pid, current_pos[0], current_pos[1], "queue"))

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


EVENT_COLORS = {
    "pick": "#2ecc71",  # grønn
    "move": "#3498db",  # blå
    "queue": "#e74c3c",  # rød
}


def _build_static_shapes(layout_df):
    shapes = []
    if layout_df is None or layout_df.empty:
        return shapes

    for _, row in layout_df.iterrows():
        shapes.append(
            dict(
                type="rect",
                x0=row["x"] - 0.6,
                x1=row["x"] + 0.6,
                y0=row["y"] - 0.6,
                y1=row["y"] + 0.6,
                line=dict(color="#2a6fdb", width=1.2),
                fillcolor="#d6e9ff",
                layer="below",
                opacity=0.8,
            )
        )
    return shapes


def _interpolate_movements(mv_df, fps=15):
    if mv_df.empty:
        return mv_df.copy()

    mv_df = mv_df.sort_values("time")
    t_min, t_max = mv_df["time"].min(), mv_df["time"].max()
    if fps <= 0:
        fps = 1
    step = 1 / fps
    timeline = np.arange(t_min, t_max + step, step)

    rows = []
    for pid in sorted(mv_df["picker"].unique()):
        path = mv_df[mv_df["picker"] == pid].sort_values("time")
        for t in timeline:
            past = path[path["time"] <= t].tail(1)
            future = path[path["time"] >= t].head(1)

            if past.empty and future.empty:
                continue
            if past.empty:
                past = future
            if future.empty:
                future = past

            t0 = past.iloc[0]["time"]
            t1 = future.iloc[0]["time"]
            x0, y0 = past.iloc[0][["x", "y"]]
            x1, y1 = future.iloc[0][["x", "y"]]

            if t1 == t0:
                frac = 0
            else:
                frac = (t - t0) / (t1 - t0)

            x = x0 + (x1 - x0) * frac
            y = y0 + (y1 - y0) * frac

            event = past.iloc[0]["event"]
            rows.append({"time": t, "picker": pid, "x": x, "y": y, "event": event})

    interp = pd.DataFrame(rows)
    interp.sort_values(["picker", "time"], inplace=True)
    interp["vx"] = interp.groupby("picker")["x"].diff().fillna(0) / step
    interp["vy"] = interp.groupby("picker")["y"].diff().fillna(0) / step
    interp["color"] = interp["event"].map(EVENT_COLORS).fillna("#95a5a6")
    return interp


def _velocity_annotations(frame_points, picker_colors, scale=0.25):
    annotations = []
    for _, row in frame_points.iterrows():
        speed = math.sqrt(row.get("vx", 0) ** 2 + row.get("vy", 0) ** 2)
        if speed < 0.05:
            continue
        x0, y0 = row["x"], row["y"]
        x1 = x0 + row.get("vx", 0) * scale
        y1 = y0 + row.get("vy", 0) * scale
        annotations.append(
            dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowwidth=2,
                arrowcolor=picker_colors.get(row["picker"], "#2c3e50"),
                opacity=0.9,
            )
        )
    return annotations


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


def build_plotly_animation(mv_plot, x_range, y_range, layout_df=None, trail_length=24, fps=15):
    mv_plot = mv_plot.copy()
    mv_plot.sort_values("time", inplace=True)
    mv_plot["event"] = mv_plot["event"].fillna("move")

    interpolated = _interpolate_movements(mv_plot, fps=fps)

    palette = px.colors.qualitative.Set3
    picker_colors = {
        pid: palette[idx % len(palette)]
        for idx, pid in enumerate(sorted(interpolated["picker"].unique()))
    }

    shapes = _build_static_shapes(layout_df)
    frames = []
    unique_times = sorted(interpolated["time"].unique())

    for idx, t in enumerate(unique_times):
        frame_points = interpolated[interpolated["time"] == t]
        trail_traces = _trail_segments(interpolated, t, trail_length, picker_colors)
        marker_outline = [picker_colors.get(p, "#2c3e50") for p in frame_points["picker"]]
        texts = [f"P{p}" for p in frame_points["picker"]]

        scatter = go.Scatter(
            x=frame_points["x"],
            y=frame_points["y"],
            mode="markers+text",
            marker=dict(
                size=16,
                color=frame_points["color"],
                line=dict(color=marker_outline, width=2),
            ),
            text=texts,
            textposition="top center",
            hovertemplate=(
                "<b>P%{customdata[0]}</b><br>" "Tid: %{customdata[1]:.2f} min<br>"
                "Hendelse: %{customdata[2]}<extra></extra>"
            ),
            customdata=np.stack(
                [frame_points["picker"], frame_points["time"], frame_points["event"]],
                axis=-1,
            ),
            showlegend=False,
            name=f"frame_{idx}",
        )

        annotations = _velocity_annotations(frame_points, picker_colors)
        frames.append(
            go.Frame(
                name=f"frame-{idx}",
                data=trail_traces + [scatter],
                layout=go.Layout(shapes=shapes, annotations=annotations),
            )
        )

    start_data = frames[0].data if frames else []

    fig = go.Figure(data=start_data, frames=frames)
    fig.update_layout(
        title="Plukkerbevegelser over tid (Plotly)",
        xaxis=dict(range=x_range, title="X (m)"),
        yaxis=dict(range=y_range, title="Y (m)", scaleanchor="x", scaleratio=1),
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
        shapes=shapes,
        legend_title="Hendelser",
    )

    slider_steps = [
        {
            "args": [[f"frame-{idx}"], {"frame": {"duration": int(1000 / fps), "redraw": True}, "mode": "immediate"}],
            "label": f"{t:.2f} min",
            "method": "animate",
        }
        for idx, t in enumerate(unique_times)
    ]

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶️ Spille av",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": int(1000 / fps), "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": slider_steps,
                "currentvalue": {"prefix": "Tid: "},
                "x": 0,
                "y": -0.05,
                "len": 1.0,
            }
        ],
    )

    for event, color in EVENT_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=color, size=12),
                name=event.capitalize(),
                showlegend=True,
                hoverinfo="skip",
            )
        )

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
                    layout_fig = go.Figure()
                    layout_fig.update_layout(
                        shapes=_build_static_shapes(result["layout_df"]),
                        xaxis=dict(range=x_range, title="X (m)"),
                        yaxis=dict(range=y_range, title="Y (m)", scaleanchor="x", scaleratio=1),
                        title="Lagerposisjoner",
                        template="plotly_white",
                        height=420,
                    )
                    layout_fig.add_trace(
                        go.Scatter(
                            x=result["layout_df"]["x"],
                            y=result["layout_df"]["y"],
                            mode="markers+text",
                            text=[f"{int(row['lokasjon'])}" for _, row in result["layout_df"].iterrows()],
                            textposition="middle center",
                            marker=dict(color="#2a6fdb", size=8),
                            hovertemplate="Lokasjon %{text}<br>X=%{x:.2f}, Y=%{y:.2f}<extra></extra>",
                            showlegend=False,
                        )
                    )
                    st.plotly_chart(layout_fig, use_container_width=True)

                    # SPOR
                    st.subheader("📍 Plukkernes spor")
                    trail_fig = go.Figure()
                    palette = px.colors.qualitative.Set3
                    for idx, pid in enumerate(sorted(mv["picker"].unique())):
                        path = mv[mv["picker"] == pid]
                        color = palette[idx % len(palette)]
                        trail_fig.add_trace(
                            go.Scatter(
                                x=path["x"],
                                y=path["y"],
                                mode="lines+markers",
                                name=f"Picker {pid}",
                                line=dict(color=color, width=2),
                                marker=dict(size=6, color=color),
                            )
                        )
                    trail_fig.update_layout(
                        xaxis=dict(range=x_range, title="X (m)"),
                        yaxis=dict(range=y_range, title="Y (m)", scaleanchor="x", scaleratio=1),
                        template="plotly_white",
                        height=380,
                    )
                    st.plotly_chart(trail_fig, use_container_width=True)

                    # HEATMAP
                    st.subheader("🔥 Heatmap")
                    heat_df = pd.DataFrame([
                        {"loc": loc, "visits": result["heatmap"].get(loc, 0),
                         "x": coord_map[loc][0], "y": coord_map[loc][1]}
                        for loc in coord_map
                    ])
                    if not heat_df.empty:
                        heat_fig = px.density_mapbox(
                            heat_df,
                            lat="y",
                            lon="x",
                            z="visits",
                            radius=18,
                            center=dict(lat=heat_df["y"].mean(), lon=heat_df["x"].mean()),
                            zoom=14,
                            mapbox_style="open-street-map",
                            height=420,
                        )
                        heat_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(heat_fig, use_container_width=True)

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
                        mv,
                        x_range,
                        y_range,
                        result["layout_df"],
                        trail_length=trail_length,
                    )
                    st.plotly_chart(fig_plotly, use_container_width=True)
    
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
            x_padding = 1
            y_padding = 1
            x_range = [optimal_layout_df["x"].min() - x_padding, optimal_layout_df["x"].max() + x_padding]
            y_range = [optimal_layout_df["y"].min() - y_padding, optimal_layout_df["y"].max() + y_padding]

            fig_layout_opt = go.Figure()
            fig_layout_opt.update_layout(
                shapes=_build_static_shapes(optimal_layout_df),
                xaxis=dict(range=x_range, title="X (m)"),
                yaxis=dict(range=y_range, title="Y (m)", scaleanchor="x", scaleratio=1),
                template="plotly_white",
                title="Optimal layout fra assignment",
                height=420,
            )
            fig_layout_opt.add_trace(
                go.Scatter(
                    x=optimal_layout_df["x"],
                    y=optimal_layout_df["y"],
                    mode="markers+text",
                    text=[f"{int(row['lokasjon'])}" for _, row in optimal_layout_df.iterrows()],
                    textposition="middle center",
                    marker=dict(color="#27ae60", size=9),
                    hovertemplate="Lokasjon %{text}<br>X=%{x:.2f}, Y=%{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )
            st.plotly_chart(fig_layout_opt, use_container_width=True)
    
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

            mv_demo = demo_result["movement_df"]
            coord_map = demo_result["coord_map"]
            loc_df = pd.DataFrame(
                [{"loc": k, "x": coord_map[k][0], "y": coord_map[k][1]} for k in coord_map]
            )

            x_padding = 1
            y_padding = 1
            x_range = [loc_df["x"].min() - x_padding, loc_df["x"].max() + x_padding]
            y_range = [loc_df["y"].min() - y_padding, loc_df["y"].max() + y_padding]

            trail_length_demo = st.slider(
                "Sporlengde i demo", 1, 200, 30, step=1, key="demo_trail_length"
            )

            fig_demo = build_plotly_animation(
                mv_demo,
                x_range,
                y_range,
                demo_result["layout_df"],
                trail_length=trail_length_demo,
            )
            st.plotly_chart(fig_demo, use_container_width=True)

            st.caption(
                "Plukkplasser tegnes som statiske reoler i bakgrunnen. Fargene på plukkerne "
                "viser hendelser (grønn=plukk, blå=bevegelse, rød=venting), og pilene viser "
                "retningen til bevegelsen."
            )

if os.environ.get("STREAMLIT_SUPPRESS_UI") != "1":
    main()
