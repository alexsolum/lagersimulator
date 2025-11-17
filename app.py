import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# === IMPORTER MODULENE I DET NYE SYSTEMET === #
from sim.simulation import run_simulation
from sim.utils import (
    generate_demo_layout,
    generate_demo_orders,
    generate_picker_profiles,
)
from sim.visualization import (
    build_static_layout,
    build_heatmap,
    build_animation
)
from sim.optimization import (
    greedy_assignment,
    assignment_to_layout
)


# ==========================================================
#                   HOVED-APP / NAVIGASJON
# ==========================================================
def main():
    st.set_page_config(
        page_title="Lagersimulering",
        page_icon="📦",
        layout="wide",
    )

    page = st.sidebar.radio(
        "Navigasjon",
        [
            "📊 Simulering",
            "🧭 Lokasjonsoptimalisering",
            "🎨 Visuell demo"
        ]
    )

    if page == "📊 Simulering":
        page_simulation()

    elif page == "🧭 Lokasjonsoptimalisering":
        page_optimization()

    elif page == "🎨 Visuell demo":
        page_visual_demo()


# ==========================================================
#                  SIDE 1 — SIMULERING
# ==========================================================
def page_simulation():
    st.title("📦 Multi-layout Lager-simulering")

    num_pickers = st.number_input(
        "Antall plukkere", 1, 50, 5
    )

    num_layouts = st.number_input(
        "Antall layouts du vil sammenligne", 1, 5, 2
    )

    uploads = []
    for i in range(num_layouts):
        up = st.file_uploader(f"Last opp layout {i+1} (Excel)", type=["xlsx"], key=f"ul{i}")
        uploads.append(up)

    if st.button("🚀 Kjør simulering"):
        scenarios = {}

        profiles = generate_picker_profiles(num_pickers)

        for i in range(num_layouts):
            if uploads[i] is None:
                st.error(f"Layout {i+1} mangler!")
                st.stop()

            df_loc = pd.read_excel(uploads[i], sheet_name="lokasjoner")
            df_orders = pd.read_excel(uploads[i], sheet_name="ordrer")

            st.write(f"⏳ Kjør layout {i+1} …")

            result = run_simulation(df_loc, df_orders, profiles)
            scenarios[f"Layout {i+1}"] = result

        st.session_state["sim_scenarios"] = scenarios
        st.success("Kjøring ferdig! Scroll ned.")

    scenarios = st.session_state.get("sim_scenarios")

    if not scenarios:
        st.info("Last opp Excel-filer og trykk på *Kjør simulering*")
        return

    # Tabs
    tabs = st.tabs(list(scenarios.keys()) + ["📊 Sammenligning"])

    # ---------- PER-LAYOUT VISNING ----------
    for tab, (name, result) in zip(tabs, scenarios.items()):
        with tab:
            st.header(name)

            st.metric("Total tid", result["total_time_str"])
            st.metric("Total distanse (m)", f"{result['total_distance_m']:.1f}")
            st.metric("Tid i kø", result["total_wait_minutes"])

            st.subheader("👷 Plukkere")
            st.table(result["pickers"])

            layout_df = result["layout_df"]
            mv = result["movement_df"]
            heat = result["heatmap"]

            # --- layout tegning ---
            st.subheader("🏗️ Lagerlayout")
            fig_layout = build_static_layout(layout_df)
            st.plotly_chart(fig_layout, use_container_width=True)

            # --- trail / path plot ---
            st.subheader("📍 Plukkernes spor")
            fig_trail = px.line(
                mv,
                x="x", y="y",
                color="picker",
                title="Bevegelsesspor"
            )
            fig_trail.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig_trail, use_container_width=True)

            # --- Heatmap ---
            st.subheader("🔥 Heatmap (automatisk row-wrapping)")
            fig_heat = build_heatmap(layout_df, heat)
            st.plotly_chart(fig_heat, use_container_width=True)

            # --- Animasjon ---
            st.subheader("🎬 Animasjon")
            fps = st.slider("FPS", 5, 40, 15, key=f"fps_{name}")
            trail_len = st.slider("Sporlengde", 5, 200, 30, key=f"tr_{name}")

            fig_anim = build_animation(
                mv,
                layout_df,
                fps=fps,
                trail_length=trail_len
            )
            st.plotly_chart(fig_anim, use_container_width=True)

    # ---------- SAMMENLIGNING ----------
    with tabs[-1]:
        st.header("📊 Sammenligning")
        df = pd.DataFrame([
            {"Layout": k,
             "Total tid (min)": v["total_minutes"],
             "Distanse (m)": v["total_distance_m"],
             "Køtid (min)": v["total_wait_minutes"]}
            for k, v in scenarios.items()
        ])
        st.dataframe(df)

        st.bar_chart(df.set_index("Layout")["Total tid (min)"])


# ==========================================================
#                  SIDE 2 — OPTIMALISERING
# ==========================================================
def page_optimization():
    st.title("🧭 Lokasjonsoptimalisering (Greedy)")

    uploaded = st.file_uploader("Last opp layoutfil", type=["xlsx"])
    if not uploaded:
        return

    df_loc = pd.read_excel(uploaded, sheet_name="lokasjoner")
    df_orders = pd.read_excel(uploaded, sheet_name="ordrer")

    entry_x = st.number_input("Inngang X", value=0.0)
    entry_y = st.number_input("Inngang Y", value=0.0)

    if st.button("🧮 Beregn greedy assignment"):
        demand = df_orders.groupby("artikkel").size()

        assign_df = greedy_assignment(df_loc, demand, entry_x, entry_y)

        st.session_state["opt_assign"] = assign_df
        st.session_state["opt_base_loc"] = df_loc
        st.session_state["opt_orders"] = df_orders

        st.success("Ferdig!")
        st.dataframe(assign_df)

    assign_df = st.session_state.get("opt_assign")
    base_loc = st.session_state.get("opt_base_loc")
    base_orders = st.session_state.get("opt_orders")

    if assign_df is None:
        return

    st.subheader("🚀 Sammenlign baseline vs optimalisert")

    pickers = st.number_input("Antall plukkere", 1, 30, 4)

    if st.button("Kjør simulering"):
        profiles = generate_picker_profiles(pickers)

        opt_layout = assignment_to_layout(assign_df)

        res_base = run_simulation(base_loc, base_orders, profiles)
        res_opt = run_simulation(opt_layout, base_orders, profiles)

        st.session_state["opt_compare"] = {
            "baseline": res_base,
            "optimal": res_opt,
        }

    comp = st.session_state.get("opt_compare")

    if comp:
        base = comp["baseline"]
        opt = comp["optimal"]

        st.metric("Baseline tid", base["total_time_str"])
        st.metric("Optimal tid", opt["total_time_str"],
                  delta=f"{round(base['total_minutes'] - opt['total_minutes'], 2)} min")

        st.subheader("Optimal layout")
        st.plotly_chart(build_static_layout(opt["layout_df"]), use_container_width=True)


# ==========================================================
#                  SIDE 3 — VISUELL DEMO
# ==========================================================
def page_visual_demo():
    st.title("🎨 Visuell SimPY-demo")

    demo_pickers = st.slider("Plukkere", 1, 10, 4)
    demo_orders = st.slider("Ordrer", 1, 30, 10)
    demo_seed = st.number_input("Seed", 1, 9999, 42)

    if st.button("🚀 Kjør demo"):
        layout = generate_demo_layout(seed=demo_seed)
        orders = generate_demo_orders(demo_seed + 1, demo_orders)
        profiles = generate_picker_profiles(demo_pickers, seed=demo_seed + 2)

        result = run_simulation(layout, orders, profiles)

        st.session_state["demo_result"] = result
        st.success("Demo ferdig!")

    res = st.session_state.get("demo_result")
    if not res:
        return

    layout_df = res["layout_df"]
    mv = res["movement_df"]
    heat = res["heatmap"]

    st.metric("Tid", res["total_time_str"])
    st.metric("Distanse", f"{res['total_distance_m']:.1f} m")

    st.subheader("Layout")
    st.plotly_chart(build_static_layout(layout_df), use_container_width=True)

    st.subheader("Heatmap")
    st.plotly_chart(build_heatmap(layout_df, heat), use_container_width=True)

    st.subheader("Animasjon")
    fps = st.slider("FPS", 5, 40, 15)
    trail = st.slider("Sporlengde", 5, 200, 30)

    fig_anim = build_animation(mv, layout_df, fps=fps, trail_length=trail)
    st.plotly_chart(fig_anim, use_container_width=True)


# ==========================================================
#             BACKWARD-COMPATIBLE VIS HELPERS FOR TESTS
# ==========================================================
def draw_frame(result, time):
    layout_df = result["layout_df"]
    mv = result["movement_df"]

    fig, ax = plt.subplots()

    for _, row in layout_df.iterrows():
        rect = mpatches.Rectangle(
            (row["x"] - 0.6, row["y"] - 0.6),
            1.2,
            1.2,
            edgecolor="#2a6fdb",
            facecolor="#e6f0ff",
        )
        ax.add_patch(rect)

    latest_positions = (
        mv[mv["time"] <= time]
        .sort_values("time")
        .groupby("picker")
        .tail(1)
    )

    if not latest_positions.empty:
        ax.scatter(latest_positions["x"], latest_positions["y"], c="#2a6fdb")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    x_min, x_max = layout_df["x"].min(), layout_df["x"].max()
    y_min, y_max = layout_df["y"].min(), layout_df["y"].max()
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)

    return fig


def build_plotly_animation(mv_df, x_range, y_range, layout_df, trail_length=25):
    fig = build_animation(mv_df, layout_df, trail_length=trail_length)

    fig.update_layout(
        xaxis=dict(range=tuple(x_range), scaleanchor="y", scaleratio=1),
        yaxis=dict(range=tuple(y_range)),
    )

    for trace in fig.data:
        if trace.text is None:
            continue
        if isinstance(trace.text, (list, tuple)):
            trace.text = [f"🧍 {t}" for t in trace.text]
        else:
            trace.text = f"🧍 {trace.text}"

    return fig


# ==========================================================
#                      MAIN
# ==========================================================
if __name__ == "__main__":
    if os.environ.get("STREAMLIT_SUPPRESS_UI") != "1":
        main()
