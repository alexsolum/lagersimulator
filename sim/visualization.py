# sim/visualization.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from matplotlib import colors as mcolors


# -------------------------------------------------------------
# UTIL: Convert long, linear layout into 2D grid (automatic wrap)
# -------------------------------------------------------------
def wrap_layout_grid(layout_df, max_cols=12):
    """
    Hvis layouten er ekstremt lang (én dimensjon),
    lager vi et kunstig grid for heatmap og visualisering.

    Eksempel:
       lokasjoner 1–80 på én linje → grupperes i rader på 12.

    Returnerer:
        layout_df med nye x' og y' for heatmap,
        og mapping tilbake.
    """

    df = layout_df.copy().sort_values("lokasjon")

    # Hvor lang er raden?
    unique_y = df["y"].nunique()
    unique_x = df["x"].nunique()

    # Hvis layouten er naturlig 2D, ikke rør noe
    if unique_y > 3 and unique_x > 3:
        df["x_wrapped"] = df["x"]
        df["y_wrapped"] = df["y"]
        return df

    # Ellers: bygg et kunstig grid
    n = len(df)
    cols = min(max_cols, max(4, int(np.sqrt(n)) * 2))
    rows = int(np.ceil(n / cols))

    df["row"] = df.index // cols
    df["col"] = df.index % cols

    # Normalisert koordinatsystem
    df["x_wrapped"] = df["col"].astype(float)
    df["y_wrapped"] = df["row"].astype(float)

    return df


# -------------------------------------------------------------
# COLOR UTIL
# -------------------------------------------------------------
def _color_with_opacity(color, opacity):
    if isinstance(color, str) and color.startswith("rgb("):
        r, g, b = [float(c) / 255 for c in color[4:-1].split(",")]
    else:
        r, g, b = mcolors.to_rgb(color)

    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{opacity})"


# -------------------------------------------------------------
# STATIC LAYOUT DRAWING
# -------------------------------------------------------------
def build_static_layout(layout_df, title="Lagerlayout"):
    """
    Viser reoler som statiske rektangler.
    """

    fig = go.Figure()

    for _, r in layout_df.iterrows():
        fig.add_shape(
            type="rect",
            x0=r["x"] - 0.6,
            x1=r["x"] + 0.6,
            y0=r["y"] - 0.6,
            y1=r["y"] + 0.6,
            line=dict(color="#2a6fdb", width=1.1),
            fillcolor="#e6f0ff",
            layer="below",
        )

    fig.add_trace(go.Scatter(
        x=layout_df["x"],
        y=layout_df["y"],
        mode="markers+text",
        text=[str(int(l)) for l in layout_df["lokasjon"]],
        textposition="middle center",
        marker=dict(size=8, color="#2a6fdb"),
        hovertemplate="Lokasjon %{text}<br>X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# -------------------------------------------------------------
# HEATMAP BUILDING
# -------------------------------------------------------------
def build_heatmap(layout_df, heatmap_counts):
    """
    Lager et heatmap ENDELIG tilpasset endimensjonale layouts.
    """

    df = layout_df.copy()
    df["visits"] = df["lokasjon"].map(lambda L: heatmap_counts.get(L, 0))

    # Hvis kun én linje → wrap
    df_wrapped = wrap_layout_grid(df)

    # Normer visits
    df_wrapped["z"] = df_wrapped["visits"]

    fig = px.imshow(
        df_wrapped.pivot(index="y_wrapped", columns="x_wrapped", values="z"),
        color_continuous_scale="YlOrRd",
        origin="lower",
    )

    fig.update_layout(
        title="Heatmap (med row wrapping ved behov)",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# -------------------------------------------------------------
# INTERPOLATION FOR ANIMATION
# -------------------------------------------------------------
def interpolate_movements(mv_df, fps=15):
    mv = mv_df.sort_values("time")

    t_min, t_max = mv["time"].min(), mv["time"].max()
    dt = 1 / fps
    T = np.arange(t_min, t_max + dt, dt)

    rows = []

    for pid in mv["picker"].unique():
        p = mv[mv["picker"] == pid].sort_values("time")

        for t in T:
            past = p[p["time"] <= t].tail(1)
            future = p[p["time"] >= t].head(1)

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

            rows.append({
                "time": t,
                "picker": pid,
                "x": x,
                "y": y,
                "event": event,
            })

    out = pd.DataFrame(rows).sort_values(["picker", "time"])
    out["vx"] = out.groupby("picker")["x"].diff().fillna(0) / dt
    out["vy"] = out.groupby("picker")["y"].diff().fillna(0) / dt
    return out


# -------------------------------------------------------------
# ANIMATION
# -------------------------------------------------------------
EVENT_COLORS = {
    "move": "#3498db",   # blå
    "pick": "#2ecc71",   # grønn
    "queue": "#e74c3c",  # rød
}


def build_animation(mv_df, layout_df, fps=15, trail_length=25):
    """
    Full plotly-animasjon med spor + piler + statisk layout.
    """

    mv = mv_df.copy()
    mv["event"] = mv["event"].fillna("move")

    interp = interpolate_movements(mv, fps=fps)
    times = sorted(interp["time"].unique())

    # Plotly fargepalett (unik per picker)
    palette = px.colors.qualitative.Set3
    pickers = sorted(interp["picker"].unique())
    picker_colors = {pid: palette[i % len(palette)] for i, pid in enumerate(pickers)}

    # Statisk reoler
    shapes = []
    for _, r in layout_df.iterrows():
        shapes.append(
            dict(
                type="rect",
                x0=r["x"] - 0.6,
                x1=r["x"] + 0.6,
                y0=r["y"] - 0.6,
                y1=r["y"] + 0.6,
                line=dict(color="#2a6fdb", width=1),
                fillcolor="#e6f0ff",
                layer="below",
            )
        )

    # Frames
    frames = []
    for i, t in enumerate(times):
        frame_df = interp[interp["time"] == t]

        # fade‐trail
        trail_traces = []
        for pid in pickers:
            hist = interp[(interp["picker"] == pid) & (interp["time"] <= t)]
            tail = hist.tail(trail_length)

            if len(tail) < 2:
                continue

            xs = tail["x"].tolist()
            ys = tail["y"].tolist()

            opacities = np.linspace(0.3, 0.9, len(xs)-1)

            for j in range(len(xs)-1):
                trail_traces.append(
                    go.Scatter(
                        x=[xs[j], xs[j+1]],
                        y=[ys[j], ys[j+1]],
                        mode="lines",
                        line=dict(
                            width=3,
                            color=_color_with_opacity(picker_colors[pid], opacities[j])
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # Picker‐markers
        scatter = go.Scatter(
            x=frame_df["x"],
            y=frame_df["y"],
            mode="markers+text",
            marker=dict(
                size=16,
                color=[EVENT_COLORS.get(ev, "#7f8c8d") for ev in frame_df["event"]],
                line=dict(width=2, color=[picker_colors[p] for p in frame_df["picker"]]),
            ),
            text=[f"P{p}" for p in frame_df["picker"]],
            textposition="top center",
            showlegend=False,
            hoverinfo="none",
        )

        # Velocity arrow annotations
        annotations = []
        for _, r in frame_df.iterrows():
            v = np.hypot(r["vx"], r["vy"])
            if v < 0.05:
                continue

            x1 = r["x"] + r["vx"] * 0.25
            y1 = r["y"] + r["vy"] * 0.25

            annotations.append(
                dict(
                    x=x1, y=y1, ax=r["x"], ay=r["y"],
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    arrowcolor=picker_colors[r["picker"]],
                    arrowhead=3,
                    arrowwidth=2,
                    opacity=0.9
                )
            )

        frames.append(
            go.Frame(
                name=f"f{i}",
                data=trail_traces + [scatter],
                layout=go.Layout(shapes=shapes, annotations=annotations),
            )
        )

    # Build figure
    start_data = frames[0].data if frames else []

    fig = go.Figure(data=start_data, frames=frames)

    fig.update_layout(
        title="Plukkerbevegelser (Plotly-animasjon)",
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y"),
        shapes=shapes,
        template="plotly_white",
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    # Animation controls
    step_ms = int(1000 / fps)

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "▶️ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": step_ms, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                },
            ],
        }],
        sliders=[{
            "steps": [
                {
                    "label": f"{t:.2f} min",
                    "method": "animate",
                    "args": [[f"f{i}"], {"frame": {"duration": step_ms, "redraw": True}}],
                }
                for i, t in enumerate(times)
            ],
            "x": 0.05,
            "y": -0.05,
        }],
    )

    return fig
