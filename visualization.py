# visualization.py
import plotly.graph_objects as go
import numpy as np

def heatmap_sections(layout_df, heatmap_dict, num_sections=12):
    xs = layout_df["x"].values
    locs = layout_df["lokasjon"].values
    visits = [heatmap_dict.get(int(l), 0) for l in locs]

    x_min, x_max = xs.min(), xs.max()
    bins = np.linspace(x_min, x_max, num_sections + 1)
    idx = np.digitize(xs, bins) - 1

    agg = np.zeros(num_sections)
    for i, v in zip(idx, visits):
        if 0 <= i < num_sections:
            agg[i] += v

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg,
        y=[f"Sone {i+1}" for i in range(num_sections)],
        orientation='h',
        marker=dict(
            color=agg,
            colorscale="Blues",
            line=dict(color="black", width=1)
        )
    ))

    fig.update_layout(
        title="Heatmap (aggregert i seksjoner)",
        height=420,
        template="plotly_white",
        xaxis_title="Antall besøk",
        yaxis_title="Seksjon"
    )
    return fig
