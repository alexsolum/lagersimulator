import os
import sys
from pathlib import Path

# Disable Streamlit UI side effects before importing the app module
os.environ["STREAMLIT_SUPPRESS_UI"] = "1"
os.environ["MPLBACKEND"] = "Agg"

# Ensure repository root is on the import path for direct app import
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app import (
    build_plotly_animation,
    draw_frame,
    generate_picker_profiles,
    run_simulation,
)


def demo_data():
    layout_df = pd.DataFrame(
        [
            {"lokasjon": 1, "x": 0.0, "y": 0.0, "artikkel": "A1", "antall": 2},
            {"lokasjon": 2, "x": 2.0, "y": 0.0, "artikkel": "A2", "antall": 1},
            {"lokasjon": 3, "x": 4.0, "y": 0.0, "artikkel": "A3", "antall": 1},
        ]
    )
    orders_df = pd.DataFrame(
        [
            {"ordre": "O1", "artikkel": "A1"},
            {"ordre": "O1", "artikkel": "A2"},
            {"ordre": "O2", "artikkel": "A3"},
        ]
    )
    return layout_df, orders_df


def test_run_simulation_produces_movement_and_heatmap():
    layout_df, orders_df = demo_data()
    profiles = generate_picker_profiles(2, seed=123)

    result = run_simulation(layout_df, orders_df, profiles)

    assert not result["movement_df"].empty
    assert set(result["coord_map"].keys()) == set(layout_df["lokasjon"])
    assert any(count > 0 for count in result["heatmap"].values())
    assert (result["pickers"]["distance_m"] >= 0).all()


def test_draw_frame_returns_figure_with_rectangles():
    layout_df, orders_df = demo_data()
    profiles = generate_picker_profiles(1, seed=99)
    result = run_simulation(layout_df, orders_df, profiles)

    fig = draw_frame(result, result["movement_df"]["time"].max())

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes[0].patches) == len(layout_df)
    plt.close(fig)


def test_build_plotly_animation_creates_frames():
    layout_df, orders_df = demo_data()
    profiles = generate_picker_profiles(1, seed=2024)
    result = run_simulation(layout_df, orders_df, profiles)

    mv = result["movement_df"]
    x_range = [layout_df["x"].min() - 1, layout_df["x"].max() + 1]
    y_range = [layout_df["y"].min() - 1, layout_df["y"].max() + 1]

    fig = build_plotly_animation(mv, x_range, y_range, layout_df)
    fig_dict = fig.to_dict()

    assert len(fig_dict.get("frames", [])) > 0
    assert len(fig_dict.get("data", [])) > 0
    assert fig.layout.xaxis.range == tuple(x_range)
    assert fig.layout.yaxis.range == tuple(y_range)
    assert len(fig.layout.shapes) == len(layout_df)

    picker_texts = []
    for trace in fig.data:
        if trace.text is None:
            continue
        if isinstance(trace.text, (list, tuple)):
            picker_texts.extend(trace.text)
        else:
            picker_texts.append(trace.text)

    assert any("🧍" in str(text) for text in picker_texts)
