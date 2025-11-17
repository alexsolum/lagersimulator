import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

# Ensure repository root is importable when running pytest directly
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

from app import run_uploaded_scenarios
from sim.utils import generate_demo_layout, generate_demo_orders


def _make_upload(layout_seed: int, orders_seed: int, n_orders: int) -> BytesIO:
    layout_df = generate_demo_layout(seed=layout_seed)
    orders_df = generate_demo_orders(orders_seed, n_orders)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        layout_df.to_excel(writer, sheet_name="lokasjoner", index=False)
        orders_df.to_excel(writer, sheet_name="ordrer", index=False)

    buffer.seek(0)
    return buffer


def test_two_layout_uploads_are_simulated_and_rendered():
    os.environ["STREAMLIT_SUPPRESS_UI"] = "1"

    upload_one = _make_upload(layout_seed=1, orders_seed=11, n_orders=5)
    upload_two = _make_upload(layout_seed=2, orders_seed=12, n_orders=5)

    scenarios = run_uploaded_scenarios([upload_one, upload_two], num_pickers=2)

    assert set(scenarios.keys()) == {"Layout 1", "Layout 2"}
    assert all(not res["movement_df"].empty for res in scenarios.values())

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60)
    at.session_state["sim_scenarios"] = scenarios
    at.run()

    assert not at.exception

    tab_labels = [tab.label for tab in at.tabs]
    assert "Layout 1" in tab_labels and "Layout 2" in tab_labels
    assert any(metric.label == "Total tid" for metric in at.metric)
