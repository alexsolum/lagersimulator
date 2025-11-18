import streamlit as st
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas
import uuid
import json

st.set_page_config(page_title="Warehouse Layout Designer", layout="wide")

st.title("🏗️ Warehouse Layout Designer")
st.markdown(
    """
Tegn lagerlayouten din med reoler, gangareal, enveiskjørte piler og start-/sluttsone.  
Deretter kan du laste ned layouten som CSV og bruke den i simulering eller optimalisering.
"""
)

# ---------------------------------------------
# Sidebar – verktøyvalg
# ---------------------------------------------
st.sidebar.header("Verktøy")

drawing_mode = st.sidebar.selectbox(
    "Tegnemodus",
    ["Reol (rack)", "Gangareal", "Enveis gang (pil)", "Startpunkt", "Sluttpunkt"],
)

stroke_color = st.sidebar.color_picker("Strekfarge", "#000000")
fill_color = st.sidebar.color_picker("Fyllfarge", "#AAAAAA")

stroke_width = st.sidebar.slider("Strekbredde", 1, 8, 2)
width = st.sidebar.slider("Canvas bredde", 800, 2000, 1200)
height = st.sidebar.slider("Canvas høyde", 400, 1500, 700)

st.sidebar.write("---")
export_btn = st.sidebar.button("Generér data fra tegningen")

# ---------------------------------------------
# Tegne-Canvas
# ---------------------------------------------
drawing_mode_map = {
    "Reol (rack)": "rect",
    "Gangareal": "rect",
    "Enveis gang (pil)": "arrow",
    "Startpunkt": "rect",
    "Sluttpunkt": "rect",
}

canvas_result = st_canvas(
    fill_color=fill_color,
    stroke_color=stroke_color,
    stroke_width=stroke_width,
    background_color="#FFFFFF",
    update_streamlit=True,
    height=height,
    width=width,
    drawing_mode=drawing_mode_map[drawing_mode],
    key="warehouse_canvas",
)

# ---------------------------------------------
# Data-konvertering fra tegninger
# ---------------------------------------------
def parse_canvas_objects(objs):
    """Konverterer Streamlit-canvas data → layout tabell + path graf"""
    if objs is None:
        return pd.DataFrame(), pd.DataFrame()

    rack_rows = []
    path_rows = []
    start_rows = []
    end_rows = []

    for obj in objs:
        obj_type = obj.get("type")
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 0)
        height = obj.get("height", 0)

        # Rektangler (racks, gangareal, start, slutt)
        if obj_type == "rect":
            fc = obj.get("fill", "").lower()

            entry = {
                "lokasjon": str(uuid.uuid4())[:8],
                "x": left + width / 2,
                "y": top + height / 2,
                "bredde": width,
                "høyde": height,
                "farge": obj.get("fill", None),
            }

            # Klassifiser etter valgt farge (eller senere: inspector mode)
            if fc == fill_color.lower():  # bruker valgt farge som indikator
                if drawing_mode == "Reol (rack)":
                    entry["type"] = "rack"
                    rack_rows.append(entry)
                elif drawing_mode == "Gangareal":
                    entry["type"] = "aisle"
                    rack_rows.append(entry)
                elif drawing_mode == "Startpunkt":
                    entry["type"] = "start"
                    start_rows.append(entry)
                elif drawing_mode == "Sluttpunkt":
                    entry["type"] = "end"
                    end_rows.append(entry)

        # Piler = enveis path
        elif obj_type == "arrow":
            start = obj.get("x1"), obj.get("y1")
            end = obj.get("x2"), obj.get("y2")
            path_rows.append({
                "fra_x": start[0],
                "fra_y": start[1],
                "til_x": end[0],
                "til_y": end[1],
            })

    # Layout tabell
    layout_df = pd.DataFrame(rack_rows + start_rows + end_rows)

    # Path graf
    paths_df = pd.DataFrame(path_rows)

    return layout_df, paths_df


# ---------------------------------------------
# Generer tabeller og gi mulighet for nedlasting
# ---------------------------------------------
if export_btn:
    st.subheader("📦 Generert lagerlayout-data")

    objects = canvas_result.json_data.get("objects", [])
    layout_df, paths_df = parse_canvas_objects(objects)

    if layout_df.empty:
        st.error("Ingen objekter funnet. Tegn noe først!")
    else:
        st.success("Layout generert!")

        st.write("### Layout-tabell")
        st.dataframe(layout_df)

        st.write("### Path-graf (enveis bevegelser)")
        if paths_df.empty:
            st.info("Ingen piler tegnet.")
        else:
            st.dataframe(paths_df)

        # Nedlasting
        st.download_button(
            "⬇️ Last ned layout.csv",
            layout_df.to_csv(index=False).encode("utf-8"),
            file_name="layout.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇️ Last ned paths.csv",
            paths_df.to_csv(index=False).encode("utf-8"),
            file_name="paths.csv",
            mime="text/csv"
        )

# ---------------------------------------------
# Tips seksjon
# ---------------------------------------------
st.markdown("---")
st.markdown(
    """
### ✨ Tips
- Bruk **Reol** for lagerplasser.
- Bruk **Gangareal** for åpne områder.
- Tegn **piler** for enveiskjørte ruter.

Alle objekter blir eksportert til tabeller som passer direkte inn i SimPy-modellen.
"""
)

