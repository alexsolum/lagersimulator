import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Lagerlayout Designer", layout="wide")

st.title("📦 Lagerlayout Designer – Excel-format")

# -------------------------
# INITIAL GRID GENERATOR
# -------------------------
st.sidebar.header("Konfigurer layout-grid")

rows = st.sidebar.number_input("Antall rader", 3, 40, 8)
cols = st.sidebar.number_input("Antall kolonner", 3, 40, 12)

default_types = ["", "REOL", "GANG", "ONEWAY→", "ONEWAY←", "ONEWAY↑", "ONEWAY↓"]

# -------------------------
# CREATE GRID TABLE
# -------------------------
@st.cache_data
def create_grid(rows, cols):
    data = {f"C{c+1}": [""] * rows for c in range(cols)}
    df = pd.DataFrame(data)
    return df

if "layout_df" not in st.session_state:
    st.session_state["layout_df"] = create_grid(rows, cols)

df = st.session_state["layout_df"]

# -------------------------
# GRID EDITOR
# -------------------------
st.subheader("✏️ Rediger layout")

gb = GridOptionsBuilder.from_dataframe(df)

gb.configure_default_column(
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": default_types},
)

grid_options = gb.build()

grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.VALUE_CHANGED,
    allow_unsafe_jscode=True,
    theme="streamlit",
    enable_enterprise_modules=False,
    fit_columns_on_grid_load=True
)

df_updated = grid_response["data"]
st.session_state["layout_df"] = df_updated

# -------------------------
# EXPORT TO SIMULATION FORMAT
# -------------------------
st.subheader("📤 Eksporter layout til simulering")

if st.button("Eksporter til koordinat-format"):
    rows_list = []
    for r in range(df_updated.shape[0]):
        for c in range(df_updated.shape[1]):
            cell = df_updated.iat[r, c]
            if cell != "":
                rows_list.append({
                    "lokasjon": f"L{r+1}_{c+1}",
                    "type": cell,
                    "x": float(c),
                    "y": float(r),
                    "reolerad": r + 1,
                    "kolonne": c + 1
                })

    layout_export = pd.DataFrame(rows_list)

    st.write("### Resultatdata (klar til SimPy)")
    st.dataframe(layout_export)

    st.download_button(
        "💾 Last ned som Excel",
        data=layout_export.to_csv(index=False).encode("utf-8"),
        file_name="lagerlayout.csv",
        mime="text/csv"
    )

# -------------------------
# IMPORT FROM EXCEL
# -------------------------
st.subheader("📥 Importer lagret layout")
uploaded = st.file_uploader("Velg en layout-fil (CSV/Excel)", type=["xlsx", "csv"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        imported = pd.read_csv(uploaded)
    else:
        imported = pd.read_excel(uploaded)

    st.write("📄 Importert layout:")
    st.dataframe(imported)

    # Convert back to grid
    max_row = imported["reolerad"].max()
    max_col = imported["kolonne"].max()

    new_grid = create_grid(max_row, max_col)

    for _, row in imported.iterrows():
        new_grid.iat[row["reolerad"] - 1, row["kolonne"] - 1] = row["type"]

    st.session_state["layout_df"] = new_grid
    st.success("Layout importert!")


