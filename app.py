# app.py
import streamlit as st
import pandas as pd
from utils import generate_picker_profiles, format_time
from simulation import run_simulation
from visualization import heatmap_sections

st.title("📦 Lager-simulering (refaktorert)")

uploaded = st.file_uploader("Last opp layoutfil (xlsx)", type=["xlsx"])

if uploaded:
    df_loc = pd.read_excel(uploaded, sheet_name="lokasjoner")
    df_orders = pd.read_excel(uploaded, sheet_name="ordrer")

    pickers = st.number_input("Antall plukkere", 1, 20, 4)
    profiles = generate_picker_profiles(pickers)

    if st.button("🚀 Kjør simulering"):
        result = run_simulation(df_loc, df_orders, profiles)

        st.success("Simulering fullført!")
        st.metric("Total tid", format_time(result["total_minutes"] * 60))

        st.subheader("🔥 Heatmap")
        fig = heatmap_sections(result["layout_df"], result["heatmap"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("👣 Spor (rå data)")
        st.dataframe(result["movement_df"].head())
