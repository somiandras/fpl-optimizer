import pandas as pd
import streamlit as st

from preprocess import get_player_data
from optimize import POSITIONS, optimize

st.markdown("## Parameters")
ft = st.number_input("Free Transfers", min_value=0, max_value=15, value=1)
itb = st.number_input("ITB", value=20, min_value=0, max_value=1000)
filler_count = st.number_input("Filler count", min_value=0, max_value=4, value=4)
margin = st.number_input("Transfer Margin", value=0, min_value=0, max_value=10)
at_gw = st.number_input("GW", value=6)

df = get_player_data(my_team_path="sample_data/team_data.json", at_gw=at_gw)

st.markdown("## Current Squad")

my_team = (
    df.query("in_squad")
    .astype({"position": pd.CategoricalDtype(POSITIONS, ordered=True)})
    .sort_values("position")
)

st.dataframe(my_team)

st.markdown("## Optimal Squad")
results = optimize(df, itb=itb, ft=ft, margin=margin, filler_count=filler_count)

st.markdown(f"__Expected points in 5GW__: {results.expected_team_points}")
st.markdown(f"__Formation__: {results.formation}")
st.markdown(f"{results.transfers} transfers for {results.transfer_cost} points.")
st.markdown(f"__Team value__: {results.value}")
st.markdown(f"__ITB after__: {results.itb_change + itb}")

st.dataframe(
    results.new_squad.astype(
        {"position": pd.CategoricalDtype(POSITIONS, ordered=True)}
    ).sort_values("position")
)
