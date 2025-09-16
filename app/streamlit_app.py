import streamlit as st
import pandas as pd
from pathlib import Path
import sys, os

# >>> make sure repo root is on sys.path (…/app -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
assert (REPO_ROOT / "engine" / "engine.py").exists(), f"engine.py not found at {REPO_ROOT/'engine'/'engine.py'}"

from engine.engine import load_weather, load_lists, load_savings_lookup, compute_hdd_cdd

st.set_page_config(page_title="CSW Savings Calculator (Prototype)", layout="wide")

st.title("Commercial Secondary Windows — Savings Calculator (Prototype)")

st.markdown("> **Preliminary estimates only.** For detailed results, a full energy model is recommended.")

# Load data
weather_df = load_weather()
lists_df = load_lists()
lookup_df = load_savings_lookup()

# --- Step 1: Project & Location ---
st.header("1) Project & Location")

def find_col(df, *candidates):
    cols = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # fuzzy contains
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

# Load weather (already done earlier via engine.load_weather)
wdf = weather_df

# Try flexible header matches
state_col = find_col(wdf, "State")
city_col  = find_col(wdf, "City")
hdd_col   = find_col(wdf, "HDD", "Heating Degree Days", "Heating Degree Days (HDD)")
cdd_col   = find_col(wdf, "CDD", "Cooling Degree Days", "Cooling Degree Days (CDD)")

# Debug helper (optional): show what we resolved
# st.caption(f"Resolved columns → State: {state_col}, City: {city_col}, HDD: {hdd_col}, CDD: {cdd_col}")

# Guardrails
missing = [name for name, col in [("State", state_col), ("City", city_col),
                                  ("HDD", hdd_col), ("CDD", cdd_col)] if col is None]
if missing:
    st.error(f"weather_information.csv is missing these columns (or headers are unexpected): {', '.join(missing)}")
    st.stop()

# Build selectors
states = sorted(wdf[state_col].dropna().astype(str).unique().tolist())
state = st.selectbox("State", states)

cities = sorted(
    wdf[wdf[state_col].astype(str) == state][city_col]
    .dropna().astype(str).unique().tolist()
)
city = st.selectbox("City", cities)

# Lookup HDD/CDD
sel = wdf[(wdf[state_col].astype(str) == state) & (wdf[city_col].astype(str) == city)]
if sel.empty:
    st.error("Selected State/City not found in weather_information.csv.")
    st.stop()

hdd = float(sel[hdd_col].iloc[0])
cdd = float(sel[cdd_col].iloc[0])

col1, col2 = st.columns(2)
with col1:
    st.metric("Location HDD (base 65)", f"{hdd:.0f}")
with col2:
    st.metric("Location CDD (base 65)", f"{cdd:.0f}")

elec_utility = st.text_input("Electric Utility (optional)")
gas_utility  = st.text_input("Natural Gas Utility (optional)")

# HDD/CDD auto
hdd, cdd = compute_hdd_cdd(weather_df, state, city)
col1, col2 = st.columns(2)
with col1:
    st.metric("Location HDD (base 65)", f"{hdd:.0f}" if hdd is not None else "—")
with col2:
    st.metric("Location CDD (base 65)", f"{cdd:.0f}" if cdd is not None else "—")

# --- Step 2: Building ---
st.header("2) Building")
building_type = st.selectbox("Building Type", ["Office","Hotel","School","Hospital","Multi-family"])
area = st.number_input("Building Area (ft²)", min_value=0.0, value=100000.0, step=1000.0)
floors = st.number_input("Number of Floors", min_value=1, value=3, step=1)
hvac_type = st.selectbox("HVAC System Type", ["VAV","Packaged RTU","Heat Pump","Boiler/Chiller"])
heating_fuel = st.selectbox("Heating Fuel", ["Elec","Gas"])
cooling_installed = st.selectbox("Cooling Installed?", ["Yes","No"])
annual_hours = st.number_input("Annual Operating Hours", min_value=0, max_value=8760, value=4000, step=100)

existing_window = st.selectbox("Type of Existing Window", ["Single pane","Double pane","New double pane (U<0.35)"])

# --- Step 3: Proposed CSW ---
st.header("3) Proposed CSW")
csw_type = st.selectbox("Type of CSW Analyzed", ["Single","Double"])
csw_area = st.number_input("Sq. Ft. of CSW Installed", min_value=0.0, value=10000.0, step=500.0)

# --- Step 4: Rates ---
st.header("4) Rates")
elec_rate = st.number_input("Electric Rate ($/kWh)", min_value=0.0, value=0.14, step=0.01, format="%.4f")
gas_rate = st.number_input("Natural Gas Rate ($/therm)", min_value=0.0, value=0.76, step=0.01, format="%.4f")

# --- Auto-computed WWR (mirrors Excel formula: =F27/((F18/F19)^0.5*4*15*F19)) ---
# Map: F27 -> csw_area, F18 -> Building Area, F19 -> Floors
import math
den = ( (area / max(floors,1)) ** 0.5 ) * 4 * 15 * max(floors,1)
wwr = csw_area / den if den > 0 else 0.0

warn = ""
if wwr > 0.5:
    warn = "⚠️ **WWR is over 0.50** — check CSW area vs. building geometry."
st.subheader("Derived Inputs")
st.write(f"Estimated Window-to-Wall Ratio (WWR): **{wwr:.3f}**  {warn}")

# --- Placeholder Results (wire engine next) ---
st.header("5) Results (Preview)")
st.info("Regression-driven energy savings will appear here once the coefficients and VLOOKUP logic are wired in.")

st.write("**Next:** tie regression tables from 'Savings Lookup' + 'Regression List_*' sheets to compute kWh/therms/$ savings exactly like Excel.")
