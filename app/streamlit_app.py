# app/streamlit_app.py  (TOP OF FILE)
import sys
from pathlib import Path

# Make repo root importable so `engine` works on Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from engine.engine import load_weather, load_lists, compute_savings

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"

st.set_page_config(page_title="CSW Savings Calculator (Prototype)", layout="wide")


# ---------------- Load data ----------------
weather_df = load_weather()
lists_df   = load_lists()

def load_hvac_overrides() -> pd.DataFrame:
    p = DATA_DIR / "hvac_overrides.csv"
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()
        return df
    return pd.DataFrame(columns=["Building Type","Sub-Building Type","HVAC Option"])

hvac_overrides_df = load_hvac_overrides()

# ---------------- Wizard state ----------------
if "step" not in st.session_state:
    st.session_state.step = 1

if "form" not in st.session_state:
    st.session_state.form = {
        # Step 1
        "state": None, "city": None, "hdd": None, "cdd": None,
        # Step 2
        "building_type": None, "school_level": None,
        # Step 3 (varies by building)
        "area_sf": None, "floors": None, "annual_hours": None,
        "existing_window": None, "hvac": None, "heating_fuel": None,
        "cooling_installed": "Yes", "mf_infiltration": "Included",
        "hotel_occupancy": None,
        # Step 4
        "electric_rate": None, "gas_rate": None, "csw_installed_sf": None,
    }

def next_step(): st.session_state.step = min(5, st.session_state.step + 1)
def prev_step(): st.session_state.step = max(1, st.session_state.step - 1)

# ---------------- Helpers ----------------
def estimate_wwr(building_type: str, floors: int) -> float:
    bt = (building_type or "").lower()
    f = max(1, int(floors or 1))
    base = 0.25
    if bt == "office": base = 0.30
    elif bt == "school": base = 0.20
    elif bt == "hotel": base = 0.28
    elif bt == "hospital": base = 0.18
    elif "multi" in bt: base = 0.22
    return min(0.60, round(base + 0.01 * max(0, f - 1), 3))

def derive_sub_building_type(form: dict) -> str:
    bt = form.get("building_type")
    floors = int(form.get("floors") or 0)
    area = float(form.get("area_sf") or 0)
    hvac = (form.get("hvac") or "").lower()

    if bt == "Office":
        is_large = (area > 30000) and hvac.startswith("built-up vav")
        return "Large Office" if is_large else "Mid-size Office"

    if bt == "School":
        lvl = form.get("school_level") or "Primary School"
        return "Secondary School" if str(lvl).lower().startswith("secondary") else "Primary School"

    if bt == "Hotel":
        small = (hvac.startswith("ptac") or hvac.startswith("pthp"))
        return "Small Hotel" if small else "Large Hotel"

    if bt == "Hospital":
        return "Hospital"

    if bt in ("Multi-family","Multifamily","Multi family"):
        return "Low-rise Multifamily" if floors < 4 else "Mid-rise Multifamily"

    return ""

def allowed_hvac_options(building_type: str, sub_building_type: str) -> list[str]:
    # Primary source: overrides CSV
    if not hvac_overrides_df.empty:
        df = hvac_overrides_df
        m = (df["Building Type"].str.strip().str.lower() == (building_type or "").strip().lower())
        if sub_building_type:
            m &= (df["Sub-Building Type"].str.strip().str.lower() == sub_building_type.strip().lower())
        opts = df.loc[m, "HVAC Option"].dropna().astype(str).str.strip().unique().tolist()
        if opts:
            return opts
    # Fallback: lists.csv column that matches building
    hvac_col = None
    bt = (building_type or "").lower()
    for c in lists_df.columns:
        if c.strip().lower().startswith("hvac type") and bt in c.strip().lower():
            hvac_col = c; break
    if hvac_col is not None:
        return lists_df[hvac_col].dropna().astype(str).str.strip().unique().tolist()
    return ["Other"]

# ---------------- UI ----------------
st.title("CSW Savings Calculator (Prototype)")

# STEP 1: Location
if st.session_state.step == 1:
    st.header("Step 1 — Project Location")

    state_col = "State" if "State" in weather_df.columns else [c for c in weather_df.columns if "state" in c.lower()][0]
    city_col  = "Cities" if "Cities" in weather_df.columns else [c for c in weather_df.columns if "city" in c.lower()][0]
    hdd_col   = "Heating Degree Days (HDD)"
    cdd_col   = "Cooling Degree Days (CDD)"

    states = sorted(weather_df[state_col].dropna().astype(str).unique().tolist())
    state = st.selectbox("State", states, index=states.index(st.session_state.form["state"]) if st.session_state.form["state"] in states else 0)
    st.session_state.form["state"] = state

    cities = sorted(weather_df.loc[weather_df[state_col]==state, city_col].dropna().astype(str).unique().tolist())
    city = st.selectbox("City", cities, index=cities.index(st.session_state.form["city"]) if st.session_state.form["city"] in cities else 0)
    st.session_state.form["city"] = city

    row = weather_df[(weather_df[state_col]==state) & (weather_df[city_col]==city)].iloc[0]
    hdd, cdd = float(row[hdd_col]), float(row[cdd_col])
    st.session_state.form["hdd"] = hdd
    st.session_state.form["cdd"] = cdd

    col1, col2 = st.columns(2)
    with col1: st.metric("Location HDD (base 65)", f"{hdd:.0f}")
    with col2: st.metric("Location CDD (base 65)", f"{cdd:.0f}")

    st.divider()
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Next →"):
            next_step()

# STEP 2: Building Type (and School level inline)
elif st.session_state.step == 2:
    st.header("Step 2 — Building Type")

    building_types = ["Office","School","Hotel","Hospital","Multi-family"]
    bt = st.selectbox("Select Building Type", building_types, index=(building_types.index(st.session_state.form["building_type"]) if st.session_state.form["building_type"] in building_types else 0))
    st.session_state.form["building_type"] = bt

    if bt == "School":
        levels = ["Primary School","Secondary School"]
        lvl = st.radio("School Level", levels, index=(levels.index(st.session_state.form["school_level"]) if st.session_state.form["school_level"] in levels else 0), horizontal=True)
        st.session_state.form["school_level"] = lvl

    st.divider()
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("← Back"): prev_step()
    with colB:
        if st.button("Next →"): next_step()

# STEP 3: Building Details (varies by type)
elif st.session_state.step == 3:
    form = st.session_state.form
    bt = form.get("building_type")
    if not bt:
        st.warning("Please choose a Building Type in Step 2.")
        if st.button("← Back to Step 2"): prev_step()
        st.stop()

    st.header("Step 3 — Building Details")

    # Common inputs
    col1, col2 = st.columns(2)
    with col1:
        area_sf = st.number_input("Building Area (sf)", min_value=0.0, value=float(form["area_sf"] or 0.0), step=1000.0, format="%.0f")
        form["area_sf"] = area_sf
        floors = st.number_input("Number of Floors", min_value=1, value=int(form["floors"] or 1), step=1)
        form["floors"] = floors
    with col2:
        existing = st.selectbox("Type of Existing Window", ["Single pane","Double pane"], index=(["Single pane","Double pane"].index(form["existing_window"]) if form["existing_window"] in ["Single pane","Double pane"] else 0))
        form["existing_window"] = existing
        cooling = st.selectbox("Cooling installed?", ["Yes","No"], index=(["Yes","No"].index(form["cooling_installed"]) if form["cooling_installed"] in ["Yes","No"] else 0))
        form["cooling_installed"] = cooling

    # Type-specific inputs
    if bt == "Office":
        col3, col4 = st.columns(2)
        with col3:
            annual_hours = st.number_input("Annual Operating Hours", min_value=0.0, value=float(form["annual_hours"] or 2912.0), step=100.0, format="%.0f")
            form["annual_hours"] = annual_hours
            fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"], index=(["Natural Gas","Electric","None"].index(form["heating_fuel"]) if form["heating_fuel"] in ["Natural Gas","Electric","None"] else 0))
            form["heating_fuel"] = fuel
        with col4:
            subtype_preview = derive_sub_building_type(form)
            hvac_choices = allowed_hvac_options("Office", subtype_preview)
            hvac = st.selectbox("HVAC System Type", hvac_choices, index=(hvac_choices.index(form["hvac"]) if form["hvac"] in hvac_choices else 0))
            form["hvac"] = hvac

    elif bt == "School":
        col3, col4 = st.columns(2)
        with col3:
            fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"], index=(["Natural Gas","Electric","None"].index(form["heating_fuel"]) if form["heating_fuel"] in ["Natural Gas","Electric","None"] else 0))
            form["heating_fuel"] = fuel
            subtype_preview = derive_sub_building_type(form)  # Primary/Secondary from Step 2
            hvac_choices = allowed_hvac_options("School", subtype_preview)
            hvac = st.selectbox("HVAC System Type", hvac_choices, index=(hvac_choices.index(form["hvac"]) if form["hvac"] in hvac_choices else 0))
            form["hvac"] = hvac
        with col4:
            pass

    elif bt == "Hotel":
        col3, col4 = st.columns(2)
        with col3:
            occ = st.number_input("Average Occupancy Rate (%)", min_value=0.0, max_value=100.0, value=float(form["hotel_occupancy"] or 100.0), step=1.0, format="%.0f")
            form["hotel_occupancy"] = occ
            fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"], index=(["Natural Gas","Electric","None"].index(form["heating_fuel"]) if form["heating_fuel"] in ["Natural Gas","Electric","None"] else 0))
            form["heating_fuel"] = fuel
        with col4:
            subtype_preview = derive_sub_building_type(form)  # depends on HVAC choice
            hvac_choices = allowed_hvac_options("Hotel", subtype_preview or "Small Hotel")
            hvac = st.selectbox("HVAC System Type", hvac_choices, index=(hvac_choices.index(form["hvac"]) if form["hvac"] in hvac_choices else 0))
            form["hvac"] = hvac

    elif bt == "Hospital":
        col3, col4 = st.columns(2)
        with col3:
            fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"], index=(["Natural Gas","Electric","None"].index(form["heating_fuel"]) if form["heating_fuel"] in ["Natural Gas","Electric","None"] else 0))
            form["heating_fuel"] = fuel
        with col4:
            hvac_choices = allowed_hvac_options("Hospital", "Hospital")
            hvac = st.selectbox("HVAC System Type", hvac_choices, index=(hvac_choices.index(form["hvac"]) if form["hvac"] in hvac_choices else 0))
            form["hvac"] = hvac

    elif bt in ("Multi-family","Multifamily","Multi family"):
        col3, col4 = st.columns(2)
        with col3:
            fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"], index=(["Natural Gas","Electric","None"].index(form["heating_fuel"]) if form["heating_fuel"] in ["Natural Gas","Electric","None"] else 0))
            form["heating_fuel"] = fuel
            infiltration = st.selectbox("Infiltration savings included?", ["Included","Excluded"], index=(["Included","Excluded"].index(form["mf_infiltration"]) if form["mf_infiltration"] in ["Included","Excluded"] else 0))
            form["mf_infiltration"] = infiltration
        with col4:
            subtype_preview = derive_sub_building_type(form)  # Low vs Mid by floors
            hvac_choices = allowed_hvac_options("Multi-family", subtype_preview)
            hvac = st.selectbox("HVAC System Type", hvac_choices, index=(hvac_choices.index(form["hvac"]) if form["hvac"] in hvac_choices else 0))
            form["hvac"] = hvac

    # Read-only WWR estimate
    wwr = estimate_wwr(form.get("building_type"), form.get("floors"))
    st.caption(f"Estimated Window-to-Wall Ratio (WWR): **{wwr:.2f}**")

    # Show derived sub-type AFTER inputs (where applicable)
    if bt in ("Office","Hotel","Multi-family"):
        sub_label = derive_sub_building_type(form)
        if sub_label:
            st.info(f"Derived Sub-Building Type: **{sub_label}**")
    elif bt == "Hospital":
        st.info("Sub-Building Type: **Hospital**")
    elif bt == "School":
        st.info(f"School Level: **{form.get('school_level') or 'Primary School'}**")

    st.divider()
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("← Back"): prev_step()
    with colB:
        if st.button("Next →"): next_step()

# STEP 4: Rates & Scope
elif st.session_state.step == 4:
    st.header("Step 4 — Energy Rates & CSW Scope")
    form = st.session_state.form

    col1, col2, col3 = st.columns(3)
    with col1:
        er = st.number_input("Electric Rate ($/kWh)", min_value=0.0, value=float(form["electric_rate"] or 0.12), step=0.01, format="%.3f")
        form["electric_rate"] = er
    with col2:
        gr = st.number_input("Natural Gas Rate ($/therm)", min_value=0.0, value=float(form["gas_rate"] or 1.00), step=0.05, format="%.3f")
        form["gas_rate"] = gr
    with col3:
        csw_sf = st.number_input("CSW Installed Area (sf)", min_value=0.0, value=float(form["csw_installed_sf"] or 0.0), step=500.0, format="%.0f")
        form["csw_installed_sf"] = csw_sf

    st.divider()
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("← Back"): prev_step()
    with colB:
        if st.button("Next →"): next_step()

# STEP 5: Review & Results
else:
    st.header("Step 5 — Review & Results")

    form = st.session_state.form
    bt = form.get("building_type")
    sub_type = derive_sub_building_type(form)

    # Review summary
    with st.expander("Review your inputs", expanded=True):
        left, right = st.columns(2)
        with left:
            st.write("**Location**")
            st.write(f"- {form.get('city')}, {form.get('state')}")
            st.write(f"- HDD: {form.get('hdd')} | CDD: {form.get('cdd')}")
            st.write("**Building**")
            st.write(f"- Type: {bt}")
            if bt == "School":
                st.write(f"- School Level: {form.get('school_level')}")
            st.write(f"- Area: {form.get('area_sf')} sf  |  Floors: {form.get('floors')}")
            if bt == "Office":
                st.write(f"- Annual Hours: {form.get('annual_hours')}")
            if bt == "Hotel":
                st.write(f"- Occupancy: {form.get('hotel_occupancy')}%")
        with right:
            st.write("**Envelope & Systems**")
            st.write(f"- Existing Window: {form.get('existing_window')}")
            st.write(f"- HVAC System: {form.get('hvac')}")
            st.write(f"- Heating Fuel: {form.get('heating_fuel')}")
            st.write(f"- Cooling Installed: {form.get('cooling_installed')}")
            if "multi" in (bt or "").lower():
                st.write(f"- Infiltration Savings: {form.get('mf_infiltration')}")
            st.write("**Rates & Scope**")
            st.write(f"- Elec Rate: ${form.get('electric_rate')} / kWh")
            st.write(f"- Gas Rate: ${form.get('gas_rate')} / therm")
            st.write(f"- CSW Installed Area: {form.get('csw_installed_sf')} sf")
            st.write(f"**Derived Sub-Building Type:** {sub_type or '(n/a)'}")

    # Compute (simple internal model)
    try:
        res = compute_savings(
            weather_hdd=float(form.get("hdd") or 0.0),
            weather_cdd=float(form.get("cdd") or 0.0),
            building_type=bt,
            sub_building_type=sub_type,
            hvac_ui=form.get("hvac"),
            heating_fuel_ui=form.get("heating_fuel"),
            cooling_installed=(form.get("cooling_installed") == "Yes"),
            existing_window_type_ui=form.get("existing_window"),
            csw_glazing_ui="Single",  # can be wired later
            building_area_sf=float(form.get("area_sf") or 0.0),
            annual_operating_hours=float(form.get("annual_hours") or 0.0),
            hotel_occupancy_pct=float(form.get("hotel_occupancy") or 100.0),
            include_infiltration=(form.get("mf_infiltration") != "Excluded"),
            floors=int(form.get("floors") or 0),
            csw_installed_sf=float(form.get("csw_installed_sf") or 0.0),
            electric_rate=float(form.get("electric_rate") or 0.0),
            gas_rate=float(form.get("gas_rate") or 0.0),
        )

        k1, k2, k3 = st.columns(3)
        with k1: st.metric("Electric Savings (kWh/yr)", f"{res['total_kwh']:,.0f}")
        with k2: st.metric("Gas Savings (therms/yr)", f"{res['total_therms']:,.0f}")
        with k3: st.metric("Estimated Cost Savings ($/yr)", f"${res['cost_savings_usd']:,.0f}")

        st.subheader("Per-SF Savings (applied to CSW installed area)")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Heating (kWh/sf)", f"{res['elec_heat_kwh_per_sf']:.3f}")
        with c2: st.metric("Cooling & Aux (kWh/sf)", f"{res['cool_kwh_per_sf']:.3f}")
        with c3: st.metric("Heating (therms/sf)", f"{res['gas_heat_therm_per_sf']:.4f}")

        st.subheader("Modeled EUI (indicative only)")
        d1, d2, d3 = st.columns(3)
        with d1: st.metric("Base EUI (kBtu/sf/yr)", f"{(res.get('eui_base') or 0):.1f}")
        with d2: st.metric("CSW EUI (kBtu/sf/yr)", f"{(res.get('eui_csw') or 0):.1f}")
        with d3: st.metric("EUI Savings (kBtu/sf/yr)", f"{(res.get('eui_savings_kbtusf') or 0):.1f}")

        st.caption("Results are preliminary estimates; a full energy model is recommended for more accurate results.")

    except Exception as e:
        st.error(f"Could not compute savings for the selected combination: {e}")

    st.divider()
    if st.button("← Back"): prev_step()
