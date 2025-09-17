# app/streamlit_app.py
from pathlib import Path
import sys

# --- Robust import of engine package (works on Streamlit Cloud) ---
HERE = Path(__file__).resolve()
for p in [HERE.parent, *HERE.parents]:
    if (p / "engine" / "engine.py").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError(
        "Couldn't find engine/engine.py. Ensure your repo has engine/engine.py at the root."
    )

import streamlit as st
import pandas as pd
from engine.engine import load_weather, load_lists, compute_savings

# --- Page config must be early ---
st.set_page_config(page_title="CSW Savings Calculator (Prototype)", layout="wide")

APP_DIR = HERE.parent
DATA_DIR = APP_DIR.parent / "data"

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

# ---------------- Session state ----------------
def init_state():
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "form" not in st.session_state:
        st.session_state.form = {
            # Step 1
            "state": None,
            "city": None,
            "hdd": None,
            "cdd": None,
            "elec_utility": "",
            "gas_utility": "",
            # Step 2
            "building_type": None,
            "school_level": None,  # Primary / Secondary (only for School)
            # Step 3 (varies by building)
            "area_sf": 0.0,
            "floors": 1,
            "annual_hours": 2912.0,  # only Office
            "existing_window": "Single pane",
            "hvac": "Other",
            "heating_fuel": "Natural Gas",
            "cooling_installed": "Yes",
            "mf_infiltration": "Included",  # only MF
            "hotel_occupancy": 100.0,       # only Hotel
            # Step 4
            "electric_rate": 0.12,
            "gas_rate": 1.00,
            "csw_installed_sf": 0.0,
        }

def go_to(step: int):
    st.session_state.step = max(1, min(5, int(step)))

init_state()

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

def weather_columns(df: pd.DataFrame):
    state_col = "State" if "State" in df.columns else [c for c in df.columns if "state" in c.lower()][0]
    city_col  = "Cities" if "Cities" in df.columns else [c for c in df.columns if "city" in c.lower()][0]
    hdd_col   = "Heating Degree Days (HDD)"
    cdd_col   = "Cooling Degree Days (CDD)"
    return state_col, city_col, hdd_col, cdd_col

# ---------------- UI ----------------
st.title("CSW Savings Calculator (Prototype)")

# ---- STEP 1: Location & Utilities (form) ----
if st.session_state.step == 1:
    st.header("Step 1 — Project Location & Utilities")

    with st.form("step1_form", clear_on_submit=False):
        form = st.session_state.form

        state_col, city_col, hdd_col, cdd_col = weather_columns(weather_df)

        states = sorted(weather_df[state_col].dropna().astype(str).unique().tolist())
        prev_state = form.get("state")
        s_idx = states.index(prev_state) if prev_state in states else 0
        state = st.selectbox("State", states, index=s_idx, key="s1_state")

        cities = sorted(
            weather_df.loc[weather_df[state_col]==state, city_col].dropna().astype(str).unique().tolist()
        )
        prev_city = form.get("city")
        c_idx = cities.index(prev_city) if prev_city in cities else 0
        city = st.selectbox("City", cities, index=c_idx, key="s1_city")

        # Resolve HDD/CDD
        row = weather_df[(weather_df[state_col]==state) & (weather_df[city_col]==city)].iloc[0]
        hdd, cdd = float(row[hdd_col]), float(row[cdd_col])

        col1, col2 = st.columns(2)
        with col1: st.metric("Location HDD (base 65)", f"{hdd:.0f}")
        with col2: st.metric("Location CDD (base 65)", f"{cdd:.0f}")

        # Optional utility names (these were present before)
        col3, col4 = st.columns(2)
        with col3:
            elec_util = st.text_input("Electric Utility (optional)", value=form.get("elec_utility") or "", key="s1_elec_util")
        with col4:
            gas_util = st.text_input("Natural Gas Utility (optional)", value=form.get("gas_utility") or "", key="s1_gas_util")

        nav = st.columns([1,1])
        with nav[0]:
            back = st.form_submit_button("← Back", use_container_width=True)
        with nav[1]:
            nxt = st.form_submit_button("Next →", use_container_width=True)

        # Persist after submit
        if nxt or back:
            form["state"] = state
            form["city"] = city
            form["hdd"] = hdd
            form["cdd"] = cdd
            form["elec_utility"] = elec_util
            form["gas_utility"]  = gas_util
            if nxt:
                go_to(2)
            elif back:
                go_to(1)

# ---- STEP 2: Building Type (+ school level) (form) ----
elif st.session_state.step == 2:
    st.header("Step 2 — Building Type")
    form = st.session_state.form

    with st.form("step2_form", clear_on_submit=False):
        building_types = ["Office","School","Hotel","Hospital","Multi-family"]
        prev_bt = form.get("building_type")
        bt_idx = building_types.index(prev_bt) if prev_bt in building_types else 0
        bt = st.selectbox("Select Building Type", building_types, index=bt_idx, key="s2_bt")

        # School level inline (only for School)
        if bt == "School":
            levels = ["Primary School","Secondary School"]
            prev_lvl = form.get("school_level")
            lvl_idx = levels.index(prev_lvl) if prev_lvl in levels else 0
            lvl = st.radio("School Level", levels, index=lvl_idx, horizontal=True, key="s2_school_level")
        else:
            lvl = None

        nav = st.columns([1,1])
        with nav[0]:
            back = st.form_submit_button("← Back", use_container_width=True)
        with nav[1]:
            nxt = st.form_submit_button("Next →", use_container_width=True)

        if nxt or back:
            form["building_type"] = bt
            form["school_level"]  = lvl
            if nxt:
                go_to(3)
            elif back:
                go_to(1)

# ---- STEP 3: Building Details (varies by type) (form) ----
elif st.session_state.step == 3:
    st.header("Step 3 — Building Details")
    form = st.session_state.form
    bt = form.get("building_type")

    if not bt:
        st.warning("Please choose a Building Type in Step 2.")
        if st.button("← Back to Step 2", key="s3_back_warn"):
            go_to(2)
        st.stop()

    with st.form("step3_form", clear_on_submit=False):
        # Common inputs
        col1, col2 = st.columns(2)
        with col1:
            area_sf = st.number_input("Building Area (sf)", min_value=0.0, value=float(form.get("area_sf") or 0.0), step=1000.0, format="%.0f", key="s3_area")
            floors = st.number_input("Number of Floors", min_value=1, value=int(form.get("floors") or 1), step=1, key="s3_floors")
        with col2:
            existing = st.selectbox("Type of Existing Window", ["Single pane","Double pane"],
                                    index=(["Single pane","Double pane"].index(form.get("existing_window")) if form.get("existing_window") in ["Single pane","Double pane"] else 0),
                                    key="s3_existing")
            cooling = st.selectbox("Cooling installed?", ["Yes","No"],
                                   index=(["Yes","No"].index(form.get("cooling_installed")) if form.get("cooling_installed") in ["Yes","No"] else 0),
                                   key="s3_cooling")

        # Type-specific inputs
        if bt == "Office":
            col3, col4 = st.columns(2)
            with col3:
                annual_hours = st.number_input("Annual Operating Hours", min_value=0.0, value=float(form.get("annual_hours") or 2912.0), step=100.0, format="%.0f", key="s3_hours")
                fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"],
                                    index=(["Natural Gas","Electric","None"].index(form.get("heating_fuel")) if form.get("heating_fuel") in ["Natural Gas","Electric","None"] else 0),
                                    key="s3_fuel")
            with col4:
                # Subtype preview (after inputs) and HVAC options bound to subtype
                tmp_form = form.copy()
                tmp_form.update({"area_sf": area_sf, "floors": floors, "hvac": form.get("hvac")})
                subtype_preview = derive_sub_building_type(tmp_form)
                hvac_choices = allowed_hvac_options("Office", subtype_preview or "Mid-size Office")
                hvac = st.selectbox("HVAC System Type", hvac_choices,
                                    index=(hvac_choices.index(form.get("hvac")) if form.get("hvac") in hvac_choices else 0),
                                    key="s3_hvac")

        elif bt == "School":
            col3, col4 = st.columns(2)
            with col3:
                fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"],
                                    index=(["Natural Gas","Electric","None"].index(form.get("heating_fuel")) if form.get("heating_fuel") in ["Natural Gas","Electric","None"] else 0),
                                    key="s3_fuel")
            with col4:
                subtype_preview = derive_sub_building_type({**form, "floors": floors})
                hvac_choices = allowed_hvac_options("School", subtype_preview)
                hvac = st.selectbox("HVAC System Type", hvac_choices,
                                    index=(hvac_choices.index(form.get("hvac")) if form.get("hvac") in hvac_choices else 0),
                                    key="s3_hvac")
            annual_hours = form.get("annual_hours", 0.0)  # not used for School

        elif bt == "Hotel":
            col3, col4 = st.columns(2)
            with col3:
                occ = st.number_input("Average Occupancy Rate (%)", min_value=0.0, max_value=100.0, value=float(form.get("hotel_occupancy") or 100.0), step=1.0, format="%.0f", key="s3_occ")
                fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"],
                                    index=(["Natural Gas","Electric","None"].index(form.get("heating_fuel")) if form.get("heating_fuel") in ["Natural Gas","Electric","None"] else 0),
                                    key="s3_fuel")
            with col4:
                tmp_form = form.copy()
                tmp_form.update({"hvac": form.get("hvac")})
                subtype_preview = derive_sub_building_type(tmp_form)
                hvac_choices = allowed_hvac_options("Hotel", subtype_preview or "Small Hotel")
                hvac = st.selectbox("HVAC System Type", hvac_choices,
                                    index=(hvac_choices.index(form.get("hvac")) if form.get("hvac") in hvac_choices else 0),
                                    key="s3_hvac")
            annual_hours = form.get("annual_hours", 0.0)

        elif bt == "Hospital":
            col3, col4 = st.columns(2)
            with col3:
                fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"],
                                    index=(["Natural Gas","Electric","None"].index(form.get("heating_fuel")) if form.get("heating_fuel") in ["Natural Gas","Electric","None"] else 0),
                                    key="s3_fuel")
            with col4:
                hvac_choices = allowed_hvac_options("Hospital", "Hospital")
                hvac = st.selectbox("HVAC System Type", hvac_choices,
                                    index=(hvac_choices.index(form.get("hvac")) if form.get("hvac") in hvac_choices else 0),
                                    key="s3_hvac")
            annual_hours = form.get("annual_hours", 0.0)

        else:  # Multi-family
            col3, col4 = st.columns(2)
            with col3:
                fuel = st.selectbox("Heating Fuel", ["Natural Gas","Electric","None"],
                                    index=(["Natural Gas","Electric","None"].index(form.get("heating_fuel")) if form.get("heating_fuel") in ["Natural Gas","Electric","None"] else 0),
                                    key="s3_fuel")
                infiltration = st.selectbox("Infiltration savings included?", ["Included","Excluded"],
                                            index=(["Included","Excluded"].index(form.get("mf_infiltration")) if form.get("mf_infiltration") in ["Included","Excluded"] else 0),
                                            key="s3_infil")
            with col4:
                subtype_preview = derive_sub_building_type({**form, "floors": floors})
                hvac_choices = allowed_hvac_options("Multi-family", subtype_preview)
                hvac = st.selectbox("HVAC System Type", hvac_choices,
                                    index=(hvac_choices.index(form.get("hvac")) if form.get("hvac") in hvac_choices else 0),
                                    key="s3_hvac")
            annual_hours = form.get("annual_hours", 0.0)

        # WWR (read-only) + Derived subtype after inputs
        wwr = estimate_wwr(bt, floors)
        st.caption(f"Estimated Window-to-Wall Ratio (WWR): **{wwr:.2f}**")

        sub_label = (
            "Hospital" if bt == "Hospital"
            else derive_sub_building_type({
                **form,
                "area_sf": area_sf,
                "floors": floors,
                "hvac": hvac
            })
        )
        if bt == "School":
            st.info(f"School Level: **{form.get('school_level') or 'Primary School'}**")
        else:
            st.info(f"Derived Sub-Building Type: **{sub_label}**")

        nav = st.columns([1,1])
        with nav[0]:
            back = st.form_submit_button("← Back", use_container_width=True)
        with nav[1]:
            nxt = st.form_submit_button("Next →", use_container_width=True)

        if nxt or back:
            form["area_sf"] = area_sf
            form["floors"] = floors
            form["existing_window"] = existing
            form["cooling_installed"] = cooling
            form["heating_fuel"] = fuel
            form["hvac"] = hvac
            if bt == "Hotel":
                form["hotel_occupancy"] = float(st.session_state.get("s3_occ", form.get("hotel_occupancy") or 100.0))
            if bt == "Office":
                form["annual_hours"] = float(st.session_state.get("s3_hours", form.get("annual_hours") or 2912.0))
            if bt not in ("Multi-family","Multifamily","Multi family"):
                form["mf_infiltration"] = form.get("mf_infiltration", "Included")
            if nxt:
                go_to(4)
            elif back:
                go_to(2)

# ---- STEP 4: Rates & Scope (form) ----
elif st.session_state.step == 4:
    st.header("Step 4 — Energy Rates & CSW Scope")
    form = st.session_state.form

    with st.form("step4_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            er = st.number_input("Electric Rate ($/kWh)", min_value=0.0, value=float(form.get("electric_rate") or 0.12), step=0.01, format="%.3f", key="s4_er")
        with col2:
            gr = st.number_input("Natural Gas Rate ($/therm)", min_value=0.0, value=float(form.get("gas_rate") or 1.00), step=0.05, format="%.3f", key="s4_gr")
        with col3:
            csw_sf = st.number_input("CSW Installed Area (sf)", min_value=0.0, value=float(form.get("csw_installed_sf") or 0.0), step=500.0, format="%.0f", key="s4_csw")

        nav = st.columns([1,1])
        with nav[0]:
            back = st.form_submit_button("← Back", use_container_width=True)
        with nav[1]:
            nxt = st.form_submit_button("Next →", use_container_width=True)

        if nxt or back:
            form["electric_rate"] = float(st.session_state.get("s4_er", form.get("electric_rate") or 0.12))
            form["gas_rate"]      = float(st.session_state.get("s4_gr", form.get("gas_rate") or 1.00))
            form["csw_installed_sf"] = float(st.session_state.get("s4_csw", form.get("csw_installed_sf") or 0.0))
            if nxt:
                go_to(5)
            elif back:
                go_to(3)

# ---- STEP 5: Review & Results (no form) ----
else:
    st.header("Step 5 — Review & Results")

    form = st.session_state.form
    bt = form.get("building_type")
    sub_type = derive_sub_building_type(form)

    with st.expander("Review your inputs", expanded=True):
        left, right = st.columns(2)
        with left:
            st.write("**Location**")
            st.write(f"- {form.get('city')}, {form.get('state')}")
            st.write(f"- HDD: {form.get('hdd')} | CDD: {form.get('cdd')}")
            st.write("**Utilities**")
            st.write(f"- Electric Utility: {form.get('elec_utility') or '—'}")
            st.write(f"- Natural Gas Utility: {form.get('gas_utility') or '—'}")
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

    # Compute using the internal placeholder model (same as before)
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
            csw_glazing_ui="Single",  # future hook
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
    if st.button("← Back", key="s5_back"):
        go_to(4)
