# app/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import re

# ============== Setup & Imports ==============
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from engine.engine import load_weather, load_lists, load_savings_lookup
    ENGINE_OK = True
except Exception:
    ENGINE_OK = False

st.set_page_config(page_title="CSW Savings Calculator (Prototype)", layout="wide")
st.title("Commercial Secondary Windows — Savings Calculator (Prototype)")
st.markdown("> **Preliminary estimates only.** For detailed results, a full energy model is recommended.")

DATA_DIR = REPO_ROOT / "data"

# ============== Data Loaders ==============
@st.cache_data(show_spinner=False)
def _load_weather_fallback():
    p = DATA_DIR / "weather_information.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def _load_lists_fallback():
    p = DATA_DIR / "lists.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def _load_lookup_fallback():
    p = DATA_DIR / "savings_lookup.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def load_hvac_overrides():
    p = DATA_DIR / "hvac_overrides.csv"
    if not p.exists():
        return pd.DataFrame(columns=["Building Type", "Sub-Building Type", "HVAC Option"])
    df = pd.read_csv(p)
    for c in ["Building Type", "Sub-Building Type", "HVAC Option"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

# Load frames (engine if available, else fallback)
try:
    weather_df = load_weather() if ENGINE_OK else _load_weather_fallback()
except Exception as e:
    st.error(f"Could not load weather data.\n\n{e}")
    st.stop()

try:
    lists_df = load_lists() if ENGINE_OK else _load_lists_fallback()
except Exception as e:
    st.warning(f"Lists data not loaded: {e}")
    lists_df = pd.DataFrame()

try:
    lookup_df = load_savings_lookup() if ENGINE_OK else _load_lookup_fallback()
except Exception as e:
    st.warning(f"Savings lookup not loaded: {e}")
    lookup_df = pd.DataFrame()

overrides_df = load_hvac_overrides()

# ============== Helpers ==============
def _to_num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None

def _clean_options(series):
    if series is None or series.empty:
        return []
    s = series.dropna().astype(str).map(lambda x: x.strip())
    opts = [x for x in s if x and x.lower() not in ("none", "nan")]
    seen, ordered = set(), []
    for x in opts:
        if x not in seen:
            ordered.append(x); seen.add(x)
    return ordered

def hvac_options_for_building(lists_df: pd.DataFrame, building_label: str):
    if lists_df is None or lists_df.empty or not building_label:
        return []
    tokens_map = {
        "Office": ["office"],
        "Hotel": ["hotel"],
        "School": ["school", "sh"],
        "Hospital": ["hospital", "hosp"],
        "Multi-family": ["multi-family", "multifamily", "mf", "m-f"],
    }
    tokens = tokens_map.get(building_label, [building_label.lower()])
    # exact "HVAC Type, <token>"
    for c in lists_df.columns:
        c_norm = str(c).strip()
        if c_norm.lower().startswith("hvac type"):
            if any(re.fullmatch(rf"(?i)hvac type[, ]+\s*{re.escape(t)}\s*", c_norm) for t in tokens):
                return _clean_options(lists_df[c])
    # contains fallback
    for c in lists_df.columns:
        c_norm = str(c).strip().lower()
        if c_norm.startswith("hvac type") and any(t in c_norm for t in tokens):
            return _clean_options(lists_df[c])
    return []

def hvac_options_from_overrides(overrides_df: pd.DataFrame, building_type: str, sub_building_type: str, lists_df: pd.DataFrame):
    if overrides_df is not None and not overrides_df.empty and sub_building_type:
        mask = (
            overrides_df["Building Type"].str.casefold() == str(building_type).casefold()
        ) & (
            overrides_df["Sub-Building Type"].str.casefold() == str(sub_building_type).casefold()
        )
        rows = overrides_df.loc[mask, "HVAC Option"]
        opts = [o for o in rows.dropna().astype(str).map(str.strip).tolist() if o and o.lower() not in ("none","nan")]
        seen, cleaned = set(), []
        for o in opts:
            if o not in seen:
                cleaned.append(o); seen.add(o)
        if cleaned:
            if not any(o.lower() == "other" for o in cleaned):
                cleaned.append("Other")
            return cleaned

    fallback = hvac_options_for_building(lists_df, building_type) or []
    if not any(o.lower() == "other" for o in fallback):
        fallback.append("Other")
    return fallback

def compute_sub_building(building_type: str, area_ft2: float, floors: int, hvac_type: str, school_subtype: str | None):
    bt = (building_type or "").strip()

    if bt == "Office":
        hvac_norm = (hvac_type or "").strip().lower()
        is_built_up = hvac_norm == "built-up vav with hydronic reheat"
        if (area_ft2 or 0) > 30000 and is_built_up:
            return "Large Office"
        return "Mid-size Office"

    if bt == "Hotel":
        hv = (hvac_type or "").strip().upper()
        return "Small Hotel" if hv in {"PTAC", "PTHP"} else "Large Hotel"

    if bt == "School":
        return (school_subtype or "").strip() or "Secondary School"

    if bt == "Multi-family":
        return "Low-rise Multifamily" if (floors or 0) < 4 else "Mid-rise Multifamily"

    if bt == "Hospital":
        return "Hospital"

    return bt

def compute_wwr(area_ft2: float, floors: int, csw_area_ft2: float, wall_height_ft: float = 15.0) -> float:
    try:
        area = float(area_ft2)
        flrs = int(floors)
        csw = float(csw_area_ft2)
        h = float(wall_height_ft)
    except Exception:
        return 0.0
    if area <= 0 or flrs <= 0 or h <= 0:
        return 0.0
    side = (area / flrs) ** 0.5
    perimeter = side * 4
    wall_area = perimeter * h * flrs
    if wall_area <= 0:
        return 0.0
    return csw / wall_area

# ============== Wizard State ==============
TOTAL_STEPS = 6
if "step" not in st.session_state:
    st.session_state.step = 1

def go_next():
    st.session_state.step = min(TOTAL_STEPS, st.session_state.step + 1)

def go_back():
    st.session_state.step = max(1, st.session_state.step - 1)

def progress_bar():
    st.progress(st.session_state.step / TOTAL_STEPS)

# ---------- Sticky restore helpers ----------
def _restore_sticky(cur_key: str, saved_key: str):
    """If current key is empty but saved exists, restore."""
    cur = st.session_state.get(cur_key)
    if (cur is None or cur == "") and st.session_state.get(saved_key):
        st.session_state[cur_key] = st.session_state[saved_key]
    return st.session_state.get(cur_key)

# ---------- Prereq guards (with restore first) ----------
def require_location_before(step_num: int):
    if step_num >= 2:
        _restore_sticky("state", "state_saved")
        _restore_sticky("city", "city_saved")
        _restore_sticky("hdd65", "hdd65_saved")
        _restore_sticky("cdd65", "cdd65_saved")
        if not st.session_state.get("state") or not st.session_state.get("city"):
            st.session_state.step = 1

def require_building_before(step_num: int):
    if step_num >= 3:
        _restore_sticky("building_type", "building_type_saved")
        _restore_sticky("school_subtype", "school_subtype_saved")
        if not st.session_state.get("building_type"):
            st.session_state.step = 2

# Run guards early
require_location_before(st.session_state.step)
require_building_before(st.session_state.step)

# ============== Step 1: Project & Location ==============
if st.session_state.step == 1:
    st.header("Step 1 — Project & Location")

    REQ = ["State", "Cities", "Heating Degree Days (HDD)", "Cooling Degree Days (CDD)"]
    missing = [c for c in REQ if c not in weather_df.columns]
    if missing:
        st.error(
            "Your 'data/weather_information.csv' must contain these columns:\n\n"
            f"- {', '.join(REQ)}\n\n"
            f"Missing: {', '.join(missing)}"
        )
        st.stop()

    wdf = weather_df[REQ].copy().dropna(subset=["State", "Cities"])
    wdf["State"] = wdf["State"].astype(str).str.strip()
    wdf["Cities"] = wdf["Cities"].astype(str).str.strip()

    states = sorted(wdf["State"].unique().tolist())
    st.selectbox("State", states, key="state")
    # persist immediately
    st.session_state["state_saved"] = st.session_state.get("state")

    cities = sorted(wdf.loc[wdf["State"] == st.session_state.get("state"), "Cities"].unique().tolist())
    st.selectbox("City", cities, key="city")
    st.session_state["city_saved"] = st.session_state.get("city")

    sel = wdf[(wdf["State"] == st.session_state.get("state")) & (wdf["Cities"] == st.session_state.get("city"))]
    if sel.empty:
        st.error("Selected State/City not found in weather_information.csv.")
        st.stop()

    hdd = _to_num(sel["Heating Degree Days (HDD)"].iloc[0])
    cdd = _to_num(sel["Cooling Degree Days (CDD)"].iloc[0])
    if hdd is None or cdd is None:
        st.error("HDD/CDD values could not be parsed as numbers.")
        st.stop()

    st.session_state["hdd65"] = hdd
    st.session_state["cdd65"] = cdd
    # persist immediately
    st.session_state["hdd65_saved"] = hdd
    st.session_state["cdd65_saved"] = cdd

    c1, c2 = st.columns(2)
    with c1: st.metric("Location HDD (base 65)", f"{hdd:.0f}")
    with c2: st.metric("Location CDD (base 65)", f"{cdd:.0f}")

    st.text_input("Electric Utility (optional)", key="elec_utility")
    st.text_input("Natural Gas Utility (optional)", key="gas_utility")

    progress_bar()
    st.button("Next →", on_click=go_next, type="primary")

# ============== Step 2: Building Type ONLY (School subtype here) ==============
elif st.session_state.step == 2:
    st.header("Step 2 — Building Type")

    st.selectbox("Building Type", ["Office", "Hotel", "School", "Hospital", "Multi-family"], key="building_type")
    if st.session_state.get("building_type"):
        st.session_state["building_type_saved"] = st.session_state["building_type"]

    if st.session_state.get("building_type") == "School":
        st.selectbox("School Type", ["Primary School", "Secondary School"], index=1, key="school_subtype")
        if st.session_state.get("school_subtype"):
            st.session_state["school_subtype_saved"] = st.session_state["school_subtype"]

    progress_bar()
    col1, col2 = st.columns(2)
    with col1: st.button("← Back", on_click=go_back)
    with col2: st.button("Next →", on_click=go_next, type="primary")

# ============== Step 3: Building Details (varies by building type) ==============
elif st.session_state.step == 3:
    st.header("Step 3 — Building Details")

    building_type = (st.session_state.get("building_type_saved") or st.session_state.get("building_type") or "").strip()
    if not building_type:
        st.session_state.step = 2
        st.experimental_rerun()

    school_subtype_val = st.session_state.get("school_subtype_saved") or st.session_state.get("school_subtype")

    def heating_fuel_select():
        st.selectbox("Heating Fuel", ["Natural gas", "Electric", "None"], key="heating_fuel")

    def existing_window_select():
        st.selectbox("Type of Existing Window", ["Single pane", "Double pane"], key="existing_window")

    def cooling_installed_select():
        st.selectbox("Cooling Installed?", ["Yes", "No"], key="cooling_installed")

    def hvac_options_for_building_local(bt):
        if not bt:
            return []
        return hvac_options_for_building(lists_df, bt)

    def hvac_select_building_wide(bt):
        options = hvac_options_for_building_local(bt)
        idx = options.index(st.session_state["hvac_type"]) if st.session_state.get("hvac_type") in options else (0 if options else None)
        st.selectbox("HVAC System Type", options, key="hvac_type", index=idx if idx is not None else 0)

    def hvac_select_with_overrides(bt, sub_building_type: str):
        options = hvac_options_from_overrides(overrides_df, bt, sub_building_type, lists_df)
        idx = options.index(st.session_state["hvac_type"]) if st.session_state.get("hvac_type") in options else (0 if options else None)
        st.selectbox("HVAC System Type", options, key="hvac_type", index=idx if idx is not None else 0, help=f"Options for **{bt} → {sub_building_type}**")

    if building_type == "Office":
        st.number_input("Building Area (ft²)", min_value=0.0, value=float(st.session_state.get("area_ft2", 100000.0)), step=1000.0, format="%.0f", key="area_ft2")
        st.number_input("Number of Floors", min_value=1, value=int(st.session_state.get("floors", 3)), step=1, key="floors")
        hvac_select_building_wide("Office")
        heating_fuel_select()
        st.number_input("Annual Operating Hours", min_value=0, max_value=8760, value=int(st.session_state.get("annual_hours", 4000)), step=100, key="annual_hours")
        existing_window_select()
        cooling_installed_select()

        sub_build = compute_sub_building("Office", st.session_state.get("area_ft2", 0.0), st.session_state.get("floors", 1), st.session_state.get("hvac_type", ""), None)
        st.session_state["sub_building_type"] = sub_build
        st.info(f"**Sub-Building Type:** {sub_build}")

    elif building_type == "School":
        st.number_input("Building Area (ft²)", min_value=0.0, value=float(st.session_state.get("area_ft2", 80000.0)), step=1000.0, format="%.0f", key="area_ft2")
        st.number_input("Number of Floors", min_value=1, value=int(st.session_state.get("floors", 2)), step=1, key="floors")
        existing_window_select()
        sub_build_known = compute_sub_building("School", st.session_state.get("area_ft2", 0.0), st.session_state.get("floors", 1), st.session_state.get("hvac_type", ""), school_subtype_val)
        hvac_select_with_overrides("School", sub_build_known)
        cooling_installed_select()
        heating_fuel_select()
        st.session_state["sub_building_type"] = sub_build_known
        st.info(f"**Sub-Building Type:** {sub_build_known}")

    elif building_type == "Hotel":
        st.number_input("Building Area (ft²)", min_value=0.0, value=float(st.session_state.get("area_ft2", 120000.0)), step=1000.0, format="%.0f", key="area_ft2")
        st.number_input("Number of Floors", min_value=1, value=int(st.session_state.get("floors", 4)), step=1, key="floors")
        hvac_select_building_wide("Hotel")
        st.number_input("Average Occupancy Rate (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.get("hotel_occupancy_pct", 70.0)), step=1.0, key="hotel_occupancy_pct")
        heating_fuel_select()
        cooling_installed_select()
        existing_window_select()
        sub_build = compute_sub_building("Hotel", st.session_state.get("area_ft2", 0.0), st.session_state.get("floors", 1), st.session_state.get("hvac_type", ""), None)
        st.session_state["sub_building_type"] = sub_build
        st.info(f"**Sub-Building Type:** {sub_build}")

    elif building_type == "Hospital":
        st.number_input("Building Area (ft²)", min_value=0.0, value=float(st.session_state.get("area_ft2", 200000.0)), step=1000.0, format="%.0f", key="area_ft2")
        st.number_input("Number of Floors", min_value=1, value=int(st.session_state.get("floors", 5)), step=1, key="floors")
        sub_build_fixed = compute_sub_building("Hospital", 0, 0, "", None)
        hvac_select_with_overrides("Hospital", sub_build_fixed)
        heating_fuel_select()
        cooling_installed_select()
        existing_window_select()
        st.session_state["sub_building_type"] = sub_build_fixed
        st.info(f"**Sub-Building Type:** {sub_build_fixed}")

    elif building_type == "Multi-family":
        st.number_input("Building Area (ft²)", min_value=0.0, value=float(st.session_state.get("area_ft2", 90000.0)), step=1000.0, format="%.0f", key="area_ft2")
        st.number_input("Number of Floors", min_value=1, value=int(st.session_state.get("floors", 3)), step=1, key="floors")
        sub_build_mf = compute_sub_building("Multi-family", st.session_state.get("area_ft2", 0.0), st.session_state.get("floors", 1), "", None)
        hvac_select_with_overrides("Multi-family", sub_build_mf)
        heating_fuel_select()
        cooling_installed_select()
        st.checkbox("Include infiltration savings", value=bool(st.session_state.get("mf_include_infiltration", True)), key="mf_include_infiltration")
        existing_window_select()
        st.session_state["sub_building_type"] = sub_build_mf
        st.info(f"**Sub-Building Type:** {sub_build_mf}")

    else:
        st.session_state.step = 2
        st.experimental_rerun()

    progress_bar()
    col1, col2 = st.columns(2)
    with col1: st.button("← Back", on_click=go_back)
    with col2: st.button("Next →", on_click=go_next, type="primary")

# ============== Step 4: Proposed CSW ==============
elif st.session_state.step == 4:
    st.header("Step 4 — Proposed CSW")

    st.selectbox("Type of CSW Analyzed", ["Single", "Double"], key="csw_type")
    st.number_input("Sq. Ft. of CSW Installed", min_value=0.0, value=float(st.session_state.get("csw_area_ft2", 10000.0)), step=500.0, format="%.0f", key="csw_area_ft2")

    area = st.session_state.get("area_ft2", 0.0)
    floors = st.session_state.get("floors", 1)
    csw_area = st.session_state.get("csw_area_ft2", 0.0)
    wwr = compute_wwr(area, floors, csw_area, wall_height_ft=15.0)
    st.session_state["wwr"] = wwr

    warn = " ⚠️ **WWR is over 0.50** — check CSW area vs. building geometry." if wwr > 0.50 else ""
    st.write(f"Estimated Window-to-Wall Ratio (WWR): **{wwr:.3f}** ({wwr*100:.1f}%)" + warn)

    progress_bar()
    col1, col2 = st.columns(2)
    with col1: st.button("← Back", on_click=go_back)
    with col2: st.button("Next →", on_click=go_next, type="primary")

# ============== Step 5: Rates ==============
elif st.session_state.step == 5:
    st.header("Step 5 — Utility Rates")

    st.number_input("Electric Rate ($/kWh)", min_value=0.0, value=float(st.session_state.get("elec_rate", 0.14)), step=0.01, format="%.4f", key="elec_rate")
    st.number_input("Natural Gas Rate ($/therm)", min_value=0.0, value=float(st.session_state.get("gas_rate", 0.76)), step=0.01, format="%.4f", key="gas_rate")

    progress_bar()
    col1, col2 = st.columns(2)
    with col1: st.button("← Back", on_click=go_back)
    with col2: st.button("Next →", on_click=go_next, type="primary")

# ============== Step 6: Review & Results (placeholder) ==============
elif st.session_state.step == 6:
    st.header("Step 6 — Review & Results")

    wwr = compute_wwr(st.session_state.get("area_ft2", 0.0), st.session_state.get("floors", 1), st.session_state.get("csw_area_ft2", 0.0), wall_height_ft=15.0)
    st.session_state["wwr"] = wwr

    summary = {
        "State": st.session_state.get("state_saved") or st.session_state.get("state"),
        "City": st.session_state.get("city_saved") or st.session_state.get("city"),
        "HDD65": st.session_state.get("hdd65_saved") or st.session_state.get("hdd65"),
        "CDD65": st.session_state.get("cdd65_saved") or st.session_state.get("cdd65"),
        "Building Type": st.session_state.get("building_type_saved") or st.session_state.get("building_type"),
        "Sub-Building Type": st.session_state.get("sub_building_type"),
        "Area (ft²)": st.session_state.get("area_ft2"),
        "Floors": st.session_state.get("floors"),
        "HVAC": st.session_state.get("hvac_type"),
        "Heating Fuel": st.session_state.get("heating_fuel"),
        "Cooling Installed": st.session_state.get("cooling_installed"),
        "Existing Window": st.session_state.get("existing_window"),
        "CSW Type": st.session_state.get("csw_type"),
        "CSW Area (ft²)": st.session_state.get("csw_area_ft2"),
        "WWR (calc)": round(wwr, 3),
        "WWR (%)": f"{wwr*100:.1f}%",
        "Elec Rate ($/kWh)": st.session_state.get("elec_rate"),
        "Gas Rate ($/therm)": st.session_state.get("gas_rate"),
    }
    if (st.session_state.get("building_type_saved") or st.session_state.get("building_type")) == "Office":
        summary["Annual Hours"] = st.session_state.get("annual_hours")
    if (st.session_state.get("building_type_saved") or st.session_state.get("building_type")) == "Hotel":
        summary["Occupancy Rate (%)"] = st.session_state.get("hotel_occupancy_pct")
    if (st.session_state.get("building_type_saved") or st.session_state.get("building_type")) == "School":
        summary["School Type"] = st.session_state.get("school_subtype_saved") or st.session_state.get("school_subtype")
    if (st.session_state.get("building_type_saved") or st.session_state.get("building_type")) == "Multi-family":
        summary["Infiltration Savings?"] = st.session_state.get("mf_include_infiltration")

    st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "Value"}))

    st.subheader("Results (Preview)")
    st.info("Regression-driven energy savings will appear here once we wire the Office regression + Savings Lookup logic.")
    cA, cB, cC = st.columns(3)
    with cA: st.metric("Annual Electric Savings (kWh/yr)", "—")
    with cB: st.metric("Annual Natural Gas Savings (therms/yr)", "—")
    with cC: st.metric("Annual Energy Cost Savings ($/yr)", "—")

    st.subheader("Contact (Lead Capture)")
    with st.form("lead_form"):
        lc_name = st.text_input("Your Name")
        lc_company = st.text_input("Company")
        lc_email = st.text_input("Email")
        lc_phone = st.text_input("Phone")
        consent = st.checkbox("I agree to be contacted about my project.")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if not consent:
            st.warning("Please check the consent box so we can contact you.")
        else:
            lead = {
                "name": lc_name,
                "company": lc_company,
                "email": lc_email,
                "phone": lc_phone,
                "state": st.session_state.get("state_saved") or st.session_state.get("state"),
                "city": st.session_state.get("city_saved") or st.session_state.get("city"),
                "hdd65": st.session_state.get("hdd65_saved") or st.session_state.get("hdd65"),
                "cdd65": st.session_state.get("cdd65_saved") or st.session_state.get("cdd65"),
                "building_type": st.session_state.get("building_type_saved") or st.session_state.get("building_type"),
                "sub_building_type": st.session_state.get("sub_building_type"),
                "area_ft2": st.session_state.get("area_ft2"),
                "floors": st.session_state.get("floors"),
                "hvac_type": st.session_state.get("hvac_type"),
                "heating_fuel": st.session_state.get("heating_fuel"),
                "cooling_installed": st.session_state.get("cooling_installed"),
                "existing_window": st.session_state.get("existing_window"),
                "annual_hours": st.session_state.get("annual_hours"),
                "hotel_occupancy_pct": st.session_state.get("hotel_occupancy_pct"),
                "school_subtype": st.session_state.get("school_subtype_saved") or st.session_state.get("school_subtype"),
                "mf_include_infiltration": st.session_state.get("mf_include_infiltration"),
                "csw_type": st.session_state.get("csw_type"),
                "csw_area_ft2": st.session_state.get("csw_area_ft2"),
                "wwr": st.session_state.get("wwr"),
                "elec_rate": st.session_state.get("elec_rate"),
                "gas_rate": st.session_state.get("gas_rate"),
                "elec_utility": st.session_state.get("elec_utility"),
                "gas_utility": st.session_state.get("gas_utility"),
            }
            try:
                tmp_dir = Path("/tmp/csw_leads")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                leads_csv = tmp_dir / "leads.csv"
                df = pd.DataFrame([lead])
                if leads_csv.exists():
                    df_existing = pd.read_csv(leads_csv)
                    df = pd.concat([df_existing, df], ignore_index=True)
                df.to_csv(leads_csv, index=False)
                st.success("Thanks! Your info was recorded. (Temporary storage for prototype.)")
            except Exception as e:
                st.warning(f"Could not store lead locally: {e}\nWe’ll wire this to Google Sheets/HubSpot next.")

    progress_bar()
    col1, col2 = st.columns(2)
    with col1: st.button("← Back", on_click=go_back)
    with col2: st.button("Start Over", on_click=lambda: st.session_state.update(step=1))
