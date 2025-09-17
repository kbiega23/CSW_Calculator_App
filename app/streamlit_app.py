# app/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import re

# ================== Setup & Imports ==================
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from engine.engine import (
        load_weather,
        load_lists,
        compute_savings,  # new
    )
    ENGINE_OK = True
except Exception:
    ENGINE_OK = False

st.set_page_config(page_title="CSW Savings Calculator (Prototype)", layout="wide")
st.title("Commercial Secondary Windows — Savings Calculator (Prototype)")
st.markdown("> **Preliminary estimates only.** For detailed results, a full energy model is recommended.")

DATA_DIR = REPO_ROOT / "data"

# ================== Data Loaders ==================
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
def load_hvac_overrides():
    p = DATA_DIR / "hvac_overrides.csv"
    if not p.exists():
        # empty schema so code paths still work
        return pd.DataFrame(columns=["Building Type", "Sub-Building Type", "HVAC Option"])
    df = pd.read_csv(p)
    # Normalize strings
    for c in ["Building Type", "Sub-Building Type", "HVAC Option"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

# Load frames (engine if available, else fallback)
try:
    _wdf = load_weather() if ENGINE_OK else _load_weather_fallback()
except Exception as e:
    st.error(f"Could not load weather data.\n\n{e}")
    st.stop()

try:
    lists_df = load_lists() if ENGINE_OK else _load_lists_fallback()
except Exception as e:
    st.warning(f"Lists data not loaded: {e}")
    lists_df = pd.DataFrame()

overrides_df = load_hvac_overrides()

# ================== Helpers ==================
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

def _standardize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize to columns: State, City, HDD, CDD (accepts 'Cities' or 'City', and long HDD/CDD names).
    """
    df = df.copy()
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    # State
    # (assume present)
    # City/Cities
    if "cities" in cols_lower:
        rename_map[cols_lower["cities"]] = "City"
    elif "city" in cols_lower:
        rename_map[cols_lower["city"]] = "City"

    # HDD
    hdd_key = None
    for key in ["hdd", "heating degree days (hdd)"]:
        if key in cols_lower:
            hdd_key = cols_lower[key]
            break
    if hdd_key:
        rename_map[hdd_key] = "HDD"

    # CDD
    cdd_key = None
    for key in ["cdd", "cooling degree days (cdd)"]:
        if key in cols_lower:
            cdd_key = cols_lower[key]
            break
    if cdd_key:
        rename_map[cdd_key] = "CDD"

    df = df.rename(columns=rename_map)
    return df

# HVAC options helpers
def hvac_options_for_building(lists_df: pd.DataFrame, building_label: str):
    """Fallback: read 'HVAC Type, <building>' column from lists.csv."""
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
    """
    Preferred: use data/hvac_overrides.csv for (Building, Sub-Building).
    Fallback: use building-wide column from lists.csv.
    Always ensure 'Other' is available.
    """
    # Use overrides if present
    if overrides_df is not None and not overrides_df.empty and sub_building_type:
        mask = (
            overrides_df["Building Type"].str.casefold() == str(building_type).casefold()
        ) & (
            overrides_df["Sub-Building Type"].str.casefold() == str(sub_building_type).casefold()
        )
        rows = overrides_df.loc[mask, "HVAC Option"]
        opts = [o for o in rows.dropna().astype(str).map(str.strip).tolist() if o and o.lower() not in ("none", "nan")]
        # De-dup while preserving order
        seen, cleaned = set(), []
        for o in opts:
            if o not in seen:
                cleaned.append(o); seen.add(o)
        if cleaned:
            if not any(o.lower() == "other" for o in cleaned):
                cleaned.append("Other")
            return cleaned

    # Fallback
    fallback = hvac_options_for_building(lists_df, building_type) or []
    if not any(o.lower() == "other" for o in fallback):
        fallback.append("Other")
    return fallback

def compute_sub_building(building_type: str, area_ft2: float, floors: int, hvac_type: str, school_subtype: str | None):
    """
    Current rules (mirroring Excel assumptions we’ve discussed):
      Office: Large if (area>30,000 ft² and HVAC == 'Built-up VAV with hydronic reheat'); else Mid-size.
      Hotel:  Small if HVAC in {PTAC, PTHP}; else Large.
      School: Subtype is user-selected (Primary/Secondary).
      Multi-family: Low-rise if floors < 4; else Mid-rise.
      Hospital: fixed ('Hospital').
    """
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
    """
    Excel WWR formula:
      WWR = CSW Area / [ (sqrt(Building Area / Floors) * 4) * Wall Height * Floors ]
    """
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

# ================== Wizard State ==================
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
    if (cur is None or cur == "") and st.session_state.get(saved_key) not in (None, ""):
        st.session_state[cur_key] = st.session_state[saved_key]
    return st.session_state.get(cur_key)

# ---------- Prereq guards (restore first) ----------
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

# ================== Step 1: Project & Location ==================
if st.session_state.step == 1:
    st.header("Step 1 — Project & Location")

    wdf = _standardize_weather_columns(_wdf)

    missing = [c for c in ["State", "City", "HDD", "CDD"] if c not in wdf.columns]
    if missing:
        st.error(
            "Your weather file must contain columns that can be mapped to:\n\n"
            "- State, City (or Cities), HDD (or Heating Degree Days (HDD)), CDD (or Cooling Degree Days (CDD))\n\n"
            f"Missing after normalization: {', '.join(missing)}"
        )
        st.stop()

    wdf = wdf.dropna(subset=["State", "City"]).copy()
    wdf["State"] = wdf["State"].astype(str).str.strip()
    wdf["City"] = wdf["City"].astype(str).str.strip()

    states = sorted(wdf["State"].unique().tolist())
    st.selectbox("State", states, key="state")
    st.session_state["state_saved"] = st.session_state.get("state")

    cities = sorted(wdf.loc[wdf["State"] == st.session_state.get("state"), "City"].unique().tolist())
    st.selectbox("City", cities, key="city")
    st.session_state["city_saved"] = st.session_state.get("city")

    sel = wdf[(wdf["State"] == st.session_state.get("state")) & (wdf["City"] == st.session_state.get("city"))]
    if sel.empty:
        st.error("Selected State/City not found in weather data.")
        st.stop()

    hdd = _to_num(sel["HDD"].iloc[0])
    cdd = _to_num(sel["CDD"].iloc[0])
    if hdd is None or cdd is None:
        st.error("HDD/CDD values could not be parsed as numbers.")
        st.stop()

    st.session_state["hdd65"] = hdd
    st.session_state["cdd65"] = cdd
    st.session_state["hdd65_saved"] = hdd
    st.session_state["cdd65_saved"] = cdd

    c1, c2 = st.columns(2)
    with c1: st.metric("Location HDD (base 65)", f"{hdd:.0f}")
    with c2: st.metric("Location CDD (base 65)", f"{cdd:.0f}")

    st.text_input("Electric Utility (optional)", key="elec_utility")
    st.text_input("Natural Gas Utility (optional)", key="gas_utility")

    progress_bar()
    st.button("Next →", on_click=go_next, type="primary")

# ================== Step 2: Building Type (School subtype here) ==================
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

# ================== Step 3: Building Details (varies by building type) ==================
elif st.session_state.step == 3:
    st.header("Step 3 — Building Details")

    building_type = (st.session_state.get("building_type_saved") or st.session_state.get("building_type") or "").strip()
    if not building_type:
        st.session_state.step = 2
        st.experimental_rerun()

    school_subtype_val = st.session_state.get("school_subtype_saved") or st.session_state.get("school_subtype")

    # Common selectors
    def heating_fuel_select():
        st.selectbox("Heating Fuel", ["Natural gas", "Electric", "None"], key="heating_fuel")

    def existing_window_select():
        st.selectbox("Type of Existing Window", ["Single pane", "Double pane"], key="existing_window")

    def cooling_installed_select():
        st.selectbox("Cooling Installed?", ["Yes", "No"], key="cooling_installed")

    def hvac_select_with_overrides(bt: str, sub_building_type: str):
        options = hvac_options_from_overrides(overrides_df, bt, sub_building_type, lists_df)
        if not options:
            options = ["Other"]
        idx = options.index(st.session_state["hvac_type"]) if st.session_state.get("hvac_type") in options else 0
        st.selectbox("HVAC System Type", options, key="hvac_type", index=idx, help=f"Options for **{bt} → {sub_building_type}**")

    # ---- Per-building layouts ----
    if building_type == "Office":
        st.number_input("Building Area (ft²)", min_value=0.0, value=float(st.session_state.get("area_ft2", 100000.0)), step=1000.0, format="%.0f", key="area_ft2")
        st.number_input("Number of Floors", min_value=1, value=int(st.session_state.get("floors", 3)), step=1, key="floors")

        sub_guess = compute_sub_building(
            "Office",
            st.session_state.get("area_ft2", 0.0),
            st.session_state.get("floors", 1),
            st.session_state.get("hvac_type", ""),
            None
        )
        hvac_select_with_overrides("Office", sub_guess)

        heating_fuel_select()
        st.number_input("Annual Operating Hours", min_value=0, max_value=8760, value=int(st.session_state.get("annual_hours", 4000)), step=100, key="annual_hours")
        existing_window_select()
        cooling_installed_select()

        sub_build = compute_sub_building(
            "Office",
            st.session_state.get("area_ft2", 0.0),
            st.session_state.get("floors", 1),
            st.session_state.get("hvac_type", ""),
            None
        )
        st.session_state["sub_building_type"] = sub_build
        st.info(f"**Sub-Building Type:** {sub_build}")

    elif building_type == "School":
        st.number_input("Building Area (ft²)", min_value=0.0, value=float(st.session_state.get("area_ft2", 80000.0)), step=1000.0, format="%.0f", key="area_ft2")
        st.number_input("Number of Floors", min_value=1, value=int(st.session_state.get("floors", 2)), step=1, key="floors")
        existing_window_select()

        sub_build_known = compute_sub_building(
            "School",
            st.session_state.get("area_ft2", 0.0),
            st.session_state.get("floors", 1),
            st.session_state.get("hvac_type", ""),
            school_subtype_val
        )
        hvac_select_with_overrides("School", sub_build_known)
        cooling_installed_select()
        heating_fuel_select()

        st.session_state["sub_building_type"] = sub_build_known
        st.info(f"**Sub-Building Type:** {sub_build_known}")

    elif building_type == "Hotel":
        st.number_input("Building Area (ft²)", min_value=0.0, value=float(st.session_state.get("area_ft2", 120000.0)), step=1000.0, format="%.0f", key="area_ft2")
        st.number_input("Number of Floors", min_value=1, value=int(st.session_state.get("floors", 4)), step=1, key="floors")

        sub_guess_hotel = compute_sub_building(
            "Hotel",
            st.session_state.get("area_ft2", 0.0),
            st.session_state.get("floors", 1),
            st.session_state.get("hvac_type", ""),
            None
        )
        hvac_select_with_overrides("Hotel", sub_guess_hotel)

        st.number_input("Average Occupancy Rate (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.get("hotel_occupancy_pct", 70.0)), step=1.0, key="hotel_occupancy_pct")
        heating_fuel_select()
        cooling_installed_select()
        existing_window_select()

        sub_build = compute_sub_building(
            "Hotel",
            st.session_state.get("area_ft2", 0.0),
            st.session_state.get("floors", 1),
            st.session_state.get("hvac_type", ""),
            None
        )
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
        sub_build_mf = compute_sub_building(
            "Multi-family",
            st.session_state.get("area_ft2", 0.0),
            st.session_state.get("floors", 1),
            "",
            None
        )
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

# ================== Step 4: Proposed CSW ==================
elif st.session_state.step == 4:
    st.header("Step 4 — Proposed CSW")

    st.selectbox("Type of CSW Analyzed", ["Single", "Double"], key="csw_type")
    st.number_input("Sq. Ft. of CSW Installed", min_value=0.0, value=float(st.session_state.get("csw_area_ft2", 10000.0)), step=500.0, format="%.0f", key="csw_area_ft2")

    # Derived: WWR (Excel-equivalent)
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

# ================== Step 5: Rates ==================
elif st.session_state.step == 5:
    st.header("Step 5 — Utility Rates")

    st.number_input("Electric Rate ($/kWh)", min_value=0.0, value=float(st.session_state.get("elec_rate", 0.14)), step=0.01, format="%.4f", key="elec_rate")
    st.number_input("Natural Gas Rate ($/therm)", min_value=0.0, value=float(st.session_state.get("gas_rate", 0.76)), step=0.01, format="%.4f", key="gas_rate")

    progress_bar()
    col1, col2 = st.columns(2)
    with col1: st.button("← Back", on_click=go_back)
    with col2: st.button("Next →", on_click=go_next, type="primary")

# ================== Step 6: Review & Results ==================
elif st.session_state.step == 6:
    st.header("Step 6 — Review & Results")

    # Pull state
    btype = (st.session_state.get("building_type_saved") or st.session_state.get("building_type") or "").strip()
    sbtype = (st.session_state.get("sub_building_type") or "").strip()

    # Weather
    hdd = float(st.session_state.get("hdd65_saved") or st.session_state.get("hdd65") or 0)
    cdd = float(st.session_state.get("cdd65_saved") or st.session_state.get("cdd65") or 0)

    # Building inputs
    area_sf = float(st.session_state.get("area_ft2") or 0)
    floors = int(st.session_state.get("floors") or 1)
    hvac_ui = st.session_state.get("hvac_type") or ""
    heat_fuel_ui = st.session_state.get("heating_fuel") or "Natural gas"
    cooling_installed_str = (st.session_state.get("cooling_installed") or "Yes").strip().lower()
    cooling_installed = True if cooling_installed_str in ("yes", "y", "true") else False
    existing_window_type_ui = st.session_state.get("existing_window") or "Single pane"
    csw_glazing_ui = st.session_state.get("csw_type") or "Single"
    annual_hours = float(st.session_state.get("annual_hours") or 0) if btype == "Office" else None
    hotel_occ = float(st.session_state.get("hotel_occupancy_pct") or 0) if btype == "Hotel" else None
    include_infil = bool(st.session_state.get("mf_include_infiltration")) if btype == "Multi-family" else None

    # Rates
    elec_rate = float(st.session_state.get("elec_rate") or 0)
    gas_rate  = float(st.session_state.get("gas_rate") or 0)

    # Compute WWR again (idempotent)
    wwr = compute_wwr(area_sf, floors, float(st.session_state.get("csw_area_ft2") or 0), wall_height_ft=15.0)
    st.session_state["wwr"] = wwr

    # Summary table
    summary = {
        "State": st.session_state.get("state_saved") or st.session_state.get("state"),
        "City": st.session_state.get("city_saved") or st.session_state.get("city"),
        "HDD65": hdd,
        "CDD65": cdd,
        "Building Type": btype,
        "Sub-Building Type": sbtype,
        "Area (ft²)": area_sf,
        "Floors": floors,
        "HVAC": hvac_ui,
        "Heating Fuel": heat_fuel_ui,
        "Cooling Installed": "Yes" if cooling_installed else "No",
        "Existing Window": existing_window_type_ui,
        "CSW Type": csw_glazing_ui,
        "CSW Area (ft²)": st.session_state.get("csw_area_ft2"),
        "WWR (calc)": round(wwr, 3),
        "WWR (%)": f"{wwr*100:.1f}%",
        "Elec Rate ($/kWh)": elec_rate,
        "Gas Rate ($/therm)": gas_rate,
    }
    if btype == "Office":
        summary["Annual Hours"] = annual_hours
    if btype == "Hotel":
        summary["Occupancy Rate (%)"] = hotel_occ
    if btype == "School":
        summary["School Type"] = st.session_state.get("school_subtype_saved") or st.session_state.get("school_subtype")
    if btype == "Multi-family":
        summary["Infiltration Savings?"] = include_infil

    st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "Value"}))

    # ---- Compute savings via engine ----
    try:
        # If user explicitly chose "Heating Fuel: None", we’ll pass "Electric" to satisfy the mapper
        # and then zero heat afterward (to mirror your Excel behavior where applicable).
        heat_fuel_effective = "Electric" if (heat_fuel_ui or "").lower() == "none" else heat_fuel_ui

        res = compute_savings(
            weather_hdd=hdd,
            weather_cdd=cdd,
            building_type=btype,
            sub_building_type=sbtype,
            hvac_ui=hvac_ui,
            heating_fuel_ui=heat_fuel_effective,
            cooling_installed=bool(cooling_installed),
            existing_window_type_ui=existing_window_type_ui,
            csw_glazing_ui=csw_glazing_ui,
            building_area_sf=area_sf,
            annual_operating_hours=annual_hours,
            hotel_occupancy_pct=hotel_occ,
            include_infiltration=include_infil,
        )

        # Enforce "Heating Fuel: None" → zero heating savings
        if (heat_fuel_ui or "").lower() == "none":
            res["elec_heat_kwh_per_sf"] = 0.0
            res["gas_heat_therm_per_sf"] = 0.0
            res["total_kwh"] = (res["cool_kwh_per_sf"]) * area_sf
            res["total_therms"] = 0.0

        # Cost savings
        annual_cost_savings = res["total_kwh"] * elec_rate + res["total_therms"] * gas_rate

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Per-SF Savings")
            st.metric("Electric heating (kWh/sf·yr)", f"{res['elec_heat_kwh_per_sf']:.3f}")
            st.metric("Cooling (kWh/sf·yr)", f"{res['cool_kwh_per_sf']:.3f}")
            st.metric("Gas heating (therms/sf·yr)", f"{res['gas_heat_therm_per_sf']:.4f}")

        with c2:
            st.subheader("Annual Totals")
            st.metric("Total electric (kWh/yr)", f"{res['total_kwh']:.0f}")
            st.metric("Total gas (therms/yr)", f"{res['total_therms']:.0f}")
            st.metric("Est. utility savings ($/yr)", f"${annual_cost_savings:,.0f}")

        st.caption("These are preliminary, estimated results. For more accuracy, a full energy model is recommended.")

    except Exception as e:
        st.error(f"Could not compute savings for the selected combination: {e}")
        st.stop()

    # ---- Lead capture ----
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
                "hdd65": hdd,
                "cdd65": cdd,
                "building_type": btype,
                "sub_building_type": sbtype,
                "area_ft2": area_sf,
                "floors": floors,
                "hvac_type": hvac_ui,
                "heating_fuel": heat_fuel_ui,
                "cooling_installed": "Yes" if cooling_installed else "No",
                "existing_window": existing_window_type_ui,
                "annual_hours": annual_hours,
                "hotel_occupancy_pct": hotel_occ,
                "school_subtype": st.session_state.get("school_subtype_saved") or st.session_state.get("school_subtype"),
                "mf_include_infiltration": include_infil,
                "csw_type": csw_glazing_ui,
                "csw_area_ft2": st.session_state.get("csw_area_ft2"),
                "wwr": st.session_state.get("wwr"),
                "elec_rate": elec_rate,
                "gas_rate": gas_rate,
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
