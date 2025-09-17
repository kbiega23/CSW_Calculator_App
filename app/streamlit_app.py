# app/streamlit_app.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import sys
import re

# ================== Setup & Imports ==================
# Make repo root importable so "engine" works when Streamlit launches from /app
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer package import; fall back to module import if __init__.py is missing
try:
    from engine import Inputs, compute_savings
except Exception:
    from engine.engine import Inputs, compute_savings  # fallback

APP_DIR = REPO_ROOT
DATA_DIR = APP_DIR / "data"

# Bump this any time you change loaders to bust Streamlit's cache easily
CACHE_BUSTER = "2025-09-17c"

st.set_page_config(page_title="CSW Savings Calculator", page_icon="🪟", layout="centered")


# ================== Data Loading ==================
@st.cache_data
def load_weather(_v: str = "1") -> pd.DataFrame:
    """
    Robust loader for weather_information.csv
    - Fuzzy-matches HDD/CDD headers (e.g., HDD65, Heating Degree Days, etc.)
    - Trims whitespace, cleans numeric cells, drops empty rows
    """
    path = DATA_DIR / "weather_information.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)

    # Drop totally empty columns and strip whitespace
    df = df.dropna(axis=1, how="all")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # Normalized name -> original
    norm = {c: re.sub(r"[^a-z0-9]+", "", c.lower()) for c in df.columns}

    def pick(regexes, exclude=None):
        exclude = exclude or []
        candidates = []
        for c in df.columns:
            n = norm[c]
            if any(re.search(rx, n) for rx in regexes) and not any(re.search(ex, n) for ex in exclude):
                candidates.append(c)
        if not candidates:
            return None

        # Rank candidates: presence of '65', 'degree/deg', 'base', and non-null count
        def score(col):
            n = norm[col]
            s = 0
            if "65" in n:
                s += 2
            if "degree" in n or "deg" in n:
                s += 1
            if "base" in n:
                s += 1
            s += int(df[col].notna().sum() > 0)
            return s

        candidates.sort(key=lambda c: (score(c), df[c].notna().sum()), reverse=True)
        return candidates[0]

    state_col = pick([r"^state", r"^province", r"^region"]) or df.columns[0]
    city_col = pick([r"^city", r"^cities", r"^municipality", r"^location"]) or df.columns[1]
    hdd_col = pick([r"hdd", r"heatingdegree"], exclude=[r"cdd"])
    cdd_col = pick([r"cdd", r"coolingdegree"])

    # Clean numeric cells (strip commas/units)
    def to_num(x):
        s = (x or "").replace(",", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        return pd.to_numeric(s, errors="coerce")

    HDD = df[hdd_col].map(to_num) if hdd_col else pd.Series([pd.NA] * len(df))
    CDD = df[cdd_col].map(to_num) if cdd_col else pd.Series([pd.NA] * len(df))

    out = pd.DataFrame(
        {
            "State": df[state_col],
            "City": df[city_col],
            "HDD": HDD,
            "CDD": CDD,
        }
    )

    # Drop rows missing state/city
    out = out[(out["State"].astype(str) != "") & (out["City"].astype(str) != "")]
    return out.reset_index(drop=True)


@st.cache_data
def load_lists() -> Dict[str, List[str]]:
    """Load lists.csv (Arrow-safe: coerce Value to string)."""
    path = DATA_DIR / "lists.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    cat_col = next((c for c in df.columns if c.lower().strip() == "category"), df.columns[0])
    val_col = next((c for c in df.columns if c.lower().strip() == "value"), df.columns[1])
    df[val_col] = df[val_col].astype(str)  # avoid mixed type warnings in Streamlit/pyarrow

    out: Dict[str, List[str]] = {}
    for cat, grp in df.groupby(cat_col):
        vals = [v for v in grp[val_col].tolist() if v]
        out[cat] = vals
    return out


@st.cache_data
def load_hvac_overrides() -> pd.DataFrame:
    """
    Load hvac_overrides.csv
    Expect columns like: BuildingType, SubType, HVACOption (case-insensitive).
    """
    path = DATA_DIR / "hvac_overrides.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    m = {c.lower().strip(): c for c in df.columns}
    bt = m.get("buildingtype", list(df.columns)[0])
    stype = m.get("subtype", list(df.columns)[1])
    hvac = m.get("hvacoption", list(df.columns)[2])
    return df.rename(columns={bt: "BuildingType", stype: "SubType", hvac: "HVACOption"})


def allowed_hvac(building_type: str, subtype: str) -> List[str]:
    """Filter HVAC options per hvac_overrides.csv, always include 'Other'."""
    df = load_hvac_overrides()
    subset = df[(df["BuildingType"] == building_type) & (df["SubType"] == subtype)]
    opts = subset["HVACOption"].unique().tolist()
    if "Other" not in opts:
        opts.append("Other")
    return opts or ["Other"]


# ================== Subtype helpers ==================
def mf_subtype_from_floors(floors: Optional[int]) -> str:
    return "Low-rise Multifamily" if (floors is not None and floors < 4) else "Mid-rise Multifamily"


def office_subtype_preview(area_sf: float, hvac_label_hint: str = "Built-up VAV with hydronic reheat") -> str:
    # The override file uses Mid-size/Large Office, but options are the same; this is for consistency.
    return "Large Office" if (area_sf > 30000 and "built-up vav" in hvac_label_hint.lower()) else "Mid-size Office"


def hotel_subtype_preview(hvac_label_hint: str = "PTAC") -> str:
    s = hvac_label_hint.lower()
    return "Small Hotel" if ("ptac" in s or "pthp" in s) else "Large Hotel"


# ================== Wizard State ==================
if "step" not in st.session_state:
    st.session_state.step = 1


def next_step():
    st.session_state.step += 1


def prev_step():
    st.session_state.step = max(1, st.session_state.step - 1)


# Load data (with cache-busting arg for weather)
weather_df = load_weather(_v=CACHE_BUSTER)
lists = load_lists()

# ================== UI ==================
st.title("CSW Savings Calculator")

# ---------- Step 1: Location ----------
if st.session_state.step == 1:
    st.header("1) Location")

    states = sorted(weather_df["State"].unique().tolist())
    default_state_idx = states.index("Colorado") if "Colorado" in states else 0
    state = st.selectbox("State", states, index=default_state_idx, key="state")

    cities = sorted(weather_df[weather_df["State"] == state]["City"].unique().tolist())
    city = st.selectbox("City", cities, key="city")

    row = weather_df[(weather_df["State"] == state) & (weather_df["City"] == city)].head(1)
    hdd_val = row["HDD"].iloc[0] if not row.empty else None
    cdd_val = row["CDD"].iloc[0] if not row.empty else None
    hdd = float(hdd_val) if pd.notna(hdd_val) else 6500.0
    cdd = float(cdd_val) if pd.notna(cdd_val) else 900.0

    st.write(f"HDD: **{hdd:.0f}**, CDD: **{cdd:.0f}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Next ➜"):
            st.session_state.location = {"state": state, "city": city, "hdd": hdd, "cdd": cdd}
            next_step()
    with col2:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# ---------- Step 2: Building Type ----------
elif st.session_state.step == 2:
    st.header("2) Building Type")
    btypes = ["Office", "School", "Hotel", "Hospital", "Multi-family"]
    btype = st.selectbox("Building Type", btypes, key="btype")

    school_sub = None
    if btype == "School":
        school_sub = st.radio("School Sub-type", ["Primary", "Secondary"], horizontal=True, key="school_sub")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back"):
            prev_step()
    with col2:
        if st.button("Next ➜"):
            st.session_state.btype = {"building_type": btype, "school_subtype": school_sub}
            next_step()

# ---------- Step 3: Building Details ----------
elif st.session_state.step == 3:
    st.header("3) Building Details")

    btype = st.session_state.btype["building_type"]
    school_sub = st.session_state.btype.get("school_subtype")

    # Common inputs
    area_sf = st.number_input("Building floor area (sf)", min_value=0.0, value=50000.0, step=1000.0, key="area")
    floors = st.number_input("Number of floors", min_value=1, value=3, step=1, key="floors")
    existing_win = st.selectbox("Existing window type", ["Single pane", "Double pane"], index=0, key="exist_win")
    heat_fuel = st.selectbox("Primary heating fuel", ["Natural Gas", "Electric", "None"], index=0, key="fuel")
    cooling_installed = (
        st.radio("Cooling installed?", ["Yes", "No"], horizontal=True, index=0, key="cooling") == "Yes"
    )

    # Optional per-building fields
    annual_hours = None
    occupancy = None
    mf_infil_include = True

    # HVAC options (filtered by hvac_overrides.csv)
    subtype_for_hvac = {
        "Office": office_subtype_preview(area_sf),
        "School": f"{school_sub} School" if btype == "School" else "",
        "Hotel": hotel_subtype_preview(),
        "Hospital": "Hospital",
        "Multi-family": mf_subtype_from_floors(floors),
    }[btype]

    if btype == "Multi-family":
        mf_infil_include = st.checkbox("Include infiltration savings?", value=True, key="mf_infil")

    hvac_opts = allowed_hvac(btype, subtype_for_hvac)

    if btype == "Office":
        annual_hours = st.number_input(
            "Annual operating hours", min_value=0.0, value=3000.0, step=100.0, key="hours"
        )
        hvac = st.selectbox("HVAC system", hvac_opts, key="hvac")
    elif btype == "School":
        hvac = st.selectbox("HVAC system", hvac_opts, key="hvac")
    elif btype == "Hotel":
        occupancy = st.slider("Average occupancy (%)", min_value=0, max_value=100, value=70, step=5, key="occ")
        hvac = st.selectbox("HVAC system", hvac_opts, key="hvac")
    elif btype == "Hospital":
        hvac = st.selectbox("HVAC system", hvac_opts, key="hvac")
    else:  # Multi-family
        hvac = st.selectbox("HVAC system", hvac_opts, key="hvac")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back"):
            prev_step()
    with col2:
        if st.button("Next ➜"):
            st.session_state.details = {
                "area_sf": area_sf,
                "floors": int(floors),
                "annual_hours": annual_hours,
                "occupancy": occupancy,
                "existing_window": existing_win,
                "hvac": hvac,
                "heating_fuel": heat_fuel,
                "cooling_installed": cooling_installed,
                "mf_infil_include": mf_infil_include,
            }
            next_step()

# ---------- Step 4: Rates & Scope ----------
elif st.session_state.step == 4:
    st.header("4) Rates & Scope")

    elec_rate = st.number_input(
        "Electricity rate ($/kWh)", min_value=0.0, value=0.14, step=0.01, format="%.4f", key="erate"
    )
    gas_rate = st.number_input(
        "Natural Gas rate ($/therm)", min_value=0.0, value=1.20, step=0.05, format="%.4f", key="grate"
    )
    csw_area = st.number_input(
        "CSW installed area (sf) — leave 0 to use building area", min_value=0.0, value=0.0, step=100.0, key="csw_area"
    )
    csw_panes = st.radio("CSW glazing", ["Double", "Single"], index=0, horizontal=True, key="csw_panes")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back"):
            prev_step()
    with col2:
        if st.button("Next ➜"):
            st.session_state.rates = {
                "elec_rate": float(elec_rate),
                "gas_rate": float(gas_rate),
                "csw_area": float(csw_area),
                "csw_panes": csw_panes,
            }
            next_step()

# ---------- Step 5: Review & Results ----------
else:
    st.header("5) Review & Results")

    # Gather inputs across steps
    loc = st.session_state.location
    bt = st.session_state.btype
    det = st.session_state.details
    rt = st.session_state.rates

    inp = Inputs(
        state=loc["state"],
        city=loc["city"],
        hdd=loc["hdd"],
        cdd=loc["cdd"],
        building_type=bt["building_type"],
        school_subtype=bt.get("school_subtype"),
        area_sf=det["area_sf"],
        floors=det["floors"],
        annual_hours=det.get("annual_hours"),
        occupancy_rate_pct=det.get("occupancy"),
        existing_window=det["existing_window"],
        hvac_label=det["hvac"],
        heating_fuel_label=det["heating_fuel"],
        cooling_installed=det["cooling_installed"],
        mf_infiltration_include=det["mf_infil_include"],
        elec_rate_per_kwh=rt["elec_rate"],
        gas_rate_per_therm=rt["gas_rate"],
        csw_installed_area_sf=rt["csw_area"],
        csw_panes=rt["csw_panes"],
    )

    err_block = st.empty()
    result_block = st.container()

    try:
        res = compute_savings(inp)

        with result_block:
            st.subheader("Savings Summary")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Electric heat savings (kWh)", f"{res.totals['elec_kwh']:,}")
                st.metric("Cooling savings (kWh)", f"{res.totals['cool_kwh']:,}")
            with c2:
                st.metric("Gas heat savings (therms)", f"{res.totals['gas_therms']:,}")
                st.metric("Total kWh", f"{res.totals['total_kwh']:,}")
            with c3:
                st.metric("Total therms", f"{res.totals['total_therms']:,}")
                st.metric("Estimated cost savings", f"${res.totals['cost_savings']:,}")

            st.caption(f"Area used for totals: **{res.totals['area_used_sf']:.0f} sf**")

            st.markdown("**Per-square-foot savings**")
            st.table(pd.DataFrame([res.per_sf]))

            st.markdown("**EUI (if available in CSV)**")
            st.table(pd.DataFrame([res.eui]))

        # Debug Panel
        with st.expander("🔎 Debug panel (LookKey tracing)"):
            st.markdown("**Attempted LookKey(s)**")
            st.code("\n".join(res.debug.get("attempted_keys", [])) or "(none)")

            st.markdown("**Matched rows (values parsed)**")
            matched = res.debug.get("matched_keys", {})
            st.json(matched if matched else {"matched": "(none)"})

            st.markdown("**CSV audit**")
            st.json(res.debug.get("csv_audit", {}))

            st.markdown("**Expected building prefix**")
            st.code(res.debug.get("expected_building_prefix", ""))

            st.markdown("**Inventory for this building prefix (top 50)**")
            inv = res.debug.get("inventory_keys_for_building_prefix", [])
            st.code("\n".join(inv) or "(none)")

            st.markdown("**Prefix used for suggestions**")
            st.code(res.debug.get("prefix_used_for_suggestions", ""))

            st.markdown("**Keys starting with that prefix (top 50)**")
            prefs = res.debug.get("prefix_candidates", [])
            st.code("\n".join(prefs) or "(none)")

            st.markdown("**Similar keys (fuzzy matches)**")
            st.code("\n".join(res.debug.get("similar_keys", [])) or "(none)")

    except Exception as e:
        with err_block.container():
            msg = getattr(e, "args", ["Error"])[0]
            st.error(f"Could not compute savings for the selected combination: {msg}")

            # If the engine attached a rich debug payload as args[1], surface it
            if len(getattr(e, "args", [])) > 1 and isinstance(e.args[1], dict):
                dbg = e.args[1]
                with st.expander("🔎 Debug panel (LookKey tracing)"):
                    st.markdown("**Attempted LookKey(s)**")
                    st.code("\n".join(dbg.get("attempted_keys", [])) or "(none)")

                    st.markdown("**CSV audit**")
                    st.json(dbg.get("csv_audit", {}))

                    st.markdown("**Expected building prefix**")
                    st.code(dbg.get("expected_building_prefix", ""))

                    st.markdown("**Inventory for this building prefix (top 50)**")
                    st.code("\n".join(dbg.get("inventory_keys_for_building_prefix", [])) or "(none)")

                    st.markdown("**Prefix used for suggestions**")
                    st.code(dbg.get("prefix_used_for_suggestions", ""))

                    st.markdown("**Keys starting with that prefix (top 50)**")
                    st.code("\n".join(dbg.get("prefix_candidates", [])) or "(none)")

                    st.markdown("**Similar keys (fuzzy matches)**")
                    st.code("\n".join(dbg.get("similar_keys", [])) or "(none)")

    col1, col2 = st.columns(2)
    with col1:
        st.button("◀ Back", on_click=prev_step)
    with col2:
        st.button("Start Over", on_click=lambda: st.session_state.clear())
