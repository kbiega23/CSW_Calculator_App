# app/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
import re

# --- Ensure repo root is importable (…/app -> repo root) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Try to import the engine helpers; fall back to direct CSV read if not available
try:
    from engine.engine import load_weather, load_lists, load_savings_lookup
    ENGINE_OK = True
except Exception:
    ENGINE_OK = False

# ------------------ App Config ------------------
st.set_page_config(page_title="CSW Savings Calculator (Prototype)", layout="wide")
st.title("Commercial Secondary Windows — Savings Calculator (Prototype)")
st.markdown("> **Preliminary estimates only.** For detailed results, a full energy model is recommended.")

# ------------------ Data Loading ------------------
DATA_DIR = REPO_ROOT / "data"

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

# ------------------ Helpers ------------------
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
    # de-dup preserve order
    seen, ordered = set(), []
    for x in opts:
        if x not in seen:
            ordered.append(x); seen.add(x)
    return ordered

def hvac_options_for_building(lists_df: pd.DataFrame, building_label: str):
    """
    Return HVAC options for the selected building using lists.csv 'as is'.

    Strategy:
    1) Find the 'HVAC Type, <Building>' column (handles variations like Hosp/MF).
    2) If there's a 'Building Type' indicator column, filter rows to just this building.
    3) Clean/unique options.
    4) Apply a strict override list for building types where we know the exact,
       allowed values (e.g., Office), to avoid cross-contamination from the sheet.
    """
    if lists_df is None or lists_df.empty:
        return []

    import re

    # Map UI building names to tokens likely to appear in header or indicator cells
    tokens_map = {
        "Office": ["office"],
        "Hotel": ["hotel"],
        "School": ["school", "sh"],
        "Hospital": ["hospital", "hosp"],
        "Multi-family": ["multi-family", "multifamily", "mf", "m-f"],
    }
    tokens = tokens_map.get(building_label, [building_label.lower()])

    # --- 1) Locate the HVAC Type column for this building ---
    hvac_col = None
    for c in lists_df.columns:
        c_norm = str(c).strip()
        if c_norm.lower().startswith("hvac type"):
            # exact: "HVAC Type, <token>"
            if any(re.fullmatch(rf"(?i)hvac type[, ]+\s*{re.escape(t)}\s*", c_norm) for t in tokens):
                hvac_col = c
                break
    if hvac_col is None:
        # contains variant, e.g. "HVAC Type - Hospital"
        for c in lists_df.columns:
            c_norm = str(c).strip().lower()
            if c_norm.startswith("hvac type") and any(t in c_norm for t in tokens):
                hvac_col = c
                break
    if hvac_col is None:
        return []

    # --- 2) Optional: filter rows to this building if an indicator column exists ---
    def find_building_key(df: pd.DataFrame):
        # Look for a column that marks the building type per row
        preferred = [
            "Building Type", "Type of Building", "Bldg Type", "Building",
            "Building_Type", "Type"
        ]
        cols_norm = {str(c).strip().lower(): c for c in df.columns}
        for p in preferred:
            if p.lower() in cols_norm:
                return cols_norm[p.lower()]
        # fallback: any col that contains both 'building' and 'type'
        for c in df.columns:
            lc = str(c).strip().lower()
            if "building" in lc and "type" in lc:
                return c
        return None

    bkey = find_building_key(lists_df)
    if bkey:
        # keep rows where indicator mentions this building token
        mask = lists_df[bkey].astype(str).str.lower().apply(
            lambda s: any(t in s for t in tokens)
        )
        hvac_series = lists_df.loc[mask, hvac_col]
    else:
        hvac_series = lists_df[hvac_col]

    # --- 3) Clean & de-dup while preserving order ---
    def _clean(series):
        if series is None or series.empty:
            return []
        s = series.dropna().astype(str).map(lambda x: x.strip())
        vals = [x for x in s if x and x.lower() not in ("none", "nan")]
        seen, ordered = set(), []
        for x in vals:
            if x not in seen:
                ordered.append(x); seen.add(x)
        return ordered

    options = _clean(hvac_series)

    # --- 4) Strict overrides for known building types (prevents sheet cross-bleed) ---
    strict_allowed = {
        "Office": [
            "Packaged VAV with electric reheat",
            "Packaged VAV with hydronic reheat",
            "Built-up VAV with hydronic reheat",
            "Other",
        ],
        # You can add per-building strict lists here as you finalize them:
        # "Hotel": [...],
        # "School": [...],
        # "Hospital": [...],
        # "Multi-family": [...],
    }

    if building_label in strict_allowed:
        allowed = strict_allowed[building_label]
        # case-insensitive match to keep original casing if found
        lower_to_orig = {x.lower(): x for x in options}
        filtered = []
        for a in allowed:
            found = lower_to_orig.get(a.lower())
            filtered.append(found if found else a)
        return filtered

    # Otherwise, ensure 'Other' is present at the end
    if not any(x.lower() == "other" for x in options):
        options.append("Other")
    return options


# ------------------ 1) Project & Location ------------------
st.header("1) Project & Location")

# Expect exact headers in weather_information.csv:
REQ = ["State", "Cities", "Heating Degree Days (HDD)", "Cooling Degree Days (CDD)"]
missing = [c for c in REQ if c not in weather_df.columns]
if missing:
    st.error(
        "Your 'data/weather_information.csv' must contain these columns:\n\n"
        f"- {', '.join(REQ)}\n\n"
        f"Missing: {', '.join(missing)}"
    )
    st.stop()

wdf = weather_df[REQ].copy()
wdf = wdf.dropna(subset=["State", "Cities"])
wdf["State"] = wdf["State"].astype(str).str.strip()
wdf["Cities"] = wdf["Cities"].astype(str).str.strip()

states = sorted(wdf["State"].unique().tolist())
state = st.selectbox("State", states)

cities = sorted(wdf.loc[wdf["State"] == state, "Cities"].unique().tolist())
city = st.selectbox("City", cities)

sel = wdf[(wdf["State"] == state) & (wdf["Cities"] == city)]
if sel.empty:
    st.error("Selected State/City not found in weather_information.csv.")
    st.stop()

hdd = _to_num(sel["Heating Degree Days (HDD)"].iloc[0])
cdd = _to_num(sel["Cooling Degree Days (CDD)"].iloc[0])
if hdd is None or cdd is None:
    st.error("HDD/CDD values could not be parsed as numbers.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.metric("Location HDD (base 65)", f"{hdd:.0f}")
with c2:
    st.metric("Location CDD (base 65)", f"{cdd:.0f}")

elec_utility = st.text_input("Electric Utility (optional)")
gas_utility  = st.text_input("Natural Gas Utility (optional)")

# ------------------ 2) Building ------------------
st.header("2) Building")

building_type = st.selectbox("Building Type", ["Office", "Hotel", "School", "Hospital", "Multi-family"])
area = st.number_input("Building Area (ft²)", min_value=0.0, value=100000.0, step=1000.0, format="%.0f")
floors = st.number_input("Number of Floors", min_value=1, value=3, step=1)

# Dynamic HVAC options from lists.csv for the selected building
hvac_dynamic_options = hvac_options_for_building(lists_df, building_type)
fallback_hvac = ["VAV", "Packaged RTU", "Heat Pump", "Boiler/Chiller"]
hvac_type = st.selectbox(
    "HVAC System Type",
    hvac_dynamic_options if hvac_dynamic_options else fallback_hvac,
    help=("Options loaded from Lists for this building type."
          if hvac_dynamic_options else "Lists column not found; showing fallback list.")
)

heating_fuel = st.selectbox("Heating Fuel", ["Elec", "Gas"])
cooling_installed = st.selectbox("Cooling Installed?", ["Yes", "No"])
annual_hours = st.number_input("Annual Operating Hours", min_value=0, max_value=8760, value=4000, step=100)

existing_window = st.selectbox(
    "Type of Existing Window",
    ["Single pane", "Double pane", "New double pane (U<0.35)"]
)

# ------------------ 3) Proposed CSW ------------------
st.header("3) Proposed CSW")

csw_type = st.selectbox("Type of CSW Analyzed", ["Single", "Double"])
csw_area = st.number_input("Sq. Ft. of CSW Installed", min_value=0.0, value=10000.0, step=500.0, format="%.0f")

# ------------------ 4) Rates ------------------
st.header("4) Rates")

elec_rate = st.number_input("Electric Rate ($/kWh)", min_value=0.0, value=0.14, step=0.01, format="%.4f")
gas_rate  = st.number_input("Natural Gas Rate ($/therm)", min_value=0.0, value=0.76, step=0.01, format="%.4f")

# ------------------ Derived Inputs (Auto) ------------------
st.subheader("Derived Inputs")

# Excel WWR formula: =F27 / ((F18/F19)^0.5 * 4 * 15 * F19)
# Map to variables: csw_area / (((area / floors) ** 0.5) * 4 * 15 * floors)
den = (((area / floors) ** 0.5) * 4 * 15 * floors) if floors > 0 and area > 0 else 0.0
wwr = csw_area / den if den > 0 else 0.0

warn = ""
if wwr > 0.50:
    warn = " ⚠️ **WWR is over 0.50** — check CSW area vs. building geometry."
st.write(f"Estimated Window-to-Wall Ratio (WWR): **{wwr:.3f}**{warn}")

# ------------------ 5) Results (Placeholder for now) ------------------
st.header("5) Results (Preview)")
st.info(
    "Regression-driven energy savings (kWh/therms/$), EUI changes, and peak cooling "
    "will appear here once we wire the Office regression + Savings Lookup logic."
)

colA, colB, colC = st.columns(3)
with colA:
    st.metric("Annual Electric Savings (kWh/yr)", "—")
with colB:
    st.metric("Annual Natural Gas Savings (therms/yr)", "—")
with colC:
    st.metric("Annual Energy Cost Savings ($/yr)", "—")

# ------------------ Lead Capture ------------------
st.header("Contact (Lead Capture)")

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
            "state": state,
            "city": city,
            "hdd65": hdd,
            "cdd65": cdd,
            "building_type": building_type,
            "area_ft2": area,
            "floors": floors,
            "hvac_type": hvac_type,
            "heating_fuel": heating_fuel,
            "cooling_installed": cooling_installed,
            "annual_hours": annual_hours,
            "existing_window": existing_window,
            "csw_type": csw_type,
            "csw_area_ft2": csw_area,
            "elec_rate": elec_rate,
            "gas_rate": gas_rate,
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
