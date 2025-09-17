# engine/engine.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import pandas as pd

# --------- Paths & simple loaders ----------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def _find_excel_file() -> Path:
    """
    Locate the Excel workbook "CSW Savings Calculator 2_0_0_Unlocked.xlsx".
    Default location is data/, but we scan the repo just in case.
    """
    expected = DATA_DIR / "CSW Savings Calculator 2_0_0_Unlocked.xlsx"
    if expected.exists():
        return expected
    # fallback: scan upwards for any matching name
    root = Path(__file__).resolve().parents[2]
    candidates = list(root.rglob("CSW Savings Calculator 2_0_0_Unlocked.xlsx"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "Could not find 'CSW Savings Calculator 2_0_0_Unlocked.xlsx'. "
        "Place it in the repo under data/."
    )

def load_weather() -> pd.DataFrame:
    """
    Load weather_information.csv (State, Cities, HDD, CDD and 'None*' extra cols).
    We drop the 'None*' columns if present.
    """
    p = DATA_DIR / "weather_information.csv"
    df = pd.read_csv(p)
    # drop accidental columns like None, None_1, ...
    keep = [c for c in df.columns if not str(c).lower().startswith("none")]
    return df[keep].rename(
        columns=lambda c: c.strip()
        .replace("Heating Degree Days (HDD)", "HDD")
        .replace("Cooling Degree Days (CDD)", "CDD")
        .replace("Cities", "City")
    )

def load_lists() -> pd.DataFrame:
    """Load lists.csv for any future lookups you want to do."""
    p = DATA_DIR / "lists.csv"
    return pd.read_csv(p)

def load_hvac_allowed() -> pd.DataFrame:
    """
    hvac_options.csv (three columns: Building Type, Sub-Building Type, HVAC Option)
    used to constrain the dropdown choices.
    """
    p = DATA_DIR / "hvac_options.csv"
    return pd.read_csv(p)

# --------- Regression tables normalizer ----------

def _read_sheet_with_header(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Find the header row (contains 'No.') and return a cleaned DataFrame."""
    raw = xl.parse(sheet, header=None)
    # find the row containing 'No.' which marks the header
    mask = raw.apply(lambda r: r.astype(str).str.contains("No.", na=False).any(), axis=1)
    if not mask.any():
        raise ValueError(f"Could not locate header in sheet {sheet}")
    hdr_idx = mask[mask].index[0]
    df = xl.parse(sheet, header=hdr_idx)
    return df

def _norm_cols(cols) -> list[str]:
    """Give sensible names to the mixed header columns."""
    out = []
    # the workbooks sometimes repeat 'a','b','c' under two headings
    # we'll map them to heat_a, heat_b, heat_c, cool_a, cool_b, cool_c
    heat_seen = 0
    cool_seen = 0
    for c in cols:
        s = str(c).strip()
        if s in {"Heating Coefficients", "Cooling Coefficients"}:
            out.append(s)
            continue
        if s in {"a", "b", "c"}:
            if "Cooling Coefficients" in out and out[-1] == "Cooling Coefficients":
                cool_seen += 1
                out.append(f"cool_{'abc'[cool_seen-1]}")
            elif "Heating Coefficients" in out and out[-1] == "Heating Coefficients":
                heat_seen += 1
                out.append(f"heat_{'abc'[heat_seen-1]}")
            else:
                # if we can't tell, assume heat first
                heat_seen += 1
                out.append(f"heat_{'abc'[heat_seen-1]}")
            continue
        # default: keep original
        out.append(s if s != "Unnamed: 0" else "No.")
    return out

def _normalize_table(df: pd.DataFrame, building_tag: str) -> pd.DataFrame:
    """
    Normalize into a common schema:

    Common columns we try to create:
      Base, CSW Type, Size, HVAC, Fuel, Occupancy, heat_a,b,c, cool_a,b,c, BuildingTag
    Size is one of: Mid/Large (Office), Small/Large (Hotel), Low/Mid (MF), '' for School/Hospital.
    """
    # remap columns
    df = df.copy()
    df.columns = _norm_cols(df.columns)

    # alias possibilities
    size_col = None
    for cand in ["Office Size", "Hotel Size", "MF Type"]:
        if cand in df.columns:
            size_col = cand
            break

    hvac_col = None
    for cand in ["HVAC/Fuel Type", "HVAC Type"]:
        if cand in df.columns:
            hvac_col = cand
            break

    fuel_col = "Fuel Type" if "Fuel Type" in df.columns else None
    occ_col = "Occupancy" if "Occupancy" in df.columns else None

    keep = {
        "Base": "Base",
        "CSW Type": "CSW",
        size_col or "": "Size",
        hvac_col or "": "HVAC",
        fuel_col or "": "Fuel",
        occ_col or "": "Occupancy",
        "heat_a": "heat_a",
        "heat_b": "heat_b",
        "heat_c": "heat_c",
        "cool_a": "cool_a",
        "cool_b": "cool_b",
        "cool_c": "cool_c",
    }

    # build normalized frame
    norm = pd.DataFrame()
    # fill safe get
    def col(name): return df[name] if name in df.columns else pd.Series([None]*len(df))

    norm["BuildingTag"] = building_tag
    norm["Base"] = col("Base").astype(str).str.strip()
    norm["CSW"] = col("CSW").astype(str).str.strip() if "CSW" in df.columns else col("CSW Type").astype(str).str.strip()
    norm["Size"] = col(size_col) if size_col else ""
    norm["HVAC"] = col(hvac_col) if hvac_col else ""
    norm["Fuel"] = col(fuel_col) if fuel_col else ""
    norm["Occupancy"] = pd.to_numeric(col(occ_col), errors="coerce") if occ_col else pd.Series([None]*len(df))
    # coefficients
    for k in ["heat_a","heat_b","heat_c","cool_a","cool_b","cool_c"]:
        norm[k] = pd.to_numeric(col(k), errors="coerce")

    # drop header row(s)
    norm = norm[norm["Base"].str.lower().isin(["single", "double"])]
    # clean sizes text
    norm["Size"] = norm["Size"].astype(str).str.strip()
    norm["HVAC"] = norm["HVAC"].astype(str).str.strip()
    norm["Fuel"] = norm["Fuel"].astype(str).str.strip()
    return norm.reset_index(drop=True)

def load_regressions() -> pd.DataFrame:
    """
    Load and combine all 'Regresson List_*' sheets into one normalized table.
    BuildingTag values used:
      Office, SH (Small Hotel), LH (Large Hotel), PS, SS, Hosp, MF
    """
    xpf = _find_excel_file()
    xl = pd.ExcelFile(xpf)

    sheets = [s for s in xl.sheet_names if s.startswith("Regresson List_")]
    pieces = []
    for s in sheets:
        tag = s.replace("Regresson List_", "").strip()
        df = _read_sheet_with_header(xl, s)
        pieces.append(_normalize_table(df, building_tag=tag))
    big = pd.concat(pieces, ignore_index=True)

    # unify hotel/mf tags to consistent size values
    # Office keeps Size Mid/Large
    # Hotel: Small/ Large → tag both as Hotel, decide by Size later
    big["Building"] = big["BuildingTag"].map({
        "Office": "Office",
        "SH": "Hotel",
        "LH": "Hotel",
        "PS": "School",
        "SS": "School",
        "Hosp": "Hospital",
        "MF": "Multi-family",
    }).fillna(big["BuildingTag"])

    # For School, keep subtype in BuildingTag (PS/SS)
    big["SchoolSubtype"] = big["BuildingTag"].map({"PS": "Primary School", "SS": "Secondary School"}).fillna("")
    # For MF, Size should be Low/Mid
    big["MFSubtype"] = big.apply(lambda r: r["Size"] if r["BuildingTag"] == "MF" else "", axis=1)
    # For Hotel, Size is in 'Size' (Small/Large)
    big["HotelSize"] = big.apply(lambda r: r["Size"] if r["Building"] == "Hotel" else "", axis=1)

    # normalize string categories
    for c in ["Base","CSW","Size","HVAC","Fuel","Building","SchoolSubtype","MFSubtype","HotelSize"]:
        big[c] = big[c].astype(str).str.strip()

    return big

# --------- Mapping from UI to regression codes ----------

def _ui_to_base(window_type_ui: str) -> str:
    # "single pane" / "double pane" → "Single"/"Double"
    return "Single" if "single" in window_type_ui.lower() else "Double"

def _ui_csw_type(csw_ui: str) -> str:
    # "Single" or "Double" glazing CSW
    return "Single" if "single" in csw_ui.lower() else "Double"

def _ui_to_hvac_code(building: str, sub_building: str, hvac_ui: str, fuel_ui: str) -> Tuple[str, str]:
    """
    Return (HVAC_code, Fuel_code) as used in regression tables.
    """
    fuel_code = "Electric" if fuel_ui.lower().startswith("electric") else "Natural Gas"

    b = building.lower()
    sb = sub_building.lower()

    # Office
    if b == "office":
        if "packaged vav with electric reheat" in hvac_ui.lower():
            return "PVAV_Elec", "Electric"
        if "packaged vav with hydronic reheat" in hvac_ui.lower():
            return "PVAV_Gas", "Natural Gas"
        if "built-up vav with hydronic reheat" in hvac_ui.lower():
            # large office VAV; let fuel follow user's choice
            return "VAV", fuel_code
        # fallback
        return "VAV", fuel_code

    # School
    if b == "school":
        if "fan coil" in hvac_ui.lower():
            return "FCU", fuel_code
        if "central ducted vav" in hvac_ui.lower() or "vav" in hvac_ui.lower():
            return "VAV", fuel_code
        return "VAV", fuel_code

    # Hotel
    if b == "hotel":
        if "ptac" in hvac_ui.lower():
            return "PTAC", "Electric"  # PTAC heat is electric
        if "pthp" in hvac_ui.lower():
            return "PTHP", "Electric"  # heat pump = electric
        if "fan coil" in hvac_ui.lower():
            return "FCU", fuel_code
        return "FCU", fuel_code

    # Hospital
    if b == "hospital":
        return "VAV", fuel_code

    # Multi-family
    if b.startswith("multi"):
        if "ptac" in hvac_ui.lower():
            return "PTAC", "Electric"
        if "fan coil" in hvac_ui.lower():
            return "FCU", fuel_code
        return "PTAC", "Electric"

    return "VAV", fuel_code

def _hours_bucket(hours: float) -> float:
    """
    Excel lookup uses three discrete hours: 2080, 2912, 8760.
    Bucket to the nearest, to mirror the workbook behavior.
    """
    buckets = [2080.0, 2912.0, 8760.0]
    return min(buckets, key=lambda b: abs(b - float(hours or 0)))

def _hotel_occ_bucket(occ_pct: float) -> float:
    """Regressions exist for 33% and 100% occupancy."""
    return 33.0 if (occ_pct or 0) <= 50 else 100.0

# --------- Core compute ----------

def _pick_coeff_row(reg: pd.DataFrame,
                    building: str,
                    sub_building: str,
                    base_glass: str,
                    csw_glass: str,
                    hvac_code: str,
                    fuel_code: str,
                    hours_bucket: Optional[float] = None,
                    occ_bucket: Optional[float] = None) -> pd.Series:
    """
    Filter regression table to the single row of coefficients that matches the user's selections.
    """
    b = building.lower()
    sb = (sub_building or "").lower()
    df = reg.copy()

    # Base & CSW
    df = df[df["Base"].str.lower() == base_glass.lower()]
    df = df[df["CSW"].str.lower() == csw_glass.lower()]

    # Building dimension filters
    if b == "office":
        size = "Mid" if "mid-size" in sb else "Large"
        df = df[(df["Building"] == "Office") & (df["Size"].str.contains(size, case=False, na=False))]
        df = df[df["HVAC"].str.upper() == hvac_code.upper()]
        df = df[df["Fuel"].str.lower() == fuel_code.lower()]

    elif b == "school":
        # PS / SS tagged in SchoolSubtype
        ss = "Primary School" if "primary" in sb else "Secondary School"
        df = df[(df["Building"] == "School") & (df["SchoolSubtype"] == ss)]
        # HVAC/Fuel
        df = df[df["HVAC"].str.upper() == hvac_code.upper()]
        df = df[df["Fuel"].str.lower() == fuel_code.lower()]

    elif b == "hospital":
        df = df[df["Building"] == "Hospital"]
        df = df[df["HVAC"].str.upper() == hvac_code.upper()]
        df = df[df["Fuel"].str.lower() == fuel_code.lower()]

    elif b == "hotel":
        size = "Small" if "small" in sb else "Large"
        df = df[(df["Building"] == "Hotel") & (df["HotelSize"].str.contains(size, case=False, na=False))]
        df = df[df["HVAC"].str.upper() == hvac_code.upper()]
        df = df[df["Fuel"].str.lower() == fuel_code.lower()]
        if occ_bucket is not None and "Occupancy" in df.columns:
            df = df[(df["Occupancy"].round(0) == round(occ_bucket,0)) | (df["Occupancy"].isna())]

    elif b.startswith("multi"):
        # Low / Mid in MFSubtype
        mf = "Low" if "low" in sb else "Mid"
        df = df[(df["Building"] == "Multi-family") & (df["MFSubtype"].str.contains(mf, case=False, na=False))]
        df = df[df["HVAC"].str.upper() == hvac_code.upper()]
        df = df[df["Fuel"].str.lower() == fuel_code.lower()]

    if df.empty:
        raise ValueError("Could not find matching regression row for the selected combination.")

    # Take the first best match
    return df.iloc[0]

def _poly(a: float, b: float, c: float, x: float) -> float:
    """a + b*x + c*x^2"""
    a = float(a or 0); b = float(b or 0); c = float(c or 0); x = float(x or 0)
    return a + b*x + c*(x*x)

def compute_savings(
    *,
    weather_hdd: float,
    weather_cdd: float,
    building_type: str,
    sub_building_type: str,
    hvac_ui: str,
    heating_fuel_ui: str,
    cooling_installed: bool,
    existing_window_type_ui: str,
    csw_glazing_ui: str,
    building_area_sf: float,
    annual_operating_hours: Optional[float] = None,
    hotel_occupancy_pct: Optional[float] = None,
    include_infiltration: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Returns a dict with per-SF and total savings:
      - elec_heat_kwh_per_sf
      - cool_kwh_per_sf
      - gas_heat_therm_per_sf
      - total_kwh   (electric_heat + cooling) * area
      - total_therms (gas_heat * area)
    """
    reg = load_regressions()

    base_glass = _ui_to_base(existing_window_type_ui)               # "Single"/"Double"
    csw_glass  = _ui_csw_type(csw_glazing_ui)                       # "Single"/"Double"
    hvac_code, fuel_code = _ui_to_hvac_code(building_type, sub_building_type, hvac_ui, heating_fuel_ui)

    hrs_bucket = _hours_bucket(annual_operating_hours or 0) if building_type.lower()=="office" else None
    occ_bucket = _hotel_occ_bucket(hotel_occupancy_pct or 0) if building_type.lower()=="hotel" else None

    row = _pick_coeff_row(
        reg=reg,
        building=building_type,
        sub_building=sub_building_type,
        base_glass=base_glass,
        csw_glass=csw_glass,
        hvac_code=hvac_code,
        fuel_code=fuel_code,
        hours_bucket=hrs_bucket,
        occ_bucket=occ_bucket,
    )

    # Base polynomial (per SF) from HDD/CDD
    heat_per_sf = _poly(row["heat_a"], row["heat_b"], row["heat_c"], float(weather_hdd))
    cool_per_sf = _poly(row["cool_a"], row["cool_b"], row["cool_c"], float(weather_cdd))

    # Office hours bucketing: mirror Excel's discrete behavior (approx).
    if building_type.lower() == "office":
        # Scale cooling roughly by hours/8760; heating is mostly load-driven,
        # but the Excel lookup varies slightly with hours. We'll keep heat as-is
        # and scale cooling by hours bucket factor to mimic the lookup choices.
        factor = (hrs_bucket or 8760.0) / 8760.0
        cool_per_sf *= factor

    # Hotel occupancy: the regression line we picked already bakes in the occupancy bucket.
    # Nothing further needed here.

    # Cooling installed?
    if not cooling_installed:
        cool_per_sf = 0.0

    # Split heat savings into electric vs gas depending on chosen fuel
    if fuel_code.lower().startswith("electric"):
        elec_heat_kwh_per_sf = max(0.0, heat_per_sf)
        gas_heat_therm_per_sf = 0.0
    else:
        elec_heat_kwh_per_sf = 0.0
        gas_heat_therm_per_sf = max(0.0, heat_per_sf)

    # Optional: infiltration toggle for Multi-family (Excel has reduction factors; if excluded, just zero the delta)
    if include_infiltration is not None and not include_infiltration and building_type.lower().startswith("multi"):
        # Simple conservative approach: remove a small portion of savings if user excludes infiltration.
        # (If you want to mirror exact Excel factors later, we can read and apply them when available.)
        elec_heat_kwh_per_sf *= 0.9
        cool_per_sf *= 0.9
        gas_heat_therm_per_sf *= 0.9

    # Totals
    area = float(building_area_sf or 0)
    total_kwh = (elec_heat_kwh_per_sf + cool_per_sf) * area
    total_therms = gas_heat_therm_per_sf * area

    return {
        "elec_heat_kwh_per_sf": elec_heat_kwh_per_sf,
        "cool_kwh_per_sf": cool_per_sf,
        "gas_heat_therm_per_sf": gas_heat_therm_per_sf,
        "total_kwh": total_kwh,
        "total_therms": total_therms,
    }
