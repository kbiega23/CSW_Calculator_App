# engine/engine.py  (CSV-first; Excel as fallback)
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import re
import pandas as pd

# ---------------- Paths & simple loaders ----------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def _find_excel_file() -> Path:
    expected = DATA_DIR / "CSW Savings Calculator 2_0_0_Unlocked.xlsx"
    if expected.exists():
        return expected
    # fallback: scan upwards for any matching name
    root = Path(__file__).resolve().parents[2]
    for p in root.rglob("CSW Savings Calculator 2_0_0_Unlocked.xlsx"):
        return p
    raise FileNotFoundError("Excel workbook not found and CSVs not detected either.")

def load_weather() -> pd.DataFrame:
    p = DATA_DIR / "weather_information.csv"
    df = pd.read_csv(p)
    keep = [c for c in df.columns if not str(c).lower().startswith("none")]
    return df[keep].rename(
        columns=lambda c: str(c).strip()
        .replace("Heating Degree Days (HDD)", "HDD")
        .replace("Cooling Degree Days (CDD)", "CDD")
        .replace("Cities", "City")
    )

def load_lists() -> pd.DataFrame:
    p = DATA_DIR / "lists.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def load_hvac_allowed() -> pd.DataFrame:
    p = DATA_DIR / "hvac_options.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["Building Type", "Sub-Building Type", "HVAC Option"])

# ---------------- Regression table helpers ----------------

def _read_sheet_with_header(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """(Excel path) find header row (contains 'No.') and return a cleaned DataFrame."""
    raw = xl.parse(sheet, header=None)
    mask = raw.apply(lambda r: r.astype(str).str.contains("No\\.|Base", na=False).any(), axis=1)
    hdr_idx = mask[mask].index[0] if mask.any() else 0
    return xl.parse(sheet, header=hdr_idx)

def _read_regression_csv(path: Path) -> pd.DataFrame:
    """
    CSVs may include a header row or the Excel-like header buried a few rows down.
    We search for a row containing 'No.' or 'Base' and use that as the header row;
    otherwise we treat the first row as header.
    """
    raw = pd.read_csv(path, header=None)
    mask = raw.apply(lambda r: r.astype(str).str.contains("No\\.|Base", na=False).any(), axis=1)
    if mask.any():
        hdr_idx = mask[mask].index[0]
        df = pd.read_csv(path, header=hdr_idx)
    else:
        df = pd.read_csv(path)  # assume first row is header
    return df

def _norm_cols(cols) -> list[str]:
    out = []
    heat_mode = False
    cool_mode = False
    heat_i = 0
    cool_i = 0
    for c in cols:
        s = str(c).strip()
        if re.search(r"(?i)heating\s*coeff", s):
            out.append("Heating Coefficients"); heat_mode, cool_mode = True, False
            continue
        if re.search(r"(?i)cooling\s*coeff", s):
            out.append("Cooling Coefficients"); heat_mode, cool_mode = False, True
            continue
        if s in {"a", "b", "c"}:
            if cool_mode:
                cool_i += 1; out.append(f"cool_{'abc'[cool_i-1]}")
            elif heat_mode:
                heat_i += 1; out.append(f"heat_{'abc'[heat_i-1]}")
            else:
                # default to heat first
                heat_i += 1; out.append(f"heat_{'abc'[heat_i-1]}")
            continue
        out.append("No." if s == "Unnamed: 0" else s)
    return out

def _normalize_table(df: pd.DataFrame, building_tag: str) -> pd.DataFrame:
    """
    Normalize mixed layouts into:
      Base, CSW, Size, HVAC, Fuel, Occupancy, heat_a,b,c, cool_a,b,c, BuildingTag
    """
    df = df.copy()
    df.columns = _norm_cols(df.columns)

    # locate logical columns by name variants
    def first_present(*names):
        for n in names:
            if n in df.columns: return n
        return None

    size_col = first_present("Office Size", "Hotel Size", "MF Type", "Size")
    hvac_col = first_present("HVAC/Fuel Type", "HVAC Type", "HVAC")
    fuel_col = first_present("Fuel Type", "Fuel")
    occ_col  = first_present("Occupancy")

    # compose normalized frame
    out = pd.DataFrame()
    def col(name): return df[name] if name and name in df.columns else pd.Series([None]*len(df))

    # base/secondary glazing can appear as 'CSW Type' or 'CSW'
    csw_col = first_present("CSW Type", "CSW")
    base_col = first_present("Base")

    out["BuildingTag"] = building_tag
    out["Base"] = col(base_col).astype(str).str.strip()
    out["CSW"]  = col(csw_col).astype(str).str.strip()
    out["Size"] = col(size_col).astype(str).str.strip() if size_col else ""
    out["HVAC"] = col(hvac_col).astype(str).str.strip() if hvac_col else ""
    out["Fuel"] = col(fuel_col).astype(str).str.strip() if fuel_col else ""
    out["Occupancy"] = pd.to_numeric(col(occ_col), errors="coerce") if occ_col else pd.Series([None]*len(df))

    for k in ["heat_a","heat_b","heat_c","cool_a","cool_b","cool_c"]:
        out[k] = pd.to_numeric(col(k), errors="coerce")

    # keep data rows
    out = out[out["Base"].str.lower().isin(["single", "double"])].reset_index(drop=True)
    return out

def _tag_from_filename(stem: str) -> str:
    """
    Extract BuildingTag from filenames like:
    'Regresson List_Office.csv', 'Regression List_SH.csv', 'Regresson_List_PS.csv', etc.
    """
    s = stem
    # remove extension semantics already
    m = re.search(r"(?i)regress\w*\s*[_\s-]*list[_\s-]*(.+)$", s)
    if m:
        return m.group(1).strip()
    # fallback: if the stem itself *is* a known tag
    return s.strip()

# ---------------- Load regressions (CSV-first) ----------------

def load_regressions() -> pd.DataFrame:
    """
    Prefer CSVs in data/ that contain 'regress' + 'list' in filename (case-insensitive).
    If none found, fall back to reading the Excel workbook.
    Then normalize and combine to a single table.
    """
    csvs = [p for p in DATA_DIR.glob("*.csv") if re.search(r"(?i)regress\w*.*list", p.stem)]
    pieces = []

    if csvs:
        for p in csvs:
            tag = _tag_from_filename(p.stem)
            df = _read_regression_csv(p)
            pieces.append(_normalize_table(df, building_tag=tag))
    else:
        # Fallback: Excel
        xpf = _find_excel_file()
        xl = pd.ExcelFile(xpf)
        sheets = [s for s in xl.sheet_names if s.lower().startswith("regress")]
        for s in sheets:
            tag = s.replace("Regresson List_", "").replace("Regression List_", "").strip()
            df = _read_sheet_with_header(xl, s)
            pieces.append(_normalize_table(df, building_tag=tag))

    if not pieces:
        raise FileNotFoundError("No regression CSVs found in data/ and no Excel workbook available.")

    big = pd.concat(pieces, ignore_index=True)

    # Map tags to canonical building labels
    big["BuildingTag"] = big["BuildingTag"].astype(str).str.strip()
    big["Building"] = big["BuildingTag"].map({
        "Office": "Office",
        "SH": "Hotel",
        "LH": "Hotel",
        "PS": "School",
        "SS": "School",
        "Hosp": "Hospital",
        "MF": "Multi-family",
    }).fillna(big["BuildingTag"].replace({
        "Regression List_Office":"Office",
        "Regression List_SH":"Hotel",
        "Regression List_LH":"Hotel",
        "Regression List_PS":"School",
        "Regression List_SS":"School",
        "Regression List_Hosp":"Hospital",
        "Regression List_MF":"Multi-family",
    }))

    # Subtype helpers
    big["SchoolSubtype"] = big["BuildingTag"].map({"PS": "Primary School", "SS": "Secondary School"}).fillna("")
    big["MFSubtype"] = big.apply(lambda r: r["Size"] if str(r["BuildingTag"]) == "MF" else "", axis=1)
    big["HotelSize"] = big.apply(lambda r: r["Size"] if str(r["Building"]) == "Hotel" else "", axis=1)

    # Clean strings
    for c in ["Base","CSW","Size","HVAC","Fuel","Building","SchoolSubtype","MFSubtype","HotelSize"]:
        big[c] = big[c].astype(str).str.strip()

    return big

# ---------------- Mapping UI → regression codes ----------------

def _ui_to_base(window_type_ui: str) -> str:
    return "Single" if "single" in (window_type_ui or "").lower() else "Double"

def _ui_csw_type(csw_ui: str) -> str:
    return "Single" if "single" in (csw_ui or "").lower() else "Double"

def _ui_to_hvac_code(building: str, sub_building: str, hvac_ui: str, fuel_ui: str) -> Tuple[str, str]:
    fuel_code = "Electric" if (fuel_ui or "").lower().startswith("electric") else "Natural Gas"
    b = (building or "").lower()

    if b == "office":
        s = (hvac_ui or "").lower()
        if "packaged vav with electric reheat" in s: return "PVAV_Elec", "Electric"
        if "packaged vav with hydronic reheat" in s: return "PVAV_Gas", "Natural Gas"
        if "built-up vav with hydronic reheat" in s: return "VAV", fuel_code
        return "VAV", fuel_code

    if b == "school":
        s = (hvac_ui or "").lower()
        if "fan coil" in s: return "FCU", fuel_code
        if "vav" in s: return "VAV", fuel_code
        return "VAV", fuel_code

    if b == "hotel":
        s = (hvac_ui or "").lower()
        if "ptac" in s: return "PTAC", "Electric"
        if "pthp" in s: return "PTHP", "Electric"
        if "fan coil" in s: return "FCU", fuel_code
        return "FCU", fuel_code

    if b == "hospital":
        return "VAV", fuel_code

    if b.startswith("multi"):
        s = (hvac_ui or "").lower()
        if "ptac" in s: return "PTAC", "Electric"
        if "fan coil" in s: return "FCU", fuel_code
        return "PTAC", "Electric"

    return "VAV", fuel_code

def _hours_bucket(hours: float) -> float:
    buckets = [2080.0, 2912.0, 8760.0]
    return min(buckets, key=lambda b: abs(b - float(hours or 0)))

def _hotel_occ_bucket(occ_pct: float) -> float:
    return 33.0 if (occ_pct or 0) <= 50 else 100.0

# ---------------- Core compute ----------------

def _pick_coeff_row(reg: pd.DataFrame,
                    building: str,
                    sub_building: str,
                    base_glass: str,
                    csw_glass: str,
                    hvac_code: str,
                    fuel_code: str,
                    hours_bucket: Optional[float] = None,
                    occ_bucket: Optional[float] = None) -> pd.Series:
    b = (building or "").lower()
    sb = (sub_building or "").lower()
    df = reg.copy()

    df = df[df["Base"].str.lower() == base_glass.lower()]
    df = df[df["CSW"].str.lower() == csw_glass.lower()]

    if b == "office":
        size = "Mid" if "mid-size" in sb else "Large"
        df = df[(df["Building"] == "Office") & (df["Size"].str.contains(size, case=False, na=False))]
        df = df[df["HVAC"].str.upper() == hvac_code.upper()]
        df = df[df["Fuel"].str.lower() == fuel_code.lower()]

    elif b == "school":
        ss = "Primary School" if "primary" in sb else "Secondary School"
        df = df[(df["Building"] == "School") & (df["SchoolSubtype"] == ss)]
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
            df = df[(df["Occupancy"].round(0) == round(occ_bucket, 0)) | (df["Occupancy"].isna())]

    elif b.startswith("multi"):
        mf = "Low" if "low" in sb else "Mid"
        df = df[(df["Building"] == "Multi-family") & (df["MFSubtype"].str.contains(mf, case=False, na=False))]
        df = df[df["HVAC"].str.upper() == hvac_code.upper()]
        df = df[df["Fuel"].str.lower() == fuel_code.lower()]

    if df.empty:
        raise ValueError("No matching regression row for the selected combination.")
    return df.iloc[0]

def _poly(a: float, b: float, c: float, x: float) -> float:
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
    reg = load_regressions()

    base_glass = _ui_to_base(existing_window_type_ui)
    csw_glass  = _ui_csw_type(csw_glazing_ui)
    hvac_code, fuel_code = _ui_to_hvac_code(building_type, sub_building_type, hvac_ui, heating_fuel_ui)

    hrs_bucket = _hours_bucket(annual_operating_hours or 0) if (building_type or "").lower()=="office" else None
    occ_bucket = _hotel_occ_bucket(hotel_occupancy_pct or 0) if (building_type or "").lower()=="hotel" else None

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

    heat_per_sf = _poly(row["heat_a"], row["heat_b"], row["heat_c"], float(weather_hdd))
    cool_per_sf = _poly(row["cool_a"], row["cool_b"], row["cool_c"], float(weather_cdd))

    if (building_type or "").lower() == "office":
        factor = (hrs_bucket or 8760.0) / 8760.0
        cool_per_sf *= factor

    if not cooling_installed:
        cool_per_sf = 0.0

    if (fuel_code or "").lower().startswith("electric"):
        elec_heat_kwh_per_sf = max(0.0, heat_per_sf)
        gas_heat_therm_per_sf = 0.0
    else:
        elec_heat_kwh_per_sf = 0.0
        gas_heat_therm_per_sf = max(0.0, heat_per_sf)

    if include_infiltration is not None and not include_infiltration and (building_type or "").lower().startswith("multi"):
        elec_heat_kwh_per_sf *= 0.9
        cool_per_sf *= 0.9
        gas_heat_therm_per_sf *= 0.9

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
