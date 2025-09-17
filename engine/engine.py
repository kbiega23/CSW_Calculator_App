# engine/engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

# ---------------- Paths ----------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ---------------- CSV helpers ----------------
def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

# ---------------- Public loaders used by the app ----------------
def load_weather() -> pd.DataFrame:
    """Load weather_information.csv and normalize the four key columns."""
    p = DATA_DIR / "weather_information.csv"
    df = _read_csv(p)
    # map likely headers -> canonical
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if "state" in lc and "hdd" not in lc and "cdd" not in lc: ren[c] = "State"
        elif "cities" in lc or lc == "city": ren[c] = "Cities"
        elif "heating degree" in lc or "(hdd" in lc or lc == "hdd": ren[c] = "Heating Degree Days (HDD)"
        elif "cooling degree" in lc or "(cdd" in lc or lc == "cdd": ren[c] = "Cooling Degree Days (CDD)"
    df = df.rename(columns=ren)
    keep = ["State","Cities","Heating Degree Days (HDD)","Cooling Degree Days (CDD)"]
    return df[[c for c in keep if c in df.columns]]

def load_lists() -> pd.DataFrame:
    p = DATA_DIR / "lists.csv"
    return _read_csv(p) if p.exists() else pd.DataFrame()

def load_savings_lookup() -> pd.DataFrame:
    p = DATA_DIR / "savings_lookup.csv"
    df = _read_csv(p)
    # normalize known columns (works whether names are already normalized or not)
    rename = {
        "Look-up_Conc":"LookKey", "LookKey":"LookKey",
        "Base":"Base", "CSW":"CSW", "Size":"Size",
        "Building_Type":"Building_Type", "HVAC_Type":"HVAC_Type", "Fuel":"Fuel",
        "PTHP":"PTHP", "Hours":"Hours",
        "Electric_savings_Heat_kWhperSF":"ElecHeat_kWh_SF",
        "electric_savings_Cooling_and_Aux_kWhperSF":"Cool_kWh_SF",
        "Gas_savings_Heat_thermsperSF":"GasHeat_therm_SF",
        "Base_EUI_kBtuperSFperyr":"Base_EUI",
        "CSW_EUI_kBtuperSFperyr":"CSW_EUI",
        "Calculated_Savings_EUI":"Savings_EUI_frac",
        "Cool_adjust":"Cool_adjust",
        "infiltration_savings_reduction_factors_Heat":"Infil_Heat_Factor",
        "infiltration_reduction_factors_Cool":"Infil_Cool_Factor",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for c in ["Hours","ElecHeat_kWh_SF","Cool_kWh_SF","GasHeat_therm_SF",
              "Base_EUI","CSW_EUI","Savings_EUI_frac","Cool_adjust",
              "Infil_Heat_Factor","Infil_Cool_Factor"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------------- Key-building utilities (mirror Excel) ----------------
def _hvac_label_to_code(building: str, hvac_label: str, fuel_ui: str) -> Tuple[str, str]:
    """Map user-facing HVAC label to regression code; return (hvac_code, fuel_code)."""
    s = (hvac_label or "").lower()
    fuel_code = "Electric" if (fuel_ui or "").lower().startswith("electric") else "Natural Gas"
    b = (building or "").lower()

    if b == "office":
        if "electric reheat" in s: return "PVAV_Elec", "Electric"
        if "hydronic reheat" in s and "packaged" in s: return "PVAV_Gas", "Natural Gas"
        if "built-up vav" in s: return "VAV", fuel_code
        return "VAV", fuel_code

    if b == "school":
        if "fan coil" in s: return "FCU", fuel_code
        if "vav" in s: return "VAV", fuel_code
        return "VAV", fuel_code

    if b == "hotel":
        if "ptac" in s: return "PTAC", "Electric"
        if "pthp" in s: return "PTHP", "Electric"
        if "fan coil" in s: return "FCU", fuel_code
        return "FCU", fuel_code

    if b == "hospital":
        return "VAV", fuel_code

    if b.startswith("multi"):
        if "ptac" in s: return "PTAC", "Electric"
        if "fan coil" in s: return "FCU", fuel_code
        return "PTAC", "Electric"

    return "VAV", fuel_code

def _office_keys(base_glz: str, csw: str, size: str, hvac_code: str, fuel: str, hours: float):
    # Excel buckets: 2080, 2912, 8760 → pick low/high and α for interpolation
    high = 8760 if hours > 2912 else 2912
    low  = 2912 if high == 8760 else 2080
    alpha = 0.0 if high == low else max(0.0, min(1.0, (hours - low) / (high - low)))
    mk = lambda H: f"{base_glz}{csw}{size}Office{hvac_code}{fuel}{int(H)}"
    return mk(low), mk(high), alpha

def _school_key(base_glz: str, csw: str, subcode: str, hvac_code: str, fuel: str):
    return f"{base_glz}{csw}{subcode}{hvac_code}{fuel}"

def _hospital_key(base_glz: str, csw: str, hvac_code: str, fuel: str):
    return f"{base_glz}{csw}Hosp{hvac_code}{fuel}"

def _hotel_key(base_glz: str, csw: str, size: str, hvac_code: str, fuel: str, pthp_band: str, occ_bucket: int):
    return f"{base_glz}{csw}{size}Hotel{hvac_code}{fuel}{pthp_band}{occ_bucket}"

def _mf_key(base_glz: str, csw: str, size_code: str, hvac_code: str, fuel: str):
    return f"{base_glz}{csw}{size_code}MF{hvac_code}{fuel}"

# ---------------- Dataclasses and exact compute ----------------
@dataclass
class ComputeInputs:
    building_type: str
    sub_building_type: Optional[str]
    area_total_sf: float
    floors: int
    annual_hours: float
    existing_window: str            # "Single pane" / "Double pane"
    csw_type: str                   # "Single" / "Double"
    hvac_label: str                 # UI label
    heating_fuel: str               # "Natural Gas" / "Electric" / "None"
    cooling_installed: str          # "Yes" / "No"
    csw_installed_sf: float
    electric_rate: float
    gas_rate: float
    infiltration_included: Optional[str] = None   # MF: "Included" / "Excluded"
    school_level: Optional[str] = None            # "Primary School" / "Secondary School"
    hotel_occupancy: Optional[float] = None      # (Excel bucket fixed at 100)
    hdd: Optional[float] = None                  # for PTHP band
    cdd: Optional[float] = None

@dataclass
class ComputeResult:
    elec_kwh_yr: float
    gas_therm_yr: float
    cost_savings_usd: float
    eui_base: Optional[float]
    eui_csw: Optional[float]
    eui_savings_kbtusf: Optional[float]
    details: Dict[str, float]

def compute_savings_exact(inp: ComputeInputs,
                          savings_lookup: Optional[pd.DataFrame] = None) -> ComputeResult:
    """Excel-accurate computation using savings_lookup.csv (no Excel workbook required)."""
    sl = savings_lookup if savings_lookup is not None else load_savings_lookup()

    bt = inp.building_type.strip()
    base_glz = "Single" if "single" in (inp.existing_window or "").lower() else "Double"
    csw = (inp.csw_type or "Single").strip().title()

    hvac_code, fuel_code = _hvac_label_to_code(bt, inp.hvac_label, inp.heating_fuel)
    # Excel treats "None" heat as Electric for savings accounting
    if (inp.heating_fuel or "").strip().lower() == "none":
        fuel_code = "Electric"

    # Cooling adjust
    cooling_no = str(inp.cooling_installed or "").strip().lower() == "no"
    cool_adjust = 1.0

    if bt == "Office":
        is_large = (inp.area_total_sf > 30000) and ("built-up vav" in (inp.hvac_label or "").lower())
        size = "Large" if is_large else "Mid"
        lo_key, hi_key, alpha = _office_keys(base_glz, csw, size, hvac_code, fuel_code, float(inp.annual_hours or 0))
        lo = sl[sl["LookKey"] == lo_key].iloc[:1]
        hi = sl[sl["LookKey"] == hi_key].iloc[:1]
        if lo.empty or hi.empty:
            raise KeyError(f"Office lookup failed for keys {lo_key} / {hi_key}")
        s_lo, t_lo, u_lo = lo["ElecHeat_kWh_SF"].item(), lo["Cool_kWh_SF"].item(), lo["GasHeat_therm_SF"].item()
        s_hi, t_hi, u_hi = hi["ElecHeat_kWh_SF"].item(), hi["Cool_kWh_SF"].item(), hi["GasHeat_therm_SF"].item()
        if cooling_no and "Cool_adjust" in hi.columns:
            ca = hi["Cool_adjust"].item()
            cool_adjust = float(ca) if pd.notna(ca) else 1.0
        elec_heat_sf = s_lo + (s_hi - s_lo) * alpha
        cool_sf      = (t_lo + (t_hi - t_lo) * alpha) * cool_adjust
        gas_heat_sf  = u_lo + (u_hi - u_lo) * alpha
        eui_base = (lo["Base_EUI"].item() + (hi["Base_EUI"].item() - lo["Base_EUI"].item()) * alpha) if "Base_EUI" in lo and "Base_EUI" in hi else None
        eui_csw  = (lo["CSW_EUI"].item()  + (hi["CSW_EUI"].item()  - lo["CSW_EUI"].item())  * alpha) if "CSW_EUI" in lo and "CSW_EUI" in hi else None

    elif bt == "School":
        subcode = "SS" if (inp.school_level or "").lower().startswith("secondary") else "PS"
        key = _school_key(base_glz, csw, subcode, hvac_code, fuel_code)
        row = sl[sl["LookKey"] == key].iloc[:1]
        if row.empty:
            raise KeyError(f"School lookup failed for key {key}")
        r = row.iloc[0]
        if cooling_no:
            cool_adjust = float(r.get("Cool_adjust", 1.0) or 1.0)
        elec_heat_sf = float(r["ElecHeat_kWh_SF"] or 0.0)
        cool_sf      = float(r["Cool_kWh_SF"] or 0.0) * cool_adjust
        gas_heat_sf  = float(r["GasHeat_therm_SF"] or 0.0)
        eui_base     = float(r.get("Base_EUI")) if "Base_EUI" in r else None
        eui_csw      = float(r.get("CSW_EUI")) if "CSW_EUI" in r else None

    elif bt == "Hotel":
        size = "Small" if hvac_code in ("PTAC","PTHP") else "Large"
        if hvac_code in ("PTAC","PTHP"):
            fuel_code = "Electric"
        pthp_band = "High" if (hvac_code == "PTHP" and float(inp.hdd or 0) > 7999) else ("Low" if hvac_code == "PTHP" else "")
        occ_bucket = 100  # workbook uses 100
        key = _hotel_key(base_glz, csw, size, hvac_code, fuel_code, pthp_band, occ_bucket)
        row = sl[sl["LookKey"] == key].iloc[:1]
        if row.empty:
            raise KeyError(f"Hotel lookup failed for key {key}")
        r = row.iloc[0]
        if cooling_no:
            cool_adjust = float(r.get("Cool_adjust", 1.0) or 1.0)
        elec_heat_sf = float(r["ElecHeat_kWh_SF"] or 0.0)
        cool_sf      = float(r["Cool_kWh_SF"] or 0.0) * cool_adjust
        gas_heat_sf  = float(r["GasHeat_therm_SF"] or 0.0)
        eui_base     = float(r.get("Base_EUI")) if "Base_EUI" in r else None
        eui_csw      = float(r.get("CSW_EUI")) if "CSW_EUI" in r else None

    elif bt == "Hospital":
        key = _hospital_key(base_glz, csw, hvac_code, fuel_code)
        row = sl[sl["LookKey"] == key].iloc[:1]
        if row.empty:
            raise KeyError(f"Hospital lookup failed for key {key}")
        r = row.iloc[0]
        if cooling_no:
            cool_adjust = float(r.get("Cool_adjust", 1.0) or 1.0)
        elec_heat_sf = float(r["ElecHeat_kWh_SF"] or 0.0)
        cool_sf      = float(r["Cool_kWh_SF"] or 0.0) * cool_adjust
        gas_heat_sf  = float(r["GasHeat_therm_SF"] or 0.0)
        eui_base     = float(r.get("Base_EUI")) if "Base_EUI" in r else None
        eui_csw      = float(r.get("CSW_EUI")) if "CSW_EUI" in r else None

    elif bt in ("Multi-family","Multifamily","Multi family"):
        size_code = "Low" if (inp.floors or 0) < 4 else "Mid"
        hvac_code = "PTAC" if size_code == "Low" else "FCU"
        fuel_code = "Electric" if (inp.heating_fuel or "").lower() == "none" or hvac_code == "PTAC" else inp.heating_fuel.title()
        key = _mf_key(base_glz, csw, size_code, hvac_code, fuel_code)
        row = sl[sl["LookKey"] == key].iloc[:1]
        if row.empty:
            raise KeyError(f"Multi-family lookup failed for key {key}")
        r = row.iloc[0]
        infil_heat = infil_cool = 1.0
        if (inp.infiltration_included or "").strip().lower() == "excluded":
            infil_heat = float(r.get("Infil_Heat_Factor", 1.0) or 1.0)
            infil_cool = float(r.get("Infil_Cool_Factor", 1.0) or 1.0)
        if cooling_no:
            cool_adjust = float(r.get("Cool_adjust", 1.0) or 1.0)
        elec_heat_sf = float(r["ElecHeat_kWh_SF"] or 0.0) * infil_heat
        cool_sf      = float(r["Cool_kWh_SF"] or 0.0) * infil_cool * cool_adjust
        gas_heat_sf  = float(r["GasHeat_therm_SF"] or 0.0) * infil_heat
        eui_base     = float(r.get("Base_EUI")) if "Base_EUI" in r else None
        eui_csw      = float(r.get("CSW_EUI")) if "CSW_EUI" in r else None

    else:
        raise ValueError(f"Unsupported building type: {bt}")

    # Totals based on CSW installed area (fallback to total area if missing)
    area_csw = float(inp.csw_installed_sf or 0.0)
    if area_csw <= 0:
        area_csw = float(inp.area_total_sf or 0.0)

    kwh_total   = (elec_heat_sf + cool_sf) * area_csw
    therm_total = (gas_heat_sf) * area_csw

    eui_savings_kbtusf = ((kwh_total * 3.413 + therm_total * 100.0) / float(inp.area_total_sf)) if float(inp.area_total_sf or 0) > 0 else None
    cost_savings = kwh_total * float(inp.electric_rate or 0.0) + therm_total * float(inp.gas_rate or 0.0)

    details = {
        "elec_heat_kwh_per_sf": float(elec_heat_sf or 0.0),
        "cool_kwh_per_sf": float(cool_sf or 0.0),
        "gas_heat_therm_per_sf": float(gas_heat_sf or 0.0),
        "cool_adjust_applied": float(cool_adjust or 1.0),
    }
    return ComputeResult(
        elec_kwh_yr=float(kwh_total or 0.0),
        gas_therm_yr=float(therm_total or 0.0),
        cost_savings_usd=float(cost_savings or 0.0),
        eui_base=float(eui_base) if eui_base is not None else None,
        eui_csw=float(eui_csw) if eui_csw is not None else None,
        eui_savings_kbtusf=float(eui_savings_kbtusf) if eui_savings_kbtusf is not None else None,
        details=details,
    )

# ---------------- Back-compat wrapper (what your app already calls) ----------------
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
    floors: Optional[int] = None,
    csw_installed_sf: Optional[float] = None,
    electric_rate: Optional[float] = None,
    gas_rate: Optional[float] = None,
) -> Dict[str, float]:
    """
    Wrapper preserving the old name/signature, now powered by savings_lookup.csv.
    This eliminates any dependency on the Excel workbook.
    """
    inp = ComputeInputs(
        building_type=building_type,
        sub_building_type=sub_building_type,
        area_total_sf=float(building_area_sf or 0.0),
        floors=int(floors or 0),
        annual_hours=float(annual_operating_hours or 0.0),
        existing_window=(existing_window_type_ui or ""),
        csw_type=(csw_glazing_ui or "Single"),
        hvac_label=(hvac_ui or ""),
        heating_fuel=(heating_fuel_ui or ""),
        cooling_installed="No" if not cooling_installed else "Yes",
        csw_installed_sf=float(csw_installed_sf or 0.0),
        electric_rate=float(electric_rate or 0.0),
        gas_rate=float(gas_rate or 0.0),
        infiltration_included=("Excluded" if include_infiltration is False else "Included" if include_infiltration else None),
        school_level=sub_building_type if (building_type == "School") else None,
        hotel_occupancy=hotel_occupancy_pct,
        hdd=float(weather_hdd or 0.0),
        cdd=float(weather_cdd or 0.0),
    )

    res = compute_savings_exact(inp)

    # Return in the same dict shape your app previously used
    return {
        "elec_heat_kwh_per_sf": res.details["elec_heat_kwh_per_sf"],
        "cool_kwh_per_sf":      res.details["cool_kwh_per_sf"],
        "gas_heat_therm_per_sf":res.details["gas_heat_therm_per_sf"],
        "total_kwh":            res.elec_kwh_yr,
        "total_therms":         res.gas_therm_yr,
        "cost_savings_usd":     res.cost_savings_usd,
        "eui_base":             res.eui_base,
        "eui_csw":              res.eui_csw,
        "eui_savings_kbtusf":   res.eui_savings_kbtusf,
        "cool_adjust_applied":  res.details["cool_adjust_applied"],
    }
